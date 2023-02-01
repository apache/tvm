/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/op/tensor/transform.h
 * \brief Transform op attributes that can be shared among Relay and its dialects.
 */
#ifndef TVM_RELAY_OP_TENSOR_TRANSFORM_H_
#define TVM_RELAY_OP_TENSOR_TRANSFORM_H_

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/error.h>
#include <tvm/relay/op_attr_types.h>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../make_op.h"

namespace tvm {
namespace relay {

template <typename AttrType>
bool ConcatenateRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  // types: [data, result]
  ICHECK_EQ(types.size(), 2) << "the arity of concatenate is 2, not " << types.size();
  /* If we receive a tuple we can continue, if we receive
   * anything but an incomplete type we should signal an
   * error.
   */
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    reporter->GetDiagCtx().EmitFatal(
        Diagnostic::Error(reporter->GetSpan())
        << "concatenate requires a tuple of tensors as the first argument, found "
        << PrettyPrint(types[0]));
    return false;
  } else if (types[0].as<IncompleteTypeNode>() != nullptr) {
    return false;
  }

  const auto* param = attrs.as<AttrType>();
  if (param == nullptr) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "the call attributes are not defined");
    return false;
  }

  if (tensor_tuple->fields[0].as<IncompleteTypeNode>()) {
    return false;
  }
  const auto& first = Downcast<TensorType>(tensor_tuple->fields[0]);
  // Sanity check: ndim and dtype.
  const int ndim = static_cast<int>(first->shape.size());
  const DataType dtype = first->dtype;

  // Sanity check: axis
  int axis = param->axis;
  if (!(-ndim <= axis && axis < ndim)) {
    throw CompileError(ErrorBuilder() << "concatenate only accepts `axis` in [-ndim, ndim)"
                                      << ", but got axis = " << axis << ", and ndim = " << ndim);
  }
  axis = axis < 0 ? ndim + axis : axis;

  for (const Type& ele : tensor_tuple->fields) {
    if (ele.as<IncompleteTypeNode>()) {
      return false;
    }

    const auto& e = Downcast<TensorType>(ele);

    int e_ndim = static_cast<int>(e->shape.size());
    const DataType& e_dtype = e->dtype;
    if (e_ndim != ndim) {
      throw Error("relay.concatenate requires all tensors have the same ndim");
    }
    if (e_dtype != dtype) {
      throw Error("relay.concatenate requires all tensors have the same dtype");
    }
  }

  // Calculate shape
  std::vector<IndexExpr> oshape(ndim);
  const size_t data_length = tensor_tuple->fields.size();

  // Accumulate the concat axis output dim or decide if this is dynamic concat
  bool is_dynamic_concat = false;
  std::vector<TensorType> input_tensors;
  IndexExpr concat_output_dim = first->shape[axis];
  for (size_t i = 0; i < data_length; ++i) {
    const auto& e = Downcast<TensorType>(tensor_tuple->fields[i]);
    input_tensors.push_back(e);
    if (e->shape[axis].as<AnyNode>()) {
      is_dynamic_concat = true;
      concat_output_dim = Any();
    } else if (i > 0 && !is_dynamic_concat) {
      // accumulate axis dimension
      concat_output_dim += e->shape[axis];
    }
  }

  oshape[axis] = concat_output_dim;

  for (int i = 0; i < ndim; ++i) {
    if (i == axis) {
      // The concat axis is already handled above.
      // The rest of the body sets the output shape for non-concat axes
      continue;
    }
    std::vector<IndexExpr> non_any;
    for (size_t j = 0; j < data_length; ++j) {
      const auto& e = input_tensors[j];
      if (!e->shape[i].as<AnyNode>()) {
        non_any.push_back(e->shape[i]);
      }
    }
    size_t non_any_size = non_any.size();
    for (size_t k = 1; k < non_any_size; k++) {
      if (reporter->AssertEQ(non_any[0], non_any[k])) continue;
      throw Error(
          "relay.concatenate requires all tensors have the same shape "
          "on non-concatenating axes");
    }

    if (non_any_size == data_length) {
      // All static case
      oshape[i] = non_any[0];
    } else if (non_any_size > 0 && is_dynamic_concat) {
      // For non-concat axes, we want to enforce static shape constraint.
      // However, if the concat axis is static, the output shape would become static while
      // the input could be partially static/dynamic. To prevent runtime segfaults due to the lack
      // of runtime input shape checking for such cases, static shape constraint is only enforced
      // when the output concat axis is dynamic.
      //
      // Examples (both concat on the first axis):
      // * [(?, 3), (?, ?)] -> (?, 3)
      // * [(1, 3), (1, ?)] -> (2, ?)
      oshape[i] = non_any[0];
    } else {
      oshape[i] = Any();
    }
  }

  auto rtype = TensorType(oshape, dtype);
  reporter->Assign(types[1], rtype);
  return true;
}

static inline InferCorrectLayoutOutput ConcatenateLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<ConcatenateAttrs>();
  ICHECK(attrs_ptr);
  ObjectPtr<ConcatenateAttrs> param = make_object<ConcatenateAttrs>(*attrs_ptr);

  Array<Array<IndexExpr>> old_in_shapes;
  ICHECK_EQ(old_in_types.size(), 1);
  for (auto old_in_tuple_t : old_in_types) {
    ICHECK(old_in_tuple_t.as<TupleTypeNode>());
    for (auto old_in_t : old_in_tuple_t.as<TupleTypeNode>()->fields) {
      old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
    }
  }

  size_t axis =
      param->axis < 0 ? param->axis + old_in_shapes[0].size() : static_cast<size_t>(param->axis);

  Layout ret;
  bool is_new_layout_selected = false;
  if (new_in_layouts.defined()) {  // this function is called after some operators are alternated.
    // If all the new input layouts are same, the new in layout gets selected.  For axis, the new
    // axis in the new layout is identified. The param->axis is then modified on the fly to conform
    // to the new input layout.
    const auto& concate_dim = old_in_layouts[0][axis];
    bool all_input_layouts_same = true;
    for (auto new_layout : new_in_layouts) {
      if (!new_layout.Equals(new_in_layouts[0])) {
        all_input_layouts_same = false;
      }
    }
    if (all_input_layouts_same) {
      auto new_index = new_in_layouts[0].IndexOf(concate_dim);
      ret = new_in_layouts[0];
      param->axis = new_index;
      is_new_layout_selected = true;
    }
  }

  if (!is_new_layout_selected) {
    // this function is called on the original correct relay ir
    for (size_t i = 0; i < old_in_layouts.size(); ++i) {
      if (old_in_layouts[i].defined()) {
        ret = old_in_layouts[i];
        break;
      }
    }

    if (ret.ndim() <= axis || !ret[axis].IsPrimal()) {
      return InferCorrectLayoutOutput({Layout::Undef()}, {Layout::Undef()}, attrs);
    }
  }

  return InferCorrectLayoutOutput(Array<Layout>(old_in_layouts.size(), ret), {ret}, Attrs(param));
}

/*!
 * \brief Infer output shape for reshape.
 *
 * \param data_shape The input data shape.
 * \param attrs The attributes.
 * \param reverse Whether to reverse the indices.
 * \return Output shape.
 */
Array<IndexExpr> InferNewShape(const Array<IndexExpr>& data_shape, const Attrs& attrs,
                               bool reverse);

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_TENSOR_TRANSFORM_H_
