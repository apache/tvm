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

#include <tvm/arith/analyzer.h>
#include <tvm/ir/error.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../make_op.h"

namespace tvm {
namespace relay {

template <typename AttrType>
bool ConcatenateRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  // types: [data, result]
  CHECK_EQ(types.size(), 2);
  /* If we receive a tuple we can continue, if we receive
   * anything but an incomplete type we should signal an
   * error.
   */
  const auto* tensor_tuple = types[0].as<TupleTypeNode>();
  if (tensor_tuple == nullptr) {
    throw Error(
        ErrorBuilder() << "concatenate requires a tuple of tensors as the first argument, found "
                       << PrettyPrint(types[0]));
  } else if (types[0].as<IncompleteTypeNode>() != nullptr) {
    return false;
  }

  const auto* param = attrs.as<AttrType>();
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
    throw Error(ErrorBuilder() << "concatenate only accepts `axis` in [-ndim, ndim)"
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
  std::vector<IndexExpr> oshape(first->shape.begin(), first->shape.end());
  int data_length = static_cast<int>(tensor_tuple->fields.size());
  for (int i = 0; i < ndim; ++i) {
    std::vector<IndexExpr> non_any;
    for (int j = 0; j < data_length; ++j) {
      const auto& e = Downcast<TensorType>(tensor_tuple->fields[j]);
      if (!e->shape[i].as<AnyNode>()) {
        non_any.push_back(e->shape[i]);
        // accumulate axis dimension
        if (j > 0 && i == axis && !oshape[i].as<AnyNode>()) {
          oshape[i] += e->shape[i];
        }
      }
    }
    int non_any_size = static_cast<int>(non_any.size());
    if (non_any_size != data_length) oshape[i] = Any();
    if (i != axis) {
      for (int k = 1; k < non_any_size; k++) {
        if (reporter->AssertEQ(non_any[0], non_any[k])) continue;
        throw Error(
            "relay.concatenate requires all tensors have the same shape "
            "on non-concatenating axes");
      }
    }
  }

  auto rtype = TensorType(oshape, dtype);
  reporter->Assign(types[1], rtype);
  return true;
}

static inline Array<Array<Layout>> ConcatenateLayout(const Attrs& attrs,
                                                     const Array<Layout>& new_in_layouts,
                                                     const Array<Layout>& old_in_layouts,
                                                     const Array<tvm::relay::Type>& old_in_types) {
  ConcatenateAttrs* param = const_cast<ConcatenateAttrs*>(attrs.as<ConcatenateAttrs>());

  Array<Array<IndexExpr>> old_in_shapes;
  CHECK_EQ(old_in_types.size(), 1);
  for (auto old_in_tuple_t : old_in_types) {
    CHECK(old_in_tuple_t.as<TupleTypeNode>());
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
      return Array<Array<Layout>>{{Layout::Undef()}, {Layout::Undef()}};
    }
  }

  return Array<Array<Layout>>{Array<Layout>(old_in_layouts.size(), ret), {ret}};
}

static inline Array<IndexExpr> infer_newshape(const Array<IndexExpr>& data_shape, 
                                              const Attrs& attrs) {
  const auto* param = attrs.as<ReshapeAttrs>();
  Array<IndexExpr> oshape;
  Array<IndexExpr> ishape;
  Array<Integer> newshape;

  if (param->reverse) {
    ishape.Assign(data_shape.rbegin(), data_shape.rend());
    newshape.Assign(param->newshape.rbegin(), param->newshape.rend());
  } else {
    ishape = data_shape;
    newshape = param->newshape;
  }

  std::unordered_set<size_t> used_input_dims;
  std::unordered_set<size_t> used_output_dims;
  size_t src_idx = 0;
  int infer_idx = -1;

  for (size_t i = 0; i < newshape.size(); ++i) {
    int svalue = newshape[i]->value;
    // special flag handling for shape inference.
    if (svalue > 0) {
      oshape.push_back(newshape[i]);
      ++src_idx;
    } else if (svalue == 0) {
      // keep same
      CHECK_LT(src_idx, ishape.size());
      used_input_dims.insert(src_idx);
      used_output_dims.insert(oshape.size());
      oshape.push_back(ishape[src_idx++]);
    } else if (svalue == -1) {
      // inference based on rest
      CHECK_LT(infer_idx, 0) << "One and only one dim can be inferred";
      infer_idx = i;
      oshape.push_back(1);
      ++src_idx;
    } else if (svalue == -2) {
      // copy all remaining dims from source
      while (src_idx < ishape.size()) {
        used_input_dims.insert(src_idx);
        used_output_dims.insert(oshape.size());
        oshape.push_back(ishape[src_idx++]);
      }
    } else if (svalue == -3) {
      // merge two dims from source
      CHECK_LT(src_idx + 1, ishape.size());
      used_input_dims.insert(src_idx);
      IndexExpr d1 = ishape[src_idx++];
      used_input_dims.insert(src_idx);
      IndexExpr d2 = ishape[src_idx++];
      used_output_dims.insert(oshape.size());
      if (d1.as<AnyNode>() || d2.as<AnyNode>()) {
        oshape.push_back(Any());
      } else {
        oshape.push_back(d1 * d2);
      }
    } else if (svalue == -4) {
      // split the source dim s into two dims
      // read the left dim and then the right dim (either can be -1)
      CHECK_LT(i + 2, newshape.size());
      CHECK_LT(src_idx, ishape.size());
      used_input_dims.insert(src_idx);
      IndexExpr d0 = ishape[src_idx++];
      Integer d1 = newshape[++i];
      Integer d2 = newshape[++i];
      if (d1->value == -1) {
        CHECK(d2->value != -1) << "Split dims cannot both be -1.";
        used_output_dims.insert(oshape.size());
        if (d0.as<AnyNode>()) {
          oshape.push_back(Any());
        } else {
          oshape.push_back(indexdiv(d0, d2));
        }
        used_output_dims.insert(oshape.size());
        oshape.push_back(d2);
      } else {
        used_output_dims.insert(oshape.size());
        oshape.push_back(d1);
        used_output_dims.insert(oshape.size());
        if (d2->value == -1) {
          if (d0.as<AnyNode>()) {
            oshape.push_back(Any());
          } else {
            oshape.push_back(indexdiv(d0, d1));
          }
        } else {
          oshape.push_back(d2);
        }
      }
    } else {
      CHECK(false) << "Unsupported special value: " << svalue;
    }
  }

  if (infer_idx >= 0) {
    IndexExpr infer_dim = 1;
    for (size_t i = 0; i < ishape.size(); ++i) {
      if (used_input_dims.count(i) != 0) {
        continue;
      }
      if (ishape[i].as<AnyNode>()) {
        infer_dim = Any();
        break;
      }
      infer_dim *= ishape[i];
    }
    if (!infer_dim.as<AnyNode>()) {
      for (size_t i = 0; i < oshape.size(); ++i) {
        if (used_output_dims.count(i) != 0) {
          continue;
        }
        if (oshape[i].as<AnyNode>()) {
          infer_dim = Any();
          break;
        }
        infer_dim = indexdiv(infer_dim, oshape[i]);
      }
    }
    arith::Analyzer ana;
    infer_dim = ana.Simplify(infer_dim);
    oshape.Set(infer_idx, infer_dim);
  }

  return oshape;
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_TENSOR_TRANSFORM_H_
