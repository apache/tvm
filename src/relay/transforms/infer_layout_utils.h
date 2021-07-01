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
 * \file infer_layout_utils.h
 * \brief Utility functions to alter the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */

#ifndef TVM_RELAY_TRANSFORMS_INFER_LAYOUT_UTILS_H_
#define TVM_RELAY_TRANSFORMS_INFER_LAYOUT_UTILS_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <string>
#include <tuple>
#include <utility>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * \brief Returns a new layout where the subordinate factors are adjusted based on the tensor
 *        shape.
 * \param old_layout The old layout before any transformation.
 * \param old_shape The shape of the original tensor.
 * \return The adjusted Layout.
 */
inline Layout AdjustSubordinateFactors(const Layout& src_layout, const Layout& old_layout,
                                       const Array<tvm::PrimExpr>& old_shape) {
  // For each subordinate axis
  //   1) Find the corresponding dual axis.
  //   2) Find the Index of this dual axis in old_layout.
  //   3) Find the shape of the that axis in old_shape.
  //   4) a) Adjust factor to 1, if that shape is 1. b) Else retain the factor.
  std::string new_layout;
  for (auto axis : src_layout->axes) {
    if (!LayoutAxis::Get(axis).IsPrimal()) {
      // 1) Find the corresponding dual axis
      const auto& dual_axis = LayoutAxis::Get(axis).ToPrimal();

      // 2) Find the index of this dual axis in old_layout
      int old_axis = old_layout.IndexOf(dual_axis);

      // 3) Find the shape of this index in old_shape
      auto shape_val = old_shape[old_axis];

      // 4) a) Check if this shape element is 1.
      bool is_shape_one = false;
      if (auto* shape_int = shape_val.as<IntImmNode>()) {
        if (shape_int->value == 1) {
          new_layout += "1";
          is_shape_one = true;
        }
      }

      // 4) b) If shape is not 1, retain the factor.
      if (!is_shape_one) {
        auto new_shape_val = src_layout.FactorOf(dual_axis);
        new_layout += std::to_string(new_shape_val);
      }
    }
    new_layout += LayoutAxis::Get(axis).name();
  }
  return Layout(new_layout);
}

/*
 * \brief An output structure to hold results from FInferCorrectLayout calls.
 * \tparam input_layouts Inferred input layouts.
 * \tparam output_layouts Inferred output layouts.
 * \tparam new_attrs Updated attributes consistent with inferred layouts.
 */
class InferCorrectLayoutOutputNode : public Object {
 public:
  Array<Layout> input_layouts;
  Array<Layout> output_layouts;
  Attrs new_attrs;
  TVM_DECLARE_BASE_OBJECT_INFO(InferCorrectLayoutOutputNode, Object);
};

class InferCorrectLayoutOutput : public ObjectRef {
 public:
  InferCorrectLayoutOutput(Array<Layout> input_layouts, Array<Layout> output_layouts,
                           Attrs new_attrs) {
    auto n = make_object<InferCorrectLayoutOutputNode>();
    n->input_layouts = std::move(input_layouts);
    n->output_layouts = std::move(output_layouts);
    n->new_attrs = std::move(new_attrs);
    data_ = n;
  }
  TVM_DEFINE_OBJECT_REF_METHODS(InferCorrectLayoutOutput, ObjectRef, InferCorrectLayoutOutputNode);
};

/*!
 * \brief Infer & correct function of node layout. See \p Layout for layout convention
 * \param attrs The attribute of the node.
 * \param new_in_layouts The layouts of input arguments after alter_op_layout.
 *                       This can be undefined, which means we call this function before alternating
 *                       any operators.
 * \param old_in_layouts The layouts of input arguments before alter_op_layout.
 * \param old_in_types The types of old input arguments.
 * \return infer_layout_output Inferred layouts and updated attributes stored in
 *                             InferCorrectLayoutOutput above.
 */
using FInferCorrectLayout = runtime::TypedPackedFunc<InferCorrectLayoutOutput(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types)>;

inline InferCorrectLayoutOutput ElemwiseArbitraryLayout(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  Layout ret;

  if (new_in_layouts.defined()) {
    ICHECK_GE(new_in_layouts.size(), 1);
    ret = new_in_layouts[0];
  } else {
    for (size_t i = 0; i < old_in_layouts.size(); ++i) {
      if (old_in_layouts[i].defined()) {
        ret = old_in_layouts[i];
        break;
      }
    }
  }

  return InferCorrectLayoutOutput(Array<Layout>(old_in_layouts.size(), ret), {ret}, attrs);
}

inline std::pair<Array<Layout>, Array<Layout>> BinaryBroadcastLayoutHelper(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  Array<Layout> layouts;
  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }

  if (new_in_layouts.defined()) {
    layouts.Assign(new_in_layouts.begin(), new_in_layouts.end());
  } else {
    layouts.Assign(old_in_layouts.begin(), old_in_layouts.end());
  }

  std::pair<Array<Layout>, Array<Layout>> out_default{{Layout::Undef(), Layout::Undef()},
                                                      {Layout::Undef()}};

  if (!layouts[0].defined() && !layouts[1].defined()) {
    // both undefined, infer fails
    return out_default;
  } else if (!layouts[0].defined() || !layouts[1].defined()) {
    // only one is defined, use shape information to help infer
    int defined_idx = layouts[0].defined() ? 0 : 1;
    int undef_idx = 1 - defined_idx;

    if (old_in_shapes[defined_idx].size() >= old_in_shapes[undef_idx].size()) {
      layouts.Set(undef_idx, layouts[defined_idx].SubLayout(old_in_shapes[defined_idx].size() -
                                                                old_in_shapes[undef_idx].size(),
                                                            old_in_shapes[undef_idx].size()));
      return {layouts, {layouts[defined_idx]}};
    } else {
      // only know the tensor with smaller dimensions,
      // so we cannot infer the final broadcasted output.
      // fails in this case.
      return out_default;
    }
  } else if (layouts[0].defined() && layouts[1].defined() &&
             (layouts[0].ndim() == 0 || layouts[1].ndim() == 0)) {
    int scalar = layouts[0].ndim() == 0 ? 0 : 1;
    return {layouts, {layouts[1 - scalar]}};
  } else {
    // Set the layout of the larger dimension. If one dimension size is lower, we call expand dims
    // while transforming layout.
    int large_idx = layouts[0].ndim_primal() >= layouts[1].ndim_primal() ? 0 : 1;
    int small_idx = 1 - large_idx;
    Layout ret = layouts[large_idx];

    if (old_in_layouts[0].Equals(old_in_layouts[1])) {
      // Support scenarios where original operands were of type [N, H, W, C] and [N, H, W, 1]
      // In this case, we might have NCHW16c coming for 1 operand. However, the other operand does
      // not have enough C dimension. To reuse broadcasting, we would want to use NCHW1c for the
      // second operand. The following section of code walks through the layouts and shapes to
      // perform that operation.
      // a in NCHWC16c
      // b in NHW1
      // b = layout_transform(b) from NHW1 -> NCHW1c
      // add(a, b)
      auto old_small_shape = old_in_shapes[small_idx];
      auto old_small_layout = old_in_layouts[small_idx];
      auto new_small_layout =
          AdjustSubordinateFactors(layouts[large_idx], old_small_layout, old_small_shape);
      layouts.Set(small_idx, new_small_layout);
    } else {
      // Support scenarios where original operands were of type [N, H, W, C] and [C]. In this case,
      // while transforming the layout, we expand dims to make C go to NHWC, and then use the
      // modified layout of the first operator to call the layout transform. E.g.
      // a in NCHWC16c
      // b in C
      // b = expand_dims(b) from C -> NHWC
      // b = layout_transform(b) from NHWC -> NCHW16c
      // add(a, b)
      layouts.Set(small_idx, ret);
    }
    return {layouts, {ret}};
  }
}

/*! \brief Infer layout for binary broadcast operators */
inline InferCorrectLayoutOutput BinaryBroadcastLayout(const Attrs& attrs,
                                                      const Array<Layout>& new_in_layouts,
                                                      const Array<Layout>& old_in_layouts,
                                                      const Array<tvm::relay::Type>& old_in_types) {
  auto inferred_layout =
      BinaryBroadcastLayoutHelper(attrs, new_in_layouts, old_in_layouts, old_in_types);
  return InferCorrectLayoutOutput(inferred_layout.first, inferred_layout.second, attrs);
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_TRANSFORMS_INFER_LAYOUT_UTILS_H_
