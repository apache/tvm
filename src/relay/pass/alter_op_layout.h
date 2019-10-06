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
 *  Copyright (c) 2018 by Contributors
 * \file alter_op_layout.h
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */

#ifndef TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_
#define TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_

#include <tvm/data_layout.h>
#include <tvm/relay/expr.h>
#include <string>

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
                                       const Array<tvm::Expr>& old_shape) {
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
      if (auto* shape_int = shape_val.as<IntImm>()) {
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

/*!
 * \brief Infer & correct function of node layout. See \p Layout for layout convention
 * \param attrs The attribute of the node.
 * \param new_in_layouts The layouts of input arguments after alter_op_layout.
 *                       This can be undefined, which means we call this function before alternating
 *                       any operators.
 * \param old_in_layouts The layouts of input arguments before alter_op_layout.
 * \param old_in_shapes The shapes of old input arguments.
 * \return infered_layout An array of two elements that are inferred input layouts and
 *                        inferred output layouts.
 */
using FInferCorrectLayout = runtime::TypedPackedFunc<
    Array<Array<Layout>>(const Attrs& attrs,
                         const Array<Layout>& new_in_layouts,
                         const Array<Layout>& old_in_layouts,
                         const Array<Array<IndexExpr>> &old_in_shapes)>;

/*! \brief take arbitrary input layout and copy to output */
inline Array<Array<Layout> > ElemwiseArbitraryLayout(const Attrs& attrs,
                                                     const Array<Layout>& new_in_layouts,
                                                     const Array<Layout>& old_in_layouts,
                                                     const Array<Array<IndexExpr>> &old_in_shapes) {
  Layout ret;

  if (new_in_layouts.defined()) {
    CHECK_GE(new_in_layouts.size(), 1);
    ret = new_in_layouts[0];
  } else {
    for (size_t i = 0; i < old_in_layouts.size(); ++i) {
      if (old_in_layouts[i].defined()) {
        ret = old_in_layouts[i];
        break;
      }
    }
  }

  return Array<Array<Layout> >{Array<Layout>(old_in_layouts.size(), ret), {ret}};
}

/*! \brief Infer layout for binary broadcast operators */
inline Array<Array<Layout> > BinaryBroadcastLayout(const Attrs& attrs,
                                                   const Array<Layout>& new_in_layouts,
                                                   const Array<Layout>& old_in_layouts,
                                                   const Array<Array<IndexExpr>> &old_in_shapes) {
  Array<Layout> layouts;

  if (new_in_layouts.defined()) {
    layouts.assign(new_in_layouts.begin(), new_in_layouts.end());
  } else {
    layouts.assign(old_in_layouts.begin(), old_in_layouts.end());
  }

  if (!layouts[0].defined() && !layouts[1].defined()) {
    // both undefined, infer fails
    return Array<Array<Layout> > {{Layout::Undef()}, {Layout::Undef()}};
  } else if (!layouts[0].defined() || !layouts[1].defined()) {
    // only one is defined, use shape information to help infer
    int defined_idx = layouts[0].defined() ? 0 : 1;
    int undef_idx = 1 - defined_idx;

    if (old_in_shapes[defined_idx].size() >= old_in_shapes[undef_idx].size()) {
      layouts.Set(undef_idx,
                  layouts[defined_idx].SubLayout(
                      old_in_shapes[defined_idx].size() - old_in_shapes[undef_idx].size(),
                      old_in_shapes[undef_idx].size()));
      return Array<Array<Layout> >{layouts, {layouts[defined_idx]}};
    } else {
      // only know the tensor with smaller dimensions,
      // so we cannot infer the final broadcasted output.
      // fails in this case.
      return Array<Array<Layout> >{{Layout::Undef()}, {Layout::Undef()}};
    }
  } else if (layouts[0].defined() && layouts[1].defined() &&
            (layouts[0].ndim() == 0 || layouts[1].ndim() == 0)) {
    int scalar = layouts[0].ndim() == 0 ? 0 : 1;
    return Array<Array<Layout> >{layouts, {layouts[1-scalar]}};
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
    return Array<Array<Layout>>{layouts, {ret}};
  }
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_
