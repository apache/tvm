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

namespace tvm {
namespace relay {

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
    // try to broadcast the tensors to the larger dimension
    int large_idx = layouts[0].ndim_primal() >= layouts[1].ndim_primal() ? 0 : 1;
    int small_idx = 1 - large_idx;
    Layout ret = layouts[large_idx];

    // extract common part
    size_t i = layouts[large_idx].ndim();
    for (; i != 0; --i) {
      const auto& axis = layouts[large_idx][i-1];
      if (!layouts[small_idx].Contains(axis.ToPrimal())) {
        break;
      }
    }

    Layout common_part = layouts[large_idx].SubLayout(i, layouts[large_idx].ndim() - i);
    if (!BijectiveLayoutNode::make(layouts[small_idx], common_part).defined()) {
      // not convertible
      return Array<Array<Layout> > {{Layout::Undef()}, {Layout::Undef()}};
    }

    layouts.Set(small_idx, common_part);
    return Array<Array<Layout> > {layouts, {ret}};
  }
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_
