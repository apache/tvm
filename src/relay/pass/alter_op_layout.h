/*!
 *  Copyright (c) 2018 by Contributors
 * \file alter_op_layout.h
 * \brief Alternate the layouts of operators or replace primitive operators with
          other expressions. This pass can be used for computing convolution in
          custom layouts or other general weight pre-transformation.
 */

#ifndef TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_
#define TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_

#include <tvm/relay/expr.h>

#include "../op/layout.h"

namespace tvm {
namespace relay {

/*!
 * \brief Infer & correct function of node layout. See \p Layout for layout convention
 * \param attrs The attribute of the node.
 * \param in_layouts The layouts of input arguments.
 * \param in_shapes The shapes of input arguments.
 * \return infered_layout An array of two elements that are inferred input layouts and
 *                        inferred output layouts.
 */
using FInferCorrectLayout = runtime::TypedPackedFunc<
    Array<Array<Layout>>(const Attrs& attrs,
                         const Array<Layout>& in_layouts,
                         const Array<Array<IndexExpr>> &in_shapes)>;

/*! \brief take arbitrary input layout and copy to output */
inline Array<Array<Layout> > ElemwiseArbitraryLayout(const Attrs& attrs,
                                                     const Array<Layout>& in_layouts,
                                                     const Array<Array<IndexExpr> > &in_shapes) {
  Array<Layout> inferred_ins;

  Layout in;
  for (size_t i = 0; i < in_layouts.size(); ++i) {
    if (!in.defined()) in = in_layouts[i];
    CHECK(in.Equals(in_layouts[i]))
      << "Incompatible layout at " << i << "-th input: expected " << in
      << ", got " << in_layouts[i];
  }
  for (size_t i = 0; i < in_layouts.size(); ++i) {
    inferred_ins.push_back(in);
  }

  return Array<Array<Layout> >{inferred_ins, {in}};
}

/*! \brief Infer layout for binary broadcast operators */
inline Array<Array<Layout> > BinaryBroadcastLayout(const Attrs& attrs,
                                                   const Array<Layout>& in_layouts,
                                                   const Array<Array<IndexExpr> > &in_shapes) {
  CHECK_EQ(in_layouts.size(), 2);
  CHECK_EQ(in_shapes.size(), 2);

  Array<Layout> layouts = in_layouts;

  if (!layouts[0].defined() && !layouts[1].defined()) {
    // both undefined, infer fails
    return Array<Array<Layout> > {{Layout::Undef()}, {Layout::Undef()}};
  } else if (!layouts[0].defined() || !layouts[1].defined()) {
    // only one is defined, use shape information to help infer
    int defined_idx = layouts[0].defined() ? 0 : 1;
    int undef_idx = 1 - defined_idx;

    if (in_shapes[defined_idx].size() >= in_shapes[undef_idx].size()) {
      layouts.Set(undef_idx,
                  layouts[defined_idx].Sublayout(
                      in_shapes[defined_idx].size() - in_shapes[undef_idx].size(),
                      in_shapes[undef_idx].size()));
      return Array<Array<Layout> > {layouts, {layouts[defined_idx]}};
    } else {
      // only know the tensor with smaller dimensions,
      // so we cannot infer the final broadcasted output.
      // fails in this case.
      return Array<Array<Layout> > {{Layout::Undef()}, {Layout::Undef()}};
    }
  } else {
    // try to broadcast to the tensors to the larger dimension
    int large_idx = layouts[0].ndim_super() >= layouts[1].ndim_super() ? 0 : 1;
    int small_idx = 1 - large_idx;
    Layout ret = layouts[large_idx];

    // extract common part
    size_t i = layouts[large_idx].ndim();
    for (; i != 0; --i) {
      auto dim = layouts[large_idx][i-1];
      if (!layouts[small_idx].Contains(Layout::ToSuperdim(dim))) {
        break;
      }
    }

    Layout common_part = layouts[large_idx].Sublayout(i, layouts[large_idx].ndim() - i);
    if (!layouts[small_idx].Convertible(common_part)) {  // fail
      return Array<Array<Layout> > {{Layout::Undef()}, {Layout::Undef()}};
    }

    layouts.Set(small_idx, common_part);
    return Array<Array<Layout> > {layouts, {ret}};
  }
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_
