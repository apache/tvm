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
 * \param input_layout The input layouts.
 * \return infered_layout An array of two elements that are inferred input layouts and
 *                        inferred output layouts.
 */
using FInferCorrectLayout = runtime::TypedPackedFunc<
    Array<Array<Layout>>(const Attrs& attrs,
                         const Array<Layout>& in_layouts)>;

/*! \brief take arbitrary input layout and copy to output */
inline Array<Array<Layout> > ElemwiseArbitraryLayout(const Attrs& attrs,
                                                     const Array<Layout>& in_layouts) {
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

/*! \brief Infer layout for binary broadcast operators. Prior to keep left layout */
inline Array<Array<Layout> > BinaryBroadcastLayout(const Attrs& attrs,
                                                   const Array<Layout>& in_layouts) {
  CHECK_EQ(in_layouts.size(), 2);
  Layout lhs = in_layouts[0];
  Layout rhs = in_layouts[1];

  // prior to keep left layout
  if (!lhs.defined()) {
    lhs = rhs;
  }

  return Array<Array<Layout> > {{lhs, lhs}, {lhs}};
}

}  //  namespace relay
}  //  namespace tvm

#endif  // TVM_RELAY_PASS_ALTER_OP_LAYOUT_H_
