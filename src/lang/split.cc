/*!
 *  Copyright (c) 2016 by Contributors
 * \file split.cc
 */
#include <tvm/split.h>

namespace tvm {

Split DimSplitNode::make(Var var,
                         Expr factor) {
  auto n = std::make_shared<DimSplitNode>();
  CHECK_EQ(factor.type().lanes(), 1);
  n->var = var;
  n->factor = factor;
  return Split(n);
}

TVM_REGISTER_NODE_TYPE(DimSplitNode);

}  // namespace tvm
