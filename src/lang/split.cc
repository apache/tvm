/*!
 *  Copyright (c) 2016 by Contributors
 * \file split.cc
 */
#include <tvm/split.h>

namespace tvm {

Split DimSplitNode::make(int dim_index,
                         Expr factor,
                         bool over_rdom) {
  auto n = std::make_shared<DimSplitNode>();
  CHECK_EQ(factor.type().lanes(), 1);
  n->split_over_rdom = over_rdom;
  n->dim_index = dim_index;
  n->factor = factor;
  return Split(n);
}

TVM_REGISTER_NODE_TYPE(DimSplitNode);

}  // namespace tvm
