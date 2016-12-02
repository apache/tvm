/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.cc
 */
#include <tvm/schedule.h>

namespace tvm {

Schedule::Schedule(Operation op, std::string scope) {
  auto n = std::make_shared<ScheduleNode>();
  n->op = op;
  n->scope = scope;
  node_ = n;
}

IterVarRelation SplitNode::make(
    IterVar parent, IterVar outer,
    IterVar inner, Expr factor) {
  auto n = std::make_shared<SplitNode>();
  n->parent = parent;
  n->outer = outer;
  n->inner = inner;
  n->factor = factor;
  return IterVarRelation(n);
}

IterVarRelation FuseNode::make(
    IterVar outer, IterVar inner, IterVar fused) {
  auto n = std::make_shared<FuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  return IterVarRelation(n);
}

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_NODE_TYPE(SplitNode);
TVM_REGISTER_NODE_TYPE(FuseNode);

}  // namespace tvm
