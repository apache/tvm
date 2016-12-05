/*!
 *  Copyright (c) 2016 by Contributors
 * \file schedule.cc
 */
#include <tvm/schedule.h>

namespace tvm {

namespace {

// find first occurance location in leaf
size_t FindIterVar(ArrayNode* array_node, const IterVar& v) {
  const Node* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* const IterVar& v) {
  size_t pos = Find(leaf_iter_vars, parent);
}

}

Schedule::Schedule(Operation op, std::string scope) {
  auto n = std::make_shared<ScheduleNode>();
  n->op = op;
  n->scope = scope;
  n->all_iter_vars = op->root_iter_vars();
  n->leaf_iter_vars = op->root_iter_vars();
  node_ = n;
}

Schedule& Schedule::compute_at(Schedule parent, IterVar scope) {   // NOLINT(*)
  CHECK_EQ((*this)->attach_type, kNone);
  (*this)->attach_type = kScope;
  (*this)->attach_parent = scope;
  parent->children.push_back(*this);
  return *this;
}

Schedule& Schedule::compute_inline(Schedule parent) {   // NOLINT(*)
  CHECK_EQ((*this)->attach_type, kNone);
  (*this)->attach_type = kInline;
  parent->children.push_back(*this);
  return *this;
}

Schedule& Schedule::compute_root(Schedule parent) {   // NOLINT(*)
  CHECK_EQ((*this)->attach_type, kNone);
  (*this)->attach_type = kRoot;
  parent->children.push_back(*this);
  return *this;
}

Schedule& Schedule::split(
    IterVar parent, IterVar* p_outer, IterVar* p_inner, Expr factor) {  // NOLINT(*)
  ScheduleNode* self = operator->();
  ArrayNode* leaf_iter_vars = self->leaf_iter_vars.CopyOnWrite();

  CHECK(pos != leaf_iter_vars->data.size())
      << "Cannot find IterVar " << parent << " in the active leaf vars"
      << " this means "

  return *this;
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
