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

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* leaf_vars, const IterVar& v) {
  size_t pos = FindIterVar(leaf_vars, v);
  if (pos < leaf_vars->data.size()) return pos;

  if (FindIterVar(all_vars, v) < all_vars->data.size()) {
    LOG(FATAL) << "Operate on iter var " << v
               << "that has already been splitted";
  } else {
    LOG(FATAL) << "Operate on iter var " << v
               << "that is not part of the schedule";
  }
  return 0;
}

void Split(ScheduleNode* self, IterVar parent,
           IterVar outer, IterVar inner, Expr factor) {
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  size_t pos = FindLeafVar(all_vars, leaf_vars, parent);

  self->relations.push_back(SplitNode::make(parent, outer, inner, factor));
  // add vars to all vars
  all_vars->data.push_back(outer.node_);
  all_vars->data.push_back(inner.node_);
  // replace the position.
  leaf_vars->data.erase(leaf_vars->data.begin() + pos);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, inner.node_);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, outer.node_);
}

}  // namespace

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
  bool found = false;
  for (size_t i = 0; i < parent->leaf_iter_vars.size(); ++i) {
    if (scope == parent->leaf_iter_vars[i]) {
      found = true; break;
    }
  }
  CHECK(found)
      << "Cannot compute at a iteration variable that is not part of parent leaf vars";
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
  // place holder for the splitted results.
  IterVar outer(Range(), parent->var->name_hint + ".outer");
  IterVar inner(Range(), parent->var->name_hint + ".inner");
  *p_outer = outer; *p_inner = inner;

  Split(operator->(), parent, outer, inner, factor);
  return *this;
}

Schedule& Schedule::split(IterVar parent, IterVar outer, IterVar* p_inner, Expr factor) { // NOLINT(*)
  // place holder for the splitted results.
  IterVar inner(Range(), parent->var->name_hint + ".inner");
  *p_inner = inner;
  Split(operator->(), parent, outer, inner, factor);

  return *this;
}

Schedule& Schedule::fuse(IterVar inner, IterVar outer, IterVar* p_target) {  // NOLINT(*)
  IterVar fused(Range(), outer->var->name_hint + "." + inner->var->name_hint + ".fused");
  ScheduleNode* self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();

  self->relations.push_back(FuseNode::make(inner, outer, fused));
  all_vars->data.push_back(fused.node_);

  size_t pos_inner = FindLeafVar(all_vars, leaf_vars, inner);
  size_t pos_outer = FindLeafVar(all_vars, leaf_vars, outer);
  CHECK_EQ(pos_inner, pos_outer + 1)
      << "Can only fuse iterations that are consecutive between each other";
  leaf_vars->data.erase(leaf_vars->data.begin() + pos_outer,
                        leaf_vars->data.begin() + pos_inner);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos_outer,
                         fused.node_);
  return *this;
}

Schedule& Schedule::reorder(const Array<IterVar>& order) {  // NOLINT(*)
  ScheduleNode* self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  std::vector<size_t> pos;

  for (size_t i = 0; i < order.size(); ++i) {
    pos.push_back(FindLeafVar(all_vars, leaf_vars, order[i]));
  }
  std::vector<std::shared_ptr<Node> > temp;
  for (size_t i = 0; i < pos.size(); ++i) {
    temp.emplace_back(leaf_vars->data[pos[i]]);
  }
  std::sort(pos.begin(), pos.end());
  for (size_t i = 0; i < pos.size(); ++i) {
    leaf_vars->data[pos[i]] = temp[i];
  }
  return *this;
}

Schedule& Schedule::tile(IterVar x_parent, IterVar y_parent, IterVar* p_x_outer,
                         IterVar* p_y_outer, IterVar* p_x_inner, IterVar* p_y_inner,
                         Expr x_factor, Expr y_factor) { // NOLINT(*)

  split(x_parent, p_x_outer, p_x_inner, x_factor);
  split(y_parent, p_y_outer, p_y_inner, y_factor);
  reorder(Array<IterVar>({*p_x_inner, *p_y_inner, *p_x_outer, *p_y_outer}));
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
