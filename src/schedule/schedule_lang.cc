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
 * \file schedule_lang.cc
 */
#include <tvm/schedule.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <unordered_set>
#include "graph.h"

namespace tvm {

namespace {

// find first occurance location in leaf
template<typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Node* n = v.get();
  for (size_t i = 0; i < array_node->data.size(); ++i) {
    if (array_node->data[i].get() == n) return i;
  }
  return array_node->data.size();
}

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* leaf_vars, const IterVar& v) {
  size_t pos = FindNodeRef(leaf_vars, v);
  if (pos < leaf_vars->data.size()) return pos;

  if (FindNodeRef(all_vars, v) < all_vars->data.size()) {
    LOG(FATAL) << "Operate on iter var " << v
               << "that has already been split";
  } else {
    LOG(FATAL) << "Operate on iter var " << v
               << "that is not part of the schedule";
  }
  return 0;
}

void Split(StageNode* self,
           IterVar parent,
           Expr factor,
           Expr nparts,
           IterVar* p_outer,
           IterVar* p_inner) {
  // Check if split is valid.
  CHECK(parent->iter_type == kDataPar ||
        parent->iter_type == kCommReduce ||
        parent->iter_type == kOrdered)
      << "Cannot split on " << IterVarType2String(parent->iter_type);
  IterVar outer = IterVarNode::make(
      Range(), parent->var.copy_with_suffix(".outer"), parent->iter_type);
  IterVar inner = IterVarNode::make(
      Range(), parent->var.copy_with_suffix(".inner"), parent->iter_type);
  *p_outer = outer;
  *p_inner = inner;
  // The splits
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  size_t pos = FindLeafVar(all_vars, leaf_vars, parent);
  self->relations.push_back(SplitNode::make(parent, outer, inner, factor, nparts));
  // add vars to all vars
  all_vars->data.push_back(outer.node_);
  all_vars->data.push_back(inner.node_);
  // replace the position.
  leaf_vars->data.erase(leaf_vars->data.begin() + pos);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, inner.node_);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, outer.node_);
}

}  // namespace

Stage::Stage(Operation op) {
  auto n = make_node<StageNode>();
  n->op = op;
  n->origin_op = op;
  n->all_iter_vars = op->root_iter_vars();
  // remove opaque var from leaf.
  Array<IterVar> clean;
  for (IterVar iv : n->all_iter_vars) {
    if (iv->iter_type != kOpaque) clean.push_back(iv);
  }
  if (clean.size() == n->all_iter_vars.size()) {
    n->leaf_iter_vars = n->all_iter_vars;
  } else {
    n->leaf_iter_vars = clean;
  }
  node_ = n;
}

bool Stage::is_scheduled() const {
  const StageNode* n = operator->();
  return !(n->relations.empty() && n->attach_type == kGroupRoot &&
           n->all_iter_vars.same_as(n->leaf_iter_vars));
}

Stage Stage::GetAttachSpec() const {
  Stage attach_spec = *this;
  while (attach_spec->attach_type == kGroupRoot &&
         attach_spec->group.defined()) {
    attach_spec = attach_spec->group;
  }
  return attach_spec;
}

Stage& Stage::set_scope(std::string scope) {  // NOLINT(*)
  (*this)->scope = scope;
  return *this;
}

Stage& Stage::compute_at(Stage parent, IterVar scope) {   // NOLINT(*)
  CHECK_NE((*this)->attach_type, kScanUpdate)
      << "Cannot specify compute_at for scan updates";
  // Group constraint checking.
  Stage group = (*this)->group;
  if (group.defined()) {
    Stage pg = parent->group;
    while (pg.defined() && !pg.same_as(group)) {
      pg = pg->group;
    }
    CHECK(pg.same_as(group))
        << "Can only assign compute_at to stages within the same group";
  }

  (*this)->attach_type = kScope;
  (*this)->attach_ivar = scope;
  (*this)->attach_stage = parent;
  bool found = false;
  for (size_t i = 0; i < parent->leaf_iter_vars.size(); ++i) {
    if (scope == parent->leaf_iter_vars[i]) {
      found = true; break;
    }
  }
  CHECK(found)
      << "Cannot find the axis " << scope
      << " in parent's leaf_iter_vars"
      << " parent=" << parent;
  return *this;
}

Stage& Stage::compute_inline() {   // NOLINT(*)
  CHECK_NE((*this)->attach_type, kScanUpdate)
      << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kInline;
  return *this;
}

Stage& Stage::compute_root() {   // NOLINT(*)
  CHECK_NE((*this)->attach_type, kScanUpdate)
      << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kGroupRoot;
  return *this;
}

Stage& Stage::bind(IterVar ivar, IterVar thread_ivar) {   // NOLINT(*)
  StageNode* self = operator->();
  CHECK(ivar->iter_type == kDataPar ||
        ivar->iter_type == kCommReduce)
      << "Cannot bind " << IterVarType2String(ivar->iter_type) << " to thread";
  CHECK(thread_ivar->iter_type == kThreadIndex)
      << "Cannot rebase by " << IterVarType2String(ivar->iter_type)
      << ", only thread axis is allowed so far";
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, ivar);

  auto it = self->iter_var_attrs.find(ivar);
  NodePtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_node<IterVarAttrNode>(*(*it).second.operator->());
    if (n->bind_thread.defined() &&
        !n->bind_thread.same_as(thread_ivar)) {
      LOG(WARNING) << "Axis " << ivar
                   << " is already bind to another thread " << n->bind_thread;
    }
  } else {
    n = make_node<IterVarAttrNode>();
  }
  n->bind_thread = thread_ivar;
  self->iter_var_attrs.Set(ivar, IterVarAttr(n));
  return *this;
}

Stage& Stage::env_threads(Array<IterVar> threads) {
  StageNode* self = operator->();
  CHECK(self->op.defined() && self->op.as<ScanOpNode>())
      << "env_threads is only valid for composite ops such as ScanOp";
  CHECK_EQ(self->env_threads.size(), 0U)
      << "Already set env_threads";
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  std::vector<NodePtr<Node> > temp;
  for (IterVar iv : threads) {
    temp.push_back(iv.node_);
  }
  leaf_vars->data.insert(
      leaf_vars->data.begin(), temp.begin(), temp.end());
  all_vars->data.insert(
      all_vars->data.end(), temp.begin(), temp.end());
  self->env_threads = threads;
  return *this;
}

Stage& Stage::set_store_predicate(Expr predicate) {
  StageNode* self = operator->();
  self->store_predicate = predicate;
  return *this;
}

Stage& Stage::split(
    IterVar parent, Expr factor, IterVar* p_outer, IterVar* p_inner) {  // NOLINT(*)
  Split(operator->(), parent, factor, Expr(), p_outer, p_inner);
  return *this;
}

Stage& Stage::split_by_nparts(
    IterVar parent, Expr nparts, IterVar* p_outer, IterVar* p_inner) { // NOLINT(*)
  Split(operator->(), parent, Expr(), nparts, p_outer, p_inner);
  return *this;
}

Stage& Stage::fuse(IterVar outer, IterVar inner, IterVar* p_target) {  // NOLINT(*)
  StageNode* self = operator->();
  CHECK(outer->iter_type == kDataPar ||
        outer->iter_type == kCommReduce ||
        outer->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(outer->iter_type);
  CHECK(inner->iter_type == kDataPar ||
        inner->iter_type == kCommReduce ||
        inner->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(inner->iter_type);

  IterVarType iter_type = outer->iter_type;
  if (inner->iter_type > iter_type) iter_type = inner->iter_type;
  std::string fused_name =
      outer->var->name_hint + "." + inner->var->name_hint + ".fused";

  IterVar fused = IterVarNode::make(
      Range(), Var(fused_name, outer->var.type()), iter_type);

  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();

  size_t pos_inner = FindLeafVar(all_vars, leaf_vars, inner);
  size_t pos_outer = FindLeafVar(all_vars, leaf_vars, outer);
  if (pos_inner + 1 == pos_outer) {
    std::swap(outer, inner);
    std::swap(pos_inner, pos_outer);
  }
  self->relations.push_back(FuseNode::make(outer, inner, fused));
  all_vars->data.push_back(fused.node_);
  CHECK_EQ(pos_inner, pos_outer + 1)
      << "Can only fuse iterations that are consecutive between each other";
  leaf_vars->data.erase(leaf_vars->data.begin() + pos_outer,
                        leaf_vars->data.begin() + pos_inner + 1);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos_outer,
                         fused.node_);
  *p_target = fused;
  return *this;
}

Stage& Stage::fuse(const Array<IterVar>& axes, IterVar* p_target) {  // NOLINT(*)
  if (axes.size() != 0) {
    IterVar fused = axes[0];
    for (size_t i = 1; i < axes.size(); ++i) {
      this->fuse(fused, axes[i], &fused);
    }
    *p_target = std::move(fused);
  } else {
    StageNode* self = operator->();
    // special handle fuse empty array.
    // insert at the outer most loop
    IterVar singleton = IterVarNode::make(
        Range::make_by_min_extent(0, 1),
        Var("singleton", Int(32)), kDataPar);
    self->relations.push_back(SingletonNode::make(singleton));
    ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
    ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
    all_vars->data.push_back(singleton.node_);
    leaf_vars->data.insert(leaf_vars->data.begin(), singleton.node_);
    *p_target = singleton;
  }
  return *this;
}

Stage& Stage::reorder(const Array<IterVar>& order) {  // NOLINT(*)
  std::unordered_set<IterVar> seen_var;
  StageNode* self = operator->();
  for (IterVar iv : order) {
    CHECK(iv->iter_type == kDataPar ||
          iv->iter_type == kCommReduce ||
          iv->iter_type == kThreadIndex)
        << "Cannot reorder IterVar("
        << IterVarType2String(iv->iter_type) << ")";

    CHECK_EQ(seen_var.count(iv), 0)
        << "Same axis can not appear more than once " << iv;
    seen_var.insert(iv);
  }
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  std::vector<size_t> pos;

  for (size_t i = 0; i < order.size(); ++i) {
    pos.push_back(FindLeafVar(all_vars, leaf_vars, order[i]));
  }
  std::vector<NodePtr<Node> > temp;
  for (size_t i = 0; i < pos.size(); ++i) {
    temp.emplace_back(leaf_vars->data[pos[i]]);
  }
  std::sort(pos.begin(), pos.end());
  for (size_t i = 0; i < pos.size(); ++i) {
    leaf_vars->data[pos[i]] = temp[i];
  }
  return *this;
}

Stage& Stage::tile(IterVar x_parent, IterVar y_parent,
                   Expr x_factor, Expr y_factor,
                   IterVar* p_x_outer, IterVar* p_y_outer,
                   IterVar* p_x_inner, IterVar* p_y_inner) {
  split(x_parent, x_factor, p_x_outer, p_x_inner);
  split(y_parent, y_factor, p_y_outer, p_y_inner);
  reorder(Array<IterVar>({*p_x_outer, *p_y_outer, *p_x_inner, *p_y_inner}));
  return *this;
}

template<typename FUpdate>
inline void UpdateIterVarAttr(StageNode* self,
                              IterVar var,
                              FUpdate fupdate,
                              bool need_leaf = true) {
  if (need_leaf) {
    ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
    ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
    FindLeafVar(all_vars, leaf_vars, var);
  }
  auto it = self->iter_var_attrs.find(var);
  NodePtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_node<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_node<IterVarAttrNode>();
  }
  fupdate(n.get());
  self->iter_var_attrs.Set(var, IterVarAttr(n));
}

inline void SetAttrIterType(StageNode* self, IterVar var, IterVarType iter_type) {
  UpdateIterVarAttr(self, var, [iter_type](IterVarAttrNode* n) {
      n->iter_type = iter_type;
    });
}

Stage& Stage::vectorize(IterVar var) {   // NOLINT(*)
  CHECK(var->iter_type == kDataPar ||
        var->iter_type == kOpaque ||
        var->iter_type == kUnrolled ||
        var->iter_type == kVectorized ||
        var->iter_type == kTensorized ||
        var->iter_type == kParallelized)
      << "Cannot vectorize on " << IterVarType2String(var->iter_type);
  SetAttrIterType(operator->(), var, kVectorized);
  return *this;
}

Stage& Stage::tensorize(IterVar var, TensorIntrin f) {   // NOLINT(*)
  UpdateIterVarAttr(operator->(), var, [f](IterVarAttrNode* n) {
      n->iter_type = kTensorized;
      n->tensor_intrin = f;
    });
  return *this;
}

Stage& Stage::unroll(IterVar var) {   // NOLINT(*)
  SetAttrIterType(operator->(), var, kUnrolled);
  return *this;
}

Stage& Stage::parallel(IterVar var) {   // NOLINT(*)
  SetAttrIterType(operator->(), var, kParallelized);
  return *this;
}

Stage& Stage::pragma(IterVar var,
                     const std::string& pragma_type,
                     const Expr& pragma_value) {   // NOLINT(*)
  if (pragma_type == "unroll") {
    this->unroll(var);
  } else if (pragma_type == "vectorize") {
    this->vectorize(var);
  } else {
    UpdateIterVarAttr(
        operator->(), var, [pragma_type, pragma_value](IterVarAttrNode* n) {
          n->pragma_keys.push_back(ir::StringImm::make(pragma_type));
          n->pragma_values.push_back(pragma_value);
        });
  }
  return *this;
}

Stage& Stage::prefetch(const Tensor &tensor, IterVar var, Expr offset) {
  StageNode *self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, var);
  auto it = self->iter_var_attrs.find(var);
  NodePtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_node<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_node<IterVarAttrNode>();
  }
  n->prefetch_data.push_back(tensor);
  n->prefetch_offset.push_back(offset);
  self->iter_var_attrs.Set(var, IterVarAttr(n));
  return *this;
}

Stage& Stage::storage_align(IterVar axis, int factor, int offset) {
  StageNode *self = operator->();
  UpdateIterVarAttr(self, axis, [factor, offset](IterVarAttrNode* n) {
      n->dim_align_factor = factor;
      n->dim_align_offset = offset;
    }, false);
  return *this;
}

Stage& Stage::double_buffer() {
  StageNode *self = operator->();
  CHECK(!self->is_output) << "Cannot apply double buffer on output";
  self->double_buffer = true;
  return *this;
}

Stage& Stage::opengl() {
  CHECK(!is_scheduled()) << "Must be a fresh schedule";
  StageNode *self = operator->();

  auto all_iter_vars = self->all_iter_vars;  // curr version of all_iter_vars
  CHECK(!all_iter_vars.empty()) << "At least one iter var";

  // Fuse all data parallel dimensions to 1.
  IterVar fused = all_iter_vars[0];
  for (size_t i = 1; i != all_iter_vars.size(); ++i) {
    auto iter_var = all_iter_vars[i];
    switch (iter_var->iter_type) {
      case IterVarType::kDataPar: {
        fuse(fused, all_iter_vars[i], &fused);
        break;
      }
      case IterVarType::kThreadIndex: {
        LOG(ERROR) << "A fresh schedule shouldn't have thread index iter var";
        break;
      }
      case IterVarType::kCommReduce:
      case IterVarType::kOrdered:
      case IterVarType::kOpaque: {
        break;
      }
      default: {
        LOG(ERROR) << "Invalid iter var type "
                   << IterVarType2String(iter_var->iter_type);
        break;
      }
    }
  }

  // Bind the only dimension to threadIdx.x.
  bind(fused, thread_axis(Range(nullptr), "threadIdx.x"));

  // Mark this stage as OpenGL.
  (*this)->is_opengl = true;

  return *this;
}

Stage CopyStage(const Stage& s) {
  NodePtr<StageNode> n =
      make_node<StageNode>(*s.operator->());
  return Stage(n);
}

Schedule Schedule::copy() const {
  // map of stages.
  const ScheduleNode* self = operator->();
  std::unordered_map<Stage, Stage, NodeHash, NodeEqual> smap;
  NodePtr<ScheduleNode> n = make_node<ScheduleNode>();
  n->outputs = self->outputs;
  // Copy the stages.
  for (Stage s : self->stages) {
    Stage scopy = CopyStage(s);
    smap[s] = scopy;
    n->stages.push_back(scopy);
  }
  for (Stage g : self->groups) {
    Stage gcopy = CopyStage(g);
    smap[g] = gcopy;
    n->groups.push_back(gcopy);
  }
  // Remaps the reference relations.
  for (auto kv : self->stage_map) {
    n->stage_map.Set(kv.first, smap.at(kv.second));
  }
  for (Stage s : n->stages) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
        << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end())
        << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  for (Stage s : n->groups) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
        << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end())
        << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  return Schedule(n);
}

Stage Schedule::operator[](const Operation& op) {
  auto it = (*this)->stage_map.find(op);
  CHECK(it != (*this)->stage_map.end())
      << "Cannot find Stage for operator " << op
      << " in the schedule";
  return (*it).second;
}

Stage LeastCommonAncestor(Stage g1, Stage g2) {
  if (!g1.defined()) return g1;
  if (!g2.defined()) return g2;
  if (g1.same_as(g2)) return g1;
  Stage g = g1;
  while (g.defined()) {
    if (g.same_as(g2)) return g2;
    g = g->group;
  }
  g = g2;
  while (g.defined()) {
    if (g.same_as(g1)) return g1;
    g = g->group;
  }
  return g;
}

Array<Tensor> RemapTensor(ScheduleNode* self,
                          const Array<Tensor>& arr) {
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      CHECK(self->stage_map.count(t->op))
          << "Given tensor is not in the schedule plan";
      t = self->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

// Group the schedule stages.
Stage Schedule::create_group(const Array<Tensor>& outputs,
                             const Array<Tensor>& inputs,
                             bool include_inputs) {
  ScheduleNode* self = operator->();
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  // Get the ops.
  Array<Operation> ops = schedule::GetSubGraph(
      RemapTensor(self, outputs),
      RemapTensor(self, inputs),
      include_inputs);
  // local counter entry
  // Automatically initialize to 0 during creation.
  struct Entry {
    int count{0};
  };
  // Map of group->touched counter
  std::unordered_map<Stage, Entry, NodeHash, NodeEqual> counter;
  // The parent group;
  Stage parent_group;
  // Detect common parent and child.
  for (size_t i = 0; i < ops.size(); ++i) {
    Operation op = ops[i];
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage op_group = it->second->group;
    if (i == 0) {
      parent_group = op_group;
    } else {
      parent_group = LeastCommonAncestor(parent_group, op_group);
    }
    if (op_group.defined()) {
      ++counter[op_group].count;
    }
  }
  // Create the new group stage.
  Stage gstage(make_node<StageNode>());
  gstage->group = parent_group;
  if (parent_group.defined()) {
    ++parent_group->num_child_stages;
  }
  // Propagate the counter statistics from by checking if subgroup
  // Is full and propagate.
  std::vector<Stage> stack;
  for (auto &kv : counter) {
    if (!kv.first.same_as(parent_group)) {
      if (kv.first->num_child_stages == kv.second.count) {
        stack.push_back(kv.first);
      }
    }
  }
  while (!stack.empty()) {
    Stage g = stack.back();
    stack.pop_back();
    if (g->group.defined() && !g->group.same_as(parent_group)) {
      Entry& e = counter[g->group];
      ++e.count;
      if (e.count == g->group->num_child_stages) {
        stack.push_back(g->group);
      }
    }
  }
  // Verification and remappig the subgroups.
  for (auto &kv : counter) {
    if (kv.first.same_as(parent_group)) continue;
    CHECK_EQ(kv.first->num_child_stages, kv.second.count)
        << "Trying to group region that intersect with an already existed group";
    if (kv.first->group.same_as(parent_group)) {
      Stage s = kv.first;
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Remap the group of op stages.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->group.same_as(parent_group)) {
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Correct the attach to keep everything in group.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->attach_type == kScope) {
      Stage cg = LeastCommonAncestor(s->attach_stage->group, gstage);
      if (!cg.same_as(gstage)) {
        LOG(WARNING) << "group invalidates some previous compute_at relation "
                     << " and keeps things to be computed inside the group";
        s.compute_root();
      }
    }
  }

  self->groups.push_back(gstage);
  return gstage;
}

void ScheduleNode::InvalidateCache() {
  op2stage_cache_.clear();
}

void ScheduleNode::InitCache() {
  if (op2stage_cache_.size() == stages.size()) return;
  InvalidateCache();
  for (Stage s : stages) {
    if (s->op.defined()) {
      op2stage_cache_[s->op.get()] = s;
    }
  }
  CHECK_EQ(op2stage_cache_.size(), stages.size());
}

bool ScheduleNode::Contain(const Operation& op) const {
  return stage_map.find(op) != stage_map.end();
}

Schedule ScheduleNode::make(Array<Operation> ops) {
  auto n = make_node<ScheduleNode>();
  Schedule sch(n);
  n->outputs = ops;
  auto g = schedule::CreateReadGraph(n->outputs);
  Array<Operation> post_order = schedule::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    Stage stage(op);
    stage->is_output = output_set.count(op) != 0;
    n->stages.push_back(stage);
    n->stage_map.Set(op, stage);
    // mark scan updates.
    if (const ScanOpNode* scan = op.as<ScanOpNode>()) {
      Array<Tensor> inputs;
      for (Tensor t : scan->state_placeholder) {
        inputs.push_back(t);
      }
      for (Tensor t : scan->inputs) {
        inputs.push_back(t);
      }
      // Create the scan group.
      Stage scan_group = sch.create_group(scan->update, inputs, false);
      scan_group->attach_type = kScanUpdate;
      scan_group->attach_stage = stage;

      for (size_t i = 0; i < scan->update.size(); ++i) {
        Stage s = n->stage_map[scan->update[i]->op];
        CHECK(scan_group.same_as(s->group));
      }
    }
  }
  return sch;
}

IterVarRelation SplitNode::make(IterVar parent,
                                IterVar outer,
                                IterVar inner,
                                Expr factor,
                                Expr nparts) {
  auto n = make_node<SplitNode>();
  n->parent = parent;
  n->outer = outer;
  n->inner = inner;
  n->factor = factor;
  n->nparts = nparts;
  return IterVarRelation(n);
}

IterVarRelation FuseNode::make(
    IterVar outer, IterVar inner, IterVar fused) {
  auto n = make_node<FuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  return IterVarRelation(n);
}

IterVarRelation RebaseNode::make(IterVar parent, IterVar rebased) {
  auto n = make_node<RebaseNode>();
  n->parent = parent;
  n->rebased = rebased;
  return IterVarRelation(n);
}

IterVarRelation SingletonNode::make(IterVar iter) {
  auto n = make_node<SingletonNode>();
  n->iter = iter;
  return IterVarRelation(n);
}

TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(IterVarAttrNode);
TVM_REGISTER_NODE_TYPE(SplitNode);
TVM_REGISTER_NODE_TYPE(FuseNode);
TVM_REGISTER_NODE_TYPE(RebaseNode);
TVM_REGISTER_NODE_TYPE(SingletonNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);

// Printer
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<StageNode>([](const StageNode *op, IRPrinter *p) {
    if (op->op.defined()) {
      p->stream << "stage(" << op->origin_op->name << ", " << op << ")";
    } else {
      p->stream << "group-stage(" << op << ")";
    }
})
.set_dispatch<IterVarAttrNode>([](const IterVarAttrNode *op, IRPrinter *p) {
    p->stream << IterVarType2String(op->iter_type);
})
.set_dispatch<SplitNode>([](const SplitNode *op, IRPrinter *p) {
    p->stream << "split(parent=";
    p->Print(op->parent);
    p->stream << ", outer=";
    p->Print(op->outer);
    p->stream << ", inner=";
    p->Print(op->inner);
    p->stream << ')';
})
.set_dispatch<FuseNode>([](const FuseNode *op, IRPrinter *p) {
    p->stream << "split(";
    p->stream << "outer=";
    p->Print(op->outer);
    p->stream << ", inner=";
    p->Print(op->inner);
    p->stream << ", fused=";
    p->Print(op->fused);
    p->stream << ')';
})
.set_dispatch<RebaseNode>([](const RebaseNode *op, IRPrinter *p) {
    p->stream << "rebase(";
    p->stream << "parent=";
    p->Print(op->parent);
    p->stream << ", rebased=";
    p->Print(op->rebased);
    p->stream << ')';
})
.set_dispatch<SingletonNode>([](const SingletonNode *op, IRPrinter *p) {
    p->stream << "singleton(";
    p->Print(op->iter);
    p->stream << ')';
})
.set_dispatch<ScheduleNode>([](const ScheduleNode *op, IRPrinter *p) {
    p->stream << "schedule(" << op << ")";
  });
}  // namespace tvm
