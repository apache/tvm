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
#include <dmlc/thread_local.h>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>

#include <algorithm>
#include <stack>
#include <unordered_set>
#include <vector>

#include "graph.h"

namespace tvm {
namespace te {

// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->size(); ++i) {
    if (array_node->at(i).get() == n) return i;
  }
  return array_node->size();
}

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* leaf_vars, const IterVar& v) {
  size_t pos = FindNodeRef(leaf_vars, v);
  if (pos < leaf_vars->size()) return pos;

  if (FindNodeRef(all_vars, v) < all_vars->size()) {
    LOG(FATAL) << "Operate on iter var " << v << "that has already been split";
  } else {
    LOG(FATAL) << "Operate on iter var " << v << "that is not part of the schedule";
  }
  return 0;
}

DataType MatchDataType(std::vector<DataType> dtypes) {
  int max_bits = -1;
  for (const auto& dtype : dtypes) {
    ICHECK(dtype.is_int());
    ICHECK(dtype.is_scalar());
    max_bits = std::max(max_bits, dtype.bits());
  }
  return DataType::Int(max_bits);
}

void SplitHelper(StageNode* self, IterVar parent, PrimExpr factor, PrimExpr nparts,
                 IterVar* p_outer, IterVar* p_inner, bool disable_predication) {
  // Check if split is valid.
  ICHECK(parent->iter_type == kDataPar || parent->iter_type == kCommReduce ||
         parent->iter_type == kOrdered)
      << "Cannot split on " << IterVarType2String(parent->iter_type);
  IterVar outer = IterVar(Range(), parent->var.copy_with_suffix(".outer"), parent->iter_type);
  IterVar inner = IterVar(Range(), parent->var.copy_with_suffix(".inner"), parent->iter_type);
  *p_outer = outer;
  *p_inner = inner;
  // The splits
  Array<IterVar>& all_vars = self->all_iter_vars;
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  size_t pos = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), parent);
  self->relations.push_back(Split(parent, outer, inner, factor, nparts, disable_predication));
  // add vars to all vars
  all_vars.push_back(outer);
  all_vars.push_back(inner);
  // replace the position.
  leaf_vars.erase(leaf_vars.begin() + pos);
  leaf_vars.insert(leaf_vars.begin() + pos, inner);
  leaf_vars.insert(leaf_vars.begin() + pos, outer);
}

Stage::Stage(Operation op, const ScheduleNode* sch) {
  auto n = make_object<StageNode>();
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
  n->attach_sch = sch;
  data_ = std::move(n);
}

bool Stage::is_scheduled() const {
  const StageNode* n = operator->();
  return !(n->relations.empty() && n->attach_type == kGroupRoot &&
           n->all_iter_vars.same_as(n->leaf_iter_vars));
}

Stage Stage::GetAttachSpec() const {
  Stage attach_spec = *this;
  while (attach_spec->attach_type == kGroupRoot && attach_spec->group.defined()) {
    attach_spec = attach_spec->group;
  }
  return attach_spec;
}

Stage& Stage::set_scope(std::string scope) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  (*this)->scope = scope;
  return *this;
}

Stage& Stage::compute_at(Stage parent, IterVar scope) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  ICHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  // Group constraint checking.
  Stage group = (*this)->group;
  if (group.defined()) {
    Stage pg = parent->group;
    while (pg.defined() && !pg.same_as(group)) {
      pg = pg->group;
    }
    ICHECK(pg.same_as(group)) << "Can only assign compute_at to stages within the same group";
  }

  (*this)->attach_type = kScope;
  (*this)->attach_ivar = scope;
  (*this)->attach_stage = parent;
  bool found = false;
  for (size_t i = 0; i < parent->leaf_iter_vars.size(); ++i) {
    if (scope == parent->leaf_iter_vars[i]) {
      found = true;
      break;
    }
  }
  ICHECK(found) << "Cannot find the axis " << scope << " in parent's leaf_iter_vars"
                << " parent=" << parent;
  return *this;
}

Stage& Stage::compute_inline() {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  ICHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kInline;
  return *this;
}

Stage& Stage::compute_root() {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  ICHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kGroupRoot;
  return *this;
}

Stage& Stage::bind(IterVar ivar, IterVar thread_ivar) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  ICHECK(ivar->iter_type == kDataPar || ivar->iter_type == kCommReduce)
      << "Cannot bind " << IterVarType2String(ivar->iter_type) << " to thread";
  ICHECK(thread_ivar->iter_type == kThreadIndex)
      << "Cannot rebase by " << IterVarType2String(ivar->iter_type)
      << ", only thread axis is allowed so far";
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, ivar);

  auto it = self->iter_var_attrs.find(ivar);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
    if (n->bind_thread.defined() && !n->bind_thread.same_as(thread_ivar)) {
      LOG(WARNING) << "Axis " << ivar << " is already bind to another thread " << n->bind_thread;
    }
  } else {
    n = make_object<IterVarAttrNode>();
  }
  n->bind_thread = thread_ivar;
  self->iter_var_attrs.Set(ivar, IterVarAttr(n));
  return *this;
}

Stage& Stage::env_threads(Array<IterVar> threads) {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  ICHECK(self->op.defined() && self->op.as<ScanOpNode>())
      << "env_threads is only valid for composite ops such as ScanOp";
  ICHECK_EQ(self->env_threads.size(), 0U) << "Already set env_threads";
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  Array<IterVar>& all_vars = self->all_iter_vars;
  std::vector<IterVar> temp;
  for (IterVar iv : threads) {
    temp.push_back(iv);
  }
  leaf_vars.insert(leaf_vars.begin(), temp.begin(), temp.end());
  all_vars.insert(all_vars.end(), temp.begin(), temp.end());
  self->env_threads = threads;
  return *this;
}

Stage& Stage::set_store_predicate(PrimExpr predicate) {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  self->store_predicate = predicate;
  return *this;
}

Stage& Stage::split(IterVar parent, PrimExpr factor, IterVar* p_outer, IterVar* p_inner,
                    bool disable_predication) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  SplitHelper(operator->(), parent, factor, PrimExpr(), p_outer, p_inner, disable_predication);
  return *this;
}

Stage& Stage::split_by_nparts(IterVar parent, PrimExpr nparts, IterVar* p_outer, IterVar* p_inner,
                              bool disable_predication) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  SplitHelper(operator->(), parent, PrimExpr(), nparts, p_outer, p_inner, disable_predication);
  return *this;
}

Stage& Stage::fuse(IterVar outer, IterVar inner, IterVar* p_target) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  ICHECK(outer->iter_type == kDataPar || outer->iter_type == kCommReduce ||
         outer->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(outer->iter_type);
  ICHECK(inner->iter_type == kDataPar || inner->iter_type == kCommReduce ||
         inner->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(inner->iter_type);

  IterVarType iter_type = outer->iter_type;
  if (inner->iter_type > iter_type) iter_type = inner->iter_type;
  std::string fused_name = outer->var->name_hint + "." + inner->var->name_hint + ".fused";
  DataType iter_dtype = MatchDataType({inner->var.dtype(), outer->var.dtype()});

  IterVar fused = IterVar(Range(), Var(fused_name, iter_dtype), iter_type);

  Array<IterVar>& all_vars = self->all_iter_vars;
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;

  size_t pos_inner = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), inner);
  size_t pos_outer = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), outer);
  if (pos_inner + 1 == pos_outer) {
    std::swap(outer, inner);
    std::swap(pos_inner, pos_outer);
  }
  ICHECK_EQ(pos_inner, pos_outer + 1)
      << "Can only fuse iterations that are consecutive between each other";
  self->relations.push_back(Fuse(outer, inner, fused));
  all_vars.push_back(fused);
  leaf_vars.erase(leaf_vars.begin() + pos_outer, leaf_vars.begin() + pos_inner + 1);
  leaf_vars.insert(leaf_vars.begin() + pos_outer, fused);
  *p_target = fused;
  return *this;
}

Stage& Stage::fuse(const Array<IterVar>& axes, IterVar* p_target) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
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
    IterVar singleton =
        IterVar(Range::FromMinExtent(0, 1), Var("singleton", DataType::Int(32)), kDataPar);
    self->relations.push_back(Singleton(singleton));
    Array<IterVar>& all_vars = self->all_iter_vars;
    Array<IterVar>& leaf_vars = self->leaf_iter_vars;
    all_vars.push_back(singleton);
    leaf_vars.insert(leaf_vars.begin(), singleton);
    *p_target = singleton;
  }
  return *this;
}

Stage& Stage::reorder(const Array<IterVar>& order) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  std::unordered_set<IterVar> seen_var;
  StageNode* self = operator->();
  for (IterVar iv : order) {
    ICHECK(iv->iter_type == kDataPar || iv->iter_type == kCommReduce ||
           iv->iter_type == kThreadIndex)
        << "Cannot reorder IterVar(" << IterVarType2String(iv->iter_type) << ")";

    ICHECK_EQ(seen_var.count(iv), 0) << "Same axis can not appear more than once " << iv;
    seen_var.insert(iv);
  }
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  std::vector<size_t> pos;

  for (size_t i = 0; i < order.size(); ++i) {
    pos.push_back(FindLeafVar(all_vars, leaf_vars, order[i]));
  }
  std::vector<ObjectRef> temp;
  for (size_t i = 0; i < pos.size(); ++i) {
    temp.emplace_back(leaf_vars->at(pos[i]));
  }
  std::sort(pos.begin(), pos.end());
  for (size_t i = 0; i < pos.size(); ++i) {
    leaf_vars->SetItem(pos[i], temp[i]);
  }
  return *this;
}

Stage& Stage::tile(IterVar x_parent, IterVar y_parent, PrimExpr x_factor, PrimExpr y_factor,
                   IterVar* p_x_outer, IterVar* p_y_outer, IterVar* p_x_inner, IterVar* p_y_inner) {
  split(x_parent, x_factor, p_x_outer, p_x_inner);
  split(y_parent, y_factor, p_y_outer, p_y_inner);
  reorder(Array<IterVar>({*p_x_outer, *p_y_outer, *p_x_inner, *p_y_inner}));
  return *this;
}

template <typename FUpdate>
inline void UpdateIterVarAttr(StageNode* self, IterVar var, FUpdate fupdate,
                              bool need_leaf = true) {
  if (need_leaf) {
    ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
    ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
    FindLeafVar(all_vars, leaf_vars, var);
  }
  auto it = self->iter_var_attrs.find(var);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_object<IterVarAttrNode>();
  }
  fupdate(n.get());
  self->iter_var_attrs.Set(var, IterVarAttr(n));
}

inline void SetAttrIterType(StageNode* self, IterVar var, IterVarType iter_type) {
  UpdateIterVarAttr(self, var, [iter_type](IterVarAttrNode* n) { n->iter_type = iter_type; });
}

Stage& Stage::vectorize(IterVar var) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  ICHECK(var->iter_type == kDataPar || var->iter_type == kOpaque || var->iter_type == kUnrolled ||
         var->iter_type == kVectorized || var->iter_type == kTensorized ||
         var->iter_type == kParallelized)
      << "Cannot vectorize on " << IterVarType2String(var->iter_type);
  SetAttrIterType(operator->(), var, kVectorized);
  return *this;
}

Stage& Stage::tensorize(IterVar var, TensorIntrin f) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  UpdateIterVarAttr(operator->(), var, [f](IterVarAttrNode* n) {
    n->iter_type = kTensorized;
    n->tensor_intrin = f;
  });
  return *this;
}

Stage& Stage::unroll(IterVar var) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  SetAttrIterType(operator->(), var, kUnrolled);
  return *this;
}

Stage& Stage::parallel(IterVar var) {  // NOLINT(*)
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  SetAttrIterType(operator->(), var, kParallelized);
  return *this;
}

Stage& Stage::pragma(IterVar var, const std::string& pragma_type,
                     const PrimExpr& pragma_value) {  // NOLINT(*)
  if (pragma_type == "unroll") {
    this->unroll(var);
  } else if (pragma_type == "vectorize") {
    this->vectorize(var);
  } else {
    With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
    UpdateIterVarAttr(operator->(), var, [pragma_type, pragma_value](IterVarAttrNode* n) {
      n->pragma_keys.push_back(tir::StringImm(pragma_type));
      n->pragma_values.push_back(pragma_value);
    });
  }
  return *this;
}

Stage& Stage::prefetch(const Tensor& tensor, IterVar var, PrimExpr offset) {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, var);
  auto it = self->iter_var_attrs.find(var);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_object<IterVarAttrNode>();
  }
  n->prefetch_data.push_back(tensor);
  n->prefetch_offset.push_back(offset);
  self->iter_var_attrs.Set(var, IterVarAttr(n));
  return *this;
}

Stage& Stage::storage_align(IterVar axis, int factor, int offset) {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  UpdateIterVarAttr(
      self, axis,
      [factor, offset](IterVarAttrNode* n) {
        n->dim_align_factor = factor;
        n->dim_align_offset = offset;
      },
      false);
  return *this;
}

Stage& Stage::double_buffer() {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  ICHECK(!self->is_output) << "Cannot apply double buffer on output";
  self->double_buffer = true;
  return *this;
}

Stage& Stage::rolling_buffer() {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  ICHECK(!self->is_output) << "Cannot apply rolling buffer on output";
  self->rolling_buffer = true;
  return *this;
}
Stage& Stage::transform_layout(const Array<Var>& initial_indices,
                               const Array<PrimExpr>& final_indices,
                               Array<IterVar>* out_iter_vars) {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  IndexMap map(initial_indices, final_indices);
  self->layout_transforms.push_back(map);

  auto* compute = self->op.as<ComputeOpNode>();

  // Can only rewrite the indices of compute op nodes.
  if (!compute) {
    return *this;
  }

  CHECK_EQ(initial_indices.size(), compute->axis.size())
      << "Expected number of initial indices in transformation to match the dimension of "
      << self->op->name;

  // Locate the IterVar objects for the data axes.
  auto leaf_iter_range = [&]() -> std::pair<size_t, size_t> {
    std::vector<size_t> leaf_var_indices;
    for (const auto& axis : compute->axis) {
      leaf_var_indices.push_back(
          FindLeafVar(self->all_iter_vars.CopyOnWrite(), self->leaf_iter_vars.CopyOnWrite(), axis));
    }
    auto minmax_element = std::minmax_element(leaf_var_indices.begin(), leaf_var_indices.end());
    return {*minmax_element.first, *minmax_element.second + 1};
  }();
  CHECK_EQ(leaf_iter_range.first + compute->axis.size(), leaf_iter_range.second)
      << "Cannot transform indices if they have already been reordered";

  // Determine the updated ranges of iteration.
  Array<Range> initial_ranges;
  for (const auto& iter_var : compute->axis) {
    initial_ranges.push_back(iter_var->dom);
  }
  arith::Analyzer analyzer;
  Array<Range> final_ranges = map->MapRanges(initial_ranges, &analyzer);

  // Make IterVar objects to represent the new iterations.
  auto inverse = map.Inverse(initial_ranges, &analyzer);
  Array<IterVar> final_indices_iter;
  ICHECK_EQ(inverse->initial_indices.size(), final_ranges.size());
  for (size_t i = 0; i < inverse->initial_indices.size(); i++) {
    final_indices_iter.push_back(IterVar(final_ranges[i], inverse->initial_indices[i], kDataPar));
  }

  // Append the new IterVar objects to all_iter_vars
  for (const auto& iter_var : final_indices_iter) {
    self->all_iter_vars.push_back(iter_var);
  }

  // Replace the existing IterVar objects in leaf_iter_vars with the
  // new IterVar objects.
  self->leaf_iter_vars.erase(self->leaf_iter_vars.begin() + leaf_iter_range.first,
                             self->leaf_iter_vars.begin() + leaf_iter_range.second);
  self->leaf_iter_vars.insert(self->leaf_iter_vars.begin() + leaf_iter_range.first,
                              final_indices_iter.begin(), final_indices_iter.end());

  // Define a relationship for each new axis
  self->relations.push_back(Transform(compute->axis, final_indices_iter, map, inverse));

  // Return the iteration variables as an output.
  if (out_iter_vars) {
    *out_iter_vars = final_indices_iter;
  }

  return *this;
}

Stage& Stage::set_axis_separators(const Array<IntImm>& axis_separators) {
  With<ScheduleContext> ctx(operator->()->attach_sch, __func__);
  StageNode* self = operator->();
  self->axis_separators = axis_separators;
  return *this;
}

Stage CopyStage(const Stage& s) {
  ObjectPtr<StageNode> n = make_object<StageNode>(*s.operator->());
  return Stage(n);
}

Schedule Schedule::copy() const {
  // map of stages.
  const ScheduleNode* self = operator->();
  std::unordered_map<Stage, Stage, ObjectPtrHash, ObjectPtrEqual> smap;
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
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
      ICHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      ICHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  for (Stage s : n->groups) {
    if (s->attach_stage.defined()) {
      ICHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      ICHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  return Schedule(n);
}

Stage Schedule::operator[](const Operation& op) {
  auto it = (*this)->stage_map.find(op);
  ICHECK(it != (*this)->stage_map.end())
      << "Cannot find Stage for operator " << op << " in the schedule";
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

Array<Tensor> RemapTensor(ScheduleNode* self, const Array<Tensor>& arr) {
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      ICHECK(self->stage_map.count(t->op)) << "Given tensor is not in the schedule plan";
      t = self->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

// Group the schedule stages.
Stage Schedule::create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs) {
  ScheduleNode* self = operator->();
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  // Get the ops.
  Array<Operation> ops =
      te::GetSubGraph(RemapTensor(self, outputs), RemapTensor(self, inputs), include_inputs);
  // local counter entry
  // Automatically initialize to 0 during creation.
  struct Entry {
    int count{0};
  };
  // Map of group->touched counter
  std::unordered_map<Stage, Entry, ObjectPtrHash, ObjectPtrEqual> counter;
  // The parent group;
  Stage parent_group;
  // Detect common parent and child.
  for (size_t i = 0; i < ops.size(); ++i) {
    Operation op = ops[i];
    auto it = op2stage_cache.find(op.get());
    ICHECK(it != op2stage_cache.end());
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
  Stage gstage(make_object<StageNode>());
  gstage->attach_sch = this->operator->();
  gstage->group = parent_group;
  if (parent_group.defined()) {
    ++parent_group->num_child_stages;
  }
  // Propagate the counter statistics from by checking if subgroup
  // Is full and propagate.
  std::vector<Stage> stack;
  for (auto& kv : counter) {
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
  for (auto& kv : counter) {
    if (kv.first.same_as(parent_group)) continue;
    ICHECK_EQ(kv.first->num_child_stages, kv.second.count)
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
    ICHECK(it != op2stage_cache.end());
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
    ICHECK(it != op2stage_cache.end());
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

void ScheduleNode::InvalidateCache() { op2stage_cache_.clear(); }

void ScheduleNode::InitCache() {
  if (op2stage_cache_.size() == stages.size()) return;
  InvalidateCache();
  for (Stage s : stages) {
    if (s->op.defined()) {
      op2stage_cache_[s->op.get()] = s;
    }
  }
  ICHECK_EQ(op2stage_cache_.size(), stages.size());
}

bool ScheduleNode::Contain(const Operation& op) const {
  return stage_map.find(op) != stage_map.end();
}

TVM_REGISTER_PASS_CONFIG_OPTION("te.keep_schedule_record", Bool);

Schedule::Schedule(Array<Operation> ops) {
  auto n = make_object<ScheduleNode>();
  data_ = n;
  n->outputs = ops;
  auto g = te::CreateReadGraph(n->outputs);
  Array<Operation> post_order = te::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    Stage stage(op, this->operator->());
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
      Stage scan_group = this->create_group(scan->update, inputs, false);
      scan_group->attach_type = kScanUpdate;
      scan_group->attach_stage = stage;

      for (size_t i = 0; i < scan->update.size(); ++i) {
        Stage s = n->stage_map[scan->update[i]->op];
        ICHECK(scan_group.same_as(s->group));
      }
    }
  }
  transform::PassContext pass_ctx = transform::PassContext::Current();
  n->keep_schedule_record = pass_ctx->GetConfig<Bool>("te.keep_schedule_record", Bool(false));
  if (n->keep_schedule_record.value()) {
    // push plain schedule as the very first one
    n->schedule_record.push_back(copy());
    n->primitive_record.push_back("vanilla");
  }
}

ScheduleContext::ScheduleContext(const ScheduleNode* sch_node, String current_primitive_name)
    : sch_(GetRef<Schedule>(sch_node)), current_primitive_name_(current_primitive_name) {}

void ScheduleContext::EnterWithScope() {}

void ScheduleContext::ExitWithScope() {
  if (sch_.defined() && sch_->keep_schedule_record.value()) {
    sch_->schedule_record.push_back(sch_.copy());
    sch_->primitive_record.push_back(current_primitive_name_);
  }
}

Split::Split(IterVar parent, IterVar outer, IterVar inner, PrimExpr factor, PrimExpr nparts,
             bool disable_predication) {
  auto n = make_object<SplitNode>();
  n->parent = parent;
  n->outer = outer;
  n->inner = inner;
  n->factor = factor;
  n->nparts = nparts;
  n->disable_predication = disable_predication;
  data_ = std::move(n);
}

Fuse::Fuse(IterVar outer, IterVar inner, IterVar fused) {
  auto n = make_object<FuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  data_ = std::move(n);
}

Rebase::Rebase(IterVar parent, IterVar rebased) {
  auto n = make_object<RebaseNode>();
  n->parent = parent;
  n->rebased = rebased;
  data_ = std::move(n);
}

Singleton::Singleton(IterVar iter) {
  auto n = make_object<SingletonNode>();
  n->iter = iter;
  data_ = std::move(n);
}

Transform::Transform(Array<IterVar> original_variables, Array<IterVar> transformed_variables,
                     IndexMap forward_transformation, IndexMap inverse_transformation) {
  auto n = make_object<TransformNode>();
  n->original_variables = original_variables;
  n->transformed_variables = transformed_variables;
  n->forward_transformation = forward_transformation;
  n->inverse_transformation = inverse_transformation;
  data_ = std::move(n);
}

SpecializedCondition::SpecializedCondition(Array<PrimExpr> conditions) {
  ObjectPtr<SpecializedConditionNode> n = make_object<SpecializedConditionNode>();
  n->clauses = std::move(conditions);
  data_ = std::move(n);
}

/*! \brief Entry to hold the SpecializedCondition context stack. */
struct TVMSpecializationThreadLocalEntry {
  /*! \brief The current specialized condition */
  std::stack<SpecializedCondition> condition_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
typedef dmlc::ThreadLocalStore<TVMSpecializationThreadLocalEntry> TVMSpecializationThreadLocalStore;

void SpecializedCondition::EnterWithScope() {
  TVMSpecializationThreadLocalEntry* entry = TVMSpecializationThreadLocalStore::Get();
  entry->condition_stack.push(*this);
}

void SpecializedCondition::ExitWithScope() {
  TVMSpecializationThreadLocalEntry* entry = TVMSpecializationThreadLocalStore::Get();
  ICHECK(!entry->condition_stack.empty());
  ICHECK(entry->condition_stack.top().same_as(*this));
  entry->condition_stack.pop();
}

SpecializedCondition SpecializedCondition::Current() {
  TVMSpecializationThreadLocalEntry* entry = TVMSpecializationThreadLocalStore::Get();
  SpecializedCondition cond;
  if (entry->condition_stack.size() > 0) {
    cond = entry->condition_stack.top();
  }
  return cond;
}

class SpecializedCondition::Internal {
 public:
  static void EnterScope(SpecializedCondition cond) { cond.EnterWithScope(); }

  static void ExitScope(SpecializedCondition cond) { cond.ExitWithScope(); }
};

TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(IterVarAttrNode);
TVM_REGISTER_NODE_TYPE(SplitNode);
TVM_REGISTER_NODE_TYPE(FuseNode);
TVM_REGISTER_NODE_TYPE(RebaseNode);
TVM_REGISTER_NODE_TYPE(SingletonNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_NODE_TYPE(SpecializedConditionNode);

// Printer
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StageNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StageNode*>(node.get());
      if (op->op.defined()) {
        p->stream << "stage(" << op->origin_op->name << ", " << op->op << ")";
      } else {
        p->stream << "group-stage(" << op << ")";
      }
    })
    .set_dispatch<IterVarAttrNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterVarAttrNode*>(node.get());
      p->stream << IterVarType2String(op->iter_type);
    })
    .set_dispatch<SplitNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SplitNode*>(node.get());
      p->stream << "split(parent=";
      p->Print(op->parent);
      p->stream << ", outer=";
      p->Print(op->outer);
      p->stream << ", inner=";
      p->Print(op->inner);
      if (op->factor.defined()) {
        p->stream << ", factor=";
        p->Print(op->factor);
      } else {
        p->stream << ", nparts=";
        p->Print(op->nparts);
      }
      p->stream << ", disable_predication=";
      p->stream << op->disable_predication;
      p->stream << ')';
    })
    .set_dispatch<FuseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FuseNode*>(node.get());
      p->stream << "fuse(";
      p->stream << "outer=";
      p->Print(op->outer);
      p->stream << ", inner=";
      p->Print(op->inner);
      p->stream << ", fused=";
      p->Print(op->fused);
      p->stream << ')';
    })
    .set_dispatch<RebaseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RebaseNode*>(node.get());
      p->stream << "rebase(";
      p->stream << "parent=";
      p->Print(op->parent);
      p->stream << ", rebased=";
      p->Print(op->rebased);
      p->stream << ')';
    })
    .set_dispatch<SingletonNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SingletonNode*>(node.get());
      p->stream << "singleton(";
      p->Print(op->iter);
      p->stream << ')';
    })
    .set_dispatch<ScheduleNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ScheduleNode*>(node.get());
      p->stream << "schedule(" << op << ")";
    })
    .set_dispatch<SpecializedConditionNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SpecializedConditionNode*>(node.get());
      p->stream << "specialized_condition(";
      p->Print(op->clauses);
      p->stream << ')';
    });

TVM_REGISTER_GLOBAL("te.CreateSchedule").set_body_typed(create_schedule);

TVM_REGISTER_GLOBAL("te.StageSetScope").set_body_method(&Stage::set_scope);

TVM_REGISTER_GLOBAL("te.StageBind").set_body_method(&Stage::bind);

TVM_REGISTER_GLOBAL("te.StageSplitByFactor")
    .set_body_typed([](Stage stage, IterVar parent, PrimExpr factor, bool disable_predication) {
      IterVar outer, inner;
      stage.split(parent, factor, &outer, &inner, disable_predication);
      return Array<IterVar>({outer, inner});
    });

TVM_REGISTER_GLOBAL("te.StageSplitByNParts")
    .set_body_typed([](Stage stage, IterVar parent, PrimExpr nparts, bool disable_predication) {
      IterVar outer, inner;
      stage.split_by_nparts(parent, nparts, &outer, &inner, disable_predication);
      return Array<IterVar>({outer, inner});
    });

TVM_REGISTER_GLOBAL("te.StageFuse").set_body_typed([](Stage stage, Array<IterVar> axes) {
  IterVar fused;
  stage.fuse(axes, &fused);
  return fused;
});

TVM_REGISTER_GLOBAL("te.StageComputeAt").set_body_method(&Stage::compute_at);

TVM_REGISTER_GLOBAL("te.StageComputeInline").set_body_method(&Stage::compute_inline);

TVM_REGISTER_GLOBAL("te.StageComputeRoot").set_body_method(&Stage::compute_root);

TVM_REGISTER_GLOBAL("te.StageReorder").set_body_method(&Stage::reorder);

TVM_REGISTER_GLOBAL("te.StageTile")
    .set_body_typed([](Stage stage, IterVar x_parent, IterVar y_parent, PrimExpr x_factor,
                       PrimExpr y_factor) {
      IterVar x_outer, y_outer, x_inner, y_inner;
      stage.tile(x_parent, y_parent, x_factor, y_factor, &x_outer, &y_outer, &x_inner, &y_inner);
      return Array<IterVar>({x_outer, y_outer, x_inner, y_inner});
    });

TVM_REGISTER_GLOBAL("te.StageEnvThreads").set_body_method(&Stage::env_threads);

TVM_REGISTER_GLOBAL("te.StageSetStorePredicate").set_body_method(&Stage::set_store_predicate);

TVM_REGISTER_GLOBAL("te.StageUnroll").set_body_method(&Stage::unroll);

TVM_REGISTER_GLOBAL("te.StageVectorize").set_body_method(&Stage::vectorize);

TVM_REGISTER_GLOBAL("te.StageTensorize").set_body_method(&Stage::tensorize);

TVM_REGISTER_GLOBAL("te.StageParallel").set_body_method(&Stage::parallel);

TVM_REGISTER_GLOBAL("te.StagePragma").set_body_method(&Stage::pragma);

TVM_REGISTER_GLOBAL("te.StagePrefetch").set_body_method(&Stage::prefetch);

TVM_REGISTER_GLOBAL("te.StageStorageAlign").set_body_method(&Stage::storage_align);

TVM_REGISTER_GLOBAL("te.StageDoubleBuffer").set_body_method(&Stage::double_buffer);

TVM_REGISTER_GLOBAL("te.StageRollingBuffer").set_body_method(&Stage::rolling_buffer);

TVM_REGISTER_GLOBAL("te.StageTransformLayout")
    .set_body_typed([](Stage stage, const Array<Var>& initial_indices,
                       const Array<PrimExpr>& final_indices) {
      Array<IterVar> new_iter_vars;
      stage.transform_layout(initial_indices, final_indices, &new_iter_vars);
      return new_iter_vars;
    });

TVM_REGISTER_GLOBAL("te.StageSetAxisSeparators").set_body_method(&Stage::set_axis_separators);

TVM_REGISTER_GLOBAL("te.ScheduleNormalize").set_body_method(&Schedule::normalize);

TVM_REGISTER_GLOBAL("te.ScheduleCreateGroup").set_body_method(&Schedule::create_group);

TVM_REGISTER_GLOBAL("te.ScheduleCacheRead").set_body_method(&Schedule::cache_read);

TVM_REGISTER_GLOBAL("te.ScheduleCacheWrite").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args[1].IsObjectRef<Tensor>()) {
    *ret = args[0].operator Schedule().cache_write(args[1].operator Tensor(), args[2]);
  } else {
    *ret = args[0].operator Schedule().cache_write(args[1].operator Array<Tensor>(), args[2]);
  }
});

TVM_REGISTER_GLOBAL("te.ScheduleRFactor").set_body_method(&Schedule::rfactor);

TVM_REGISTER_GLOBAL("te.CreateSpecializedCondition").set_body_typed([](Array<PrimExpr> condition) {
  return SpecializedCondition(condition);
});

TVM_REGISTER_GLOBAL("te.GetCurrentSpecialization").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = SpecializedCondition::Current();
});

TVM_REGISTER_GLOBAL("te.EnterSpecializationScope")
    .set_body_typed(SpecializedCondition::Internal::EnterScope);

TVM_REGISTER_GLOBAL("te.ExitSpecializationScope")
    .set_body_typed(SpecializedCondition::Internal::ExitScope);

}  // namespace te
}  // namespace tvm
