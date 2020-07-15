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
 * \file auto_scheduler/loop_state.cc
 * \brief An lightweight IR (intermediate representation) for loop structures.
 * see auto_scheduler/loop_state.h for more explanation.
 */

#include "loop_state.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

#include <utility>

#include "transform_step.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(StepNode);
TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(StateNode);
TVM_REGISTER_NODE_TYPE(IteratorNode);

/********** Iterator **********/
Iterator::Iterator(String name, Range range, IteratorKind iter_kind,
                   IteratorAnnotation annotation) {
  auto node = make_object<IteratorNode>();
  node->name = std::move(name);
  node->range = std::move(range);
  node->iter_kind = iter_kind;
  node->annotation = annotation;
  data_ = std::move(node);
}

/********** Stage **********/
Stage::Stage(te::Operation op) {
  auto node = make_object<StageNode>();
  if (op->IsInstance<te::ComputeOpNode>()) {
    node->op_type = StageKind::kCompute;
    auto* pop = op.as<te::ComputeOpNode>();
    for (const auto& axis : pop->axis) {
      node->iters.push_back(Iterator(CleanName(axis->var->name_hint), axis->dom,
                                     IteratorKind::kSpatial, IteratorAnnotation::kNone));
    }
    for (const auto& axis : pop->reduce_axis) {
      node->iters.push_back(Iterator(CleanName(axis->var->name_hint), axis->dom,
                                     IteratorKind::kReduction, IteratorAnnotation::kNone));
    }
  } else if (op->IsInstance<te::PlaceholderOpNode>()) {
    node->op_type = StageKind::kPlaceholder;
  } else {
    LOG(FATAL) << "Unsupported operator type" << op->_type_key;
  }

  node->compute_at = ComputeAtKind::kRoot;
  node->op = std::move(op);
  node->attrs.auto_unroll_max_step = 0;
  node->attrs.storage_offset = 0;
  data_ = std::move(node);
}

Stage::Stage(te::Operation op, StageKind op_type, const Array<Iterator>& iters,
             ComputeAtKind compute_at, StageAttributes attrs) {
  auto node = make_object<StageNode>();
  node->op = std::move(op);
  node->op_type = op_type;
  node->iters = iters;
  node->compute_at = compute_at;
  node->attrs = attrs;
  data_ = std::move(node);
}

/********** AttachMap **********/
void AttachMap::SetComputeAtIter(int stage_id, int target_stage_id,
                                 int target_iter_id) {
  AttachMapNode* pnode = CopyOnWrite();

  // delete the current entry of stage
  DeleteStageEntry(pnode, stage_id);

  // store the new relation
  IterKey iter_key(target_stage_id, target_iter_id);
  pnode->stage_to_attach_iter[stage_id] =
      std::make_pair(target_stage_id, target_iter_id);
  pnode->iter_to_attached_stages[iter_key].push_back(stage_id);
}

void AttachMap::DeleteStage(int stage_id) {
  AttachMapNode* pnode = CopyOnWrite();

  // delete the entry of old stage
  DeleteStageEntry(pnode, stage_id);
}

void AttachMap::ReplaceIters(const std::vector<IterKey>& old_iters,
                             const std::vector<IterKey>& new_iters) {
  AttachMapNode* pnode = CopyOnWrite();

  CHECK_EQ(old_iters.size(), new_iters.size());
  for (size_t i = 0; i < old_iters.size(); ++i) {
    auto entry = pnode->iter_to_attached_stages.find(old_iters[i]);
    if (entry == pnode->iter_to_attached_stages.end()) {
      continue;
    }

    // replace iter in the value of `stage_to_attach_iter`
    for (const auto& s : entry->second) {
      pnode->stage_to_attach_iter[s] = new_iters[i];
    }

    // replace iter in the key of `iter_to_attached_stages`
    std::vector<int> attached_stages = std::move(entry->second);
    pnode->iter_to_attached_stages.erase(entry);
    pnode->iter_to_attached_stages[new_iters[i]] = std::move(attached_stages);
  }
}

void AttachMap::DeleteStageEntry(AttachMapNode* pnode, int stage_id) {
  auto old_entry = pnode->stage_to_attach_iter.find(stage_id);
  if (old_entry != pnode->stage_to_attach_iter.end()) {
    // delete value in `iter_to_attached_stages`
    auto entry2 = pnode->iter_to_attached_stages.find(old_entry->second);
    DeleteItem(&entry2->second, stage_id);
    if (entry2->second.size() == 0) {
      pnode->iter_to_attached_stages.erase(entry2);
    }
    // delete key in `stage_to_attach_iter`
    pnode->stage_to_attach_iter.erase(old_entry);
  }
}

AttachMap AttachMap::ApplyStageIdOfffset(int start_id, int offset) const {
  AttachMap map = AttachMap(make_object<AttachMapNode>());
  auto pmap = map.CopyOnWrite();
  for (const auto& x : operator->()->stage_to_attach_iter) {
    auto key = x.first;
    if (key >= start_id) {
      key += offset;
    }
    auto value = x.second;
    if (value.first >= start_id) {
      value.first += offset;
    }
    pmap->stage_to_attach_iter.insert(std::make_pair(key, value));
  }
  for (const auto& x : operator->()->iter_to_attached_stages) {
    auto key = x.first;
    if (key.first >= start_id) {
      key.first += offset;
    }
    auto value = x.second;
    for (auto& i : value) {
      if (i >= start_id) {
        i += offset;
      }
    }
    pmap->iter_to_attached_stages.insert(std::make_pair(key, value));
  }
  return map;
}

/********** State **********/
State::State(const Array<te::Operation>& ops) {
  auto node = make_object<StateNode>();
  for (const auto& op : ops) {
    node->stages.push_back(Stage(op));
  }
  node->attach_map = AttachMap(make_object<AttachMapNode>());
  node->concrete = true;
  data_ = std::move(node);
}

/********** Schedule primitives apis for state **********/
void State::reorder(int stage_id, const Array<Iterator>& order) {
  const Stage& stage = operator->()->stages[stage_id];
  CHECK_EQ(order.size(), stage->iters.size()) << "The order of all iterators "
                                              << "should be specified";
  Array<Integer> after_ids;
  GetIndices(stage->iters, order, &after_ids);
  ReorderStep step = ReorderStep(stage_id, after_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  DoReorderStep(step);
}

void State::compute_at(int stage_id, int target_stage_id,
                       const Iterator& target_iter) {
  const Stage& target_stage = operator->()->stages[target_stage_id];
  ComputeAtStep step = ComputeAtStep(
      stage_id, target_stage_id, GetIndex(target_stage->iters, target_iter));
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeAtStep(step);
}

void State::compute_root(int stage_id) {
  ComputeRootStep step = ComputeRootStep(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeRootStep(step);
}

void State::compute_inline(int stage_id) {
  ComputeInlineStep step = ComputeInlineStep(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeInlineStep(step);
}

Array<Iterator> State::split(int stage_id, const Iterator& it,
                             const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];
  SplitStep step =
      SplitStep(stage_id, GetIndex(stage->iters, it),
                it->range.defined() ? it->range->extent : PrimExpr(), lengths, inner_to_outer);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoSplitStep(step);
}

Iterator State::fuse(int stage_id, const Array<Iterator>& iters) {
  const Stage& stage = operator->()->stages[stage_id];
  Array<Integer> indices;
  GetIndices(stage->iters, iters, &indices);
  FuseStep step = FuseStep(stage_id, indices);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFuseStep(step);
}

Iterator State::vectorize(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step = AnnotationStep(
      stage_id, GetIndex(stage->iters, it), IteratorAnnotation::kVectorize);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

Iterator State::parallel(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step =
      AnnotationStep(stage_id, GetIndex(stage->iters, it), IteratorAnnotation::kParallel);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

Iterator State::unroll(int stage_id, const Iterator& it, int max_unroll) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step =
      AnnotationStep(stage_id, GetIndex(stage->iters, it), IteratorAnnotation::kUnroll);

  // don't unroll if the extent is larger than max_unroll
  if (max_unroll != -1 && it->range.defined()) {
    if (auto imm = it->range->extent.as<IntImmNode>()) {
      if (imm->value > max_unroll) {
        return it;
      }
    }
  }

  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

Iterator State::bind_thread(int stage_id, const Iterator& it,
                            IteratorAnnotation thread_type) {
  const Stage& stage = operator->()->stages[stage_id];
  if (thread_type < IteratorAnnotation::kVThread || thread_type > IteratorAnnotation::kThreadY) {
    LOG(FATAL) << "thread_type error, valide: kVThread, kBlockX, kBlockY, "
               << "kThreadX, kThreadY";
  }
  AnnotationStep step = AnnotationStep(
      stage_id, GetIndex(stage->iters, it), thread_type);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}


/********** Step implementations for state **********/
void State::DoReorderStep(const ReorderStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];
  Array<Iterator> iters;
  for (auto x : step->after_ids) {
    iters.push_back(stage->iters[x]);
  }
  StateNode* pstate = CopyOnWrite();
  pstate->stages.Set(step->stage_id,
                     Stage(stage->op, stage->op_type, iters, stage->compute_at, stage->attrs));
}

void State::DoComputeAtStep(const ComputeAtStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  // after compute_at, we don't know the accurate length information any more
  // If we do want to know the accurate lengths, we can call
  // ComputeDAG::ReplayAndInferBound
  std::vector<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    new_iters.push_back(Iterator(it->name, Range(), it->iter_kind,
                        it->annotation));
  }

  StateNode* pstate = CopyOnWrite();
  pstate->stages.Set(step->stage_id,
      Stage(stage->op, stage->op_type, std::move(new_iters), ComputeAtKind::kIter,
            stage->attrs));
  pstate->attach_map.SetComputeAtIter(step->stage_id, step->target_stage_id,
                                      step->target_iter_id);
}

void State::DoComputeRootStep(const ComputeRootStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  // after compute_root, we don't know the accurate length information any more
  // If we do want to know the accurate lengths, we can call
  // ComputeDAG::ReplayAndInferBound
  std::vector<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    new_iters.push_back(Iterator(it->name, Range(), it->iter_kind,
                                 it->annotation));
  }

  // update attach map
  StateNode* pstate = CopyOnWrite();
  pstate->stages.Set(step->stage_id, Stage(stage->op, stage->op_type,
                                         std::move(new_iters), ComputeAtKind::kRoot,
                                         stage->attrs));
  pstate->attach_map.DeleteStage(step->stage_id);
}

void State::DoComputeInlineStep(const ComputeInlineStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  StateNode* pstate = CopyOnWrite();

  // CHECK the validity of compute_inline
  const auto& iter_to_attached_stages =
      pstate->attach_map->iter_to_attached_stages;
  for (size_t i = 0; i < stage->iters.size(); ++i) {
    CHECK_EQ(iter_to_attached_stages.count(std::make_pair(step->stage_id, i)),
             0)
        << "Invalid compute_inline: Because there are some other stages "
           "that are attached to the target stage";
  }

  auto new_stage = pstate->stages[step->stage_id];
  new_stage.CopyOnWrite()->compute_at = ComputeAtKind::kInlined;
  pstate->stages.Set(step->stage_id, std::move(new_stage));
  pstate->attach_map.DeleteStage(step->stage_id);
}

// common part for DoSplitStep, DoFollowSplitStep, and DoFollowFusedSplitStep
Array<Iterator> State::DoSplitStepCommon(int stage_id, int iter_id,
                                         const Array<Optional<Integer>>& lengths,
                                         bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];
  const Iterator& it = stage->iters[iter_id];
  size_t old_iter_size = stage->iters.size();
  bool concrete = true;

  Optional<PrimExpr> tosplit_min, tosplit_extent;
  if (it->range.defined()) {
    tosplit_min = it->range->min;
    tosplit_extent = it->range->extent;
  } else {
    tosplit_min = NullOpt;
    tosplit_extent = NullOpt;
  }

  Array<Iterator> outs;
  for (size_t i = 0; i < lengths.size(); ++i) {
    Optional<Integer> l;
    String name;
    if (inner_to_outer) {
      l = lengths[lengths.size() - i - 1];
      name = it->name + "." + std::to_string(lengths.size() - i);
    } else {
      l = lengths[i];
      name = it->name + "." + std::to_string(i);
    }
    Iterator res;
    if (l && tosplit_min && tosplit_extent) {
      res = Iterator(name, Range::FromMinExtent(tosplit_min.value(), l.value()), it->iter_kind,
                     IteratorAnnotation::kNone);
      tosplit_min = Integer(0);
      tosplit_extent = indexdiv(tosplit_extent.value() + l.value() - 1, l.value());
    } else {
      res = Iterator(name, Range(), it->iter_kind, IteratorAnnotation::kNone);
      tosplit_min = NullOpt;
      tosplit_extent = NullOpt;
      concrete = false;
    }
    outs.push_back(std::move(res));
  }

  Range range;
  if (tosplit_min && tosplit_extent) {
    range = Range::FromMinExtent(tosplit_min.value(), tosplit_extent.value());
  }
  if (inner_to_outer) {
    outs.push_back(Iterator(it->name + ".0", range, it->iter_kind, IteratorAnnotation::kNone));
    // Reverse the Iterator array
    Array<Iterator> temp(outs.rbegin(), outs.rend());
    outs = std::move(temp);
  } else {
    outs.push_back(Iterator(it->name + "." + std::to_string(lengths.size()), range, it->iter_kind,
                            IteratorAnnotation::kNone));
  }

  Array<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(), stage->iters.begin() + iter_id);
  new_iters.insert(new_iters.end(), outs.begin(), outs.end());
  new_iters.insert(new_iters.end(), stage->iters.begin() + iter_id + 1, stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages.Set(stage_id,
                     Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));
  pstate->concrete &= concrete;

  // we have to replace the iterators in attach map,
  // these two vectors keep the replacement mapping
  std::vector<AttachMap::IterKey> from_iters;
  std::vector<AttachMap::IterKey> to_iters;
  for (size_t i = iter_id; i < old_iter_size; ++i) {
    from_iters.emplace_back(stage_id, i);
    to_iters.emplace_back(stage_id, i + lengths.size());
  }
  pstate->attach_map.ReplaceIters(from_iters, to_iters);

  return outs;
}

Array<Iterator> State::DoSplitStep(const SplitStep& step) {
  return DoSplitStepCommon(step->stage_id, step->iter_id, step->lengths, step->inner_to_outer);
}

Iterator State::DoFuseStep(const FuseStep& step) {
  int stage_id = step->stage_id;
  const Stage& stage = operator->()->stages[stage_id];
  size_t old_iter_size = static_cast<int>(stage->iters.size());

  String new_name;
  PrimExpr new_extent = 1;
  IteratorKind new_iter_kind = IteratorKind::kSpecial;

  for (size_t i = 0; i < step->fused_ids.size(); ++i) {
    if (i > 0) {
      CHECK_EQ(step->fused_ids[i]->value, step->fused_ids[i - 1]->value + 1);
    }

    if (i != step->fused_ids.size() - 1) {
      const auto& iter_to_attached_stage =
      operator->()->attach_map->iter_to_attached_stages;
      if (iter_to_attached_stage.find(std::make_pair(
              stage_id, step->fused_ids[i])) != iter_to_attached_stage.end()) {
        LOG(FATAL) << "Invalid Fuse. Trying to fuse iterators that have been attached by some "
                   << "stages. State before fusion:\n"
                   << *this;
      }
    }

    const Iterator& it = stage->iters[step->fused_ids[i]];
    new_name = new_name + it->name + "@";

    if (it->range.defined() && new_extent.defined()) {
      new_extent = new_extent * it->range->extent;
    } else {
      new_extent = PrimExpr();
    }

    if (i == 0) {
      new_iter_kind = it->iter_kind;
    } else {
      if (new_iter_kind != it->iter_kind) {
        new_iter_kind = IteratorKind::kMixed;
      }
    }
  }

  Range range;
  if (new_extent.defined()) {
    range = Range::FromMinExtent(0, new_extent);
  }
  Iterator new_it = Iterator(new_name, range, new_iter_kind, IteratorAnnotation::kNone);
  Array<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(),
                   stage->iters.begin() + step->fused_ids.front());
  new_iters.push_back(new_it);
  new_iters.insert(new_iters.end(), stage->iters.begin() + step->fused_ids.back() + 1,
                   stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages.Set(stage_id,
                     Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));

  // we have to replace the iterators in attach map,
  // these two vectors keep the replacement mapping
  std::vector<AttachMap::IterKey> from_iters;
  std::vector<AttachMap::IterKey> to_iters;
  const size_t begin_id = step->fused_ids.front(), end_id = step->fused_ids.back();
  for (size_t i = 0; i < old_iter_size; ++i) {
    if (i <= begin_id) {
      continue;
    } else if (i > end_id) {  // move forward
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, i - end_id + begin_id);
    } else {  // move to the fused id
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, begin_id);
    }
  }
  pstate->attach_map.ReplaceIters(from_iters, to_iters);

  return new_it;
}

Iterator State::DoAnnotationStep(const AnnotationStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];
  Iterator it = stage->iters[step->iter_id];

  CHECK(it->annotation == IteratorAnnotation::kNone);
  Iterator new_it = Iterator(it->name, it->range, it->iter_kind,
                             step->annotation);
  Stage new_stage = stage;
  new_stage.CopyOnWrite()->iters.Set(step->iter_id, std::move(new_it));
  StateNode* pstate = CopyOnWrite();
  pstate->stages.Set(step->stage_id, std::move(new_stage));
  return new_it;
}

void State::DoSteps(const ComputeDAG& dag) {
  CHECK(operator->()->stages.size()) << "Invalid State with empty operation stages.";

  for (const auto& step : operator->()->transform_steps) {
    if (auto ps = step.as<ReorderStepNode>()) {
      DoReorderStep(GetRef<ReorderStep>(ps));
    } else if (auto ps = step.as<ComputeAtStepNode>()) {
      DoComputeAtStep(GetRef<ComputeAtStep>(ps));
    } else if (auto ps = step.as<ComputeRootStepNode>()) {
      DoComputeRootStep(GetRef<ComputeRootStep>(ps));
    } else if (auto ps = step.as<ComputeInlineStepNode>()) {
      DoComputeInlineStep(GetRef<ComputeInlineStep>(ps));
    } else if (auto ps = step.as<SplitStepNode>()) {
      DoSplitStep(GetRef<SplitStep>(ps));
    } else if (auto ps = step.as<FuseStepNode>()) {
      DoFuseStep(GetRef<FuseStep>(ps));
    } else if (auto ps = step.as<AnnotationStepNode>()) {
      DoAnnotationStep(GetRef<AnnotationStep>(ps));
    } else {
      LOG(FATAL) << "Invalid step: " << step;
    }
  }
}

static const char* IteratorAnnotationString[] = {
    "for",              // kNone = 0
    "unroll",           // kUnroll = 1
    "vectorize",        // kVectorize = 2
    "parallel",         // kParallel = 3
    "vthread",          // kVThread = 4
    "gpu.blockIdx.x",   // kBlockX = 5
    "gpu.threadIdx.x",  // kThreadX = 6
    "gpu.blockIdx.y",   // kBlockY = 7
    "gpu.threadIdx.y",  // kThreadY = 8
    "tensorize"         // kTensorized = 9
};

// Print stage to ostream
void PrintStage(std::ostream* os, int stage_id, const State& state, size_t base_indent,
                bool delete_trivial_loop) {
  const Stage& stage = state->stages[stage_id];

  if (stage->attrs.auto_unroll_max_step != 0) {
    for (size_t j = 0; j < base_indent; ++j) {
      *os << " ";
    }
    *os << stage->op->name << " auto_unroll: " << stage->attrs.auto_unroll_max_step << "\n";
  }
  if (stage->attrs.storage_offset != 0) {
    for (size_t j = 0; j < base_indent; ++j) {
      *os << " ";
    }
    *os << stage->op->name << " storage_offset: " << stage->attrs.storage_offset << "\n";
  }

  size_t indent = 0;
  for (size_t i = 0; i < stage->iters.size(); ++i) {
    const Iterator& iter = stage->iters[i];

    if (!(delete_trivial_loop && iter->range.defined() && is_one(iter->range->extent))) {
      for (size_t j = 0; j < base_indent + indent; ++j) {
        *os << " ";
      }
      *os << IteratorAnnotationString[static_cast<int>(iter->annotation)] << " ";
      if (iter->range.defined()) {
        *os << iter->name << " (" << iter->range->min << "," << iter->range->extent << ")";
      } else {
        *os << iter->name << " (None)";
      }
      *os << "\n";

      indent += 2;
    }

    if (state.defined()) {
      AttachMap::IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        for (const auto& attach_stage_id : pair->second) {
          PrintStage(os, attach_stage_id, state, base_indent + indent,
                     delete_trivial_loop);
        }
      }
    }
  }

  for (size_t j = 0; j < base_indent + indent; ++j) {
    *os << " ";
  }
  *os << stage->op->name << " = ...\n";
}

// Print state to ostream
void PrintState(std::ostream* os, const State& state, bool delete_trivial_loop) {
  // Gather placeholders
  Array<String> placeholders;
  for (const auto& stage : state->stages) {
    if (stage->op_type == StageKind::kPlaceholder) {
      placeholders.push_back(stage->op->name);
    }
  }

  *os << "Placeholder: ";
  for (size_t i = 0; i < placeholders.size(); ++i) {
    *os << placeholders[i];
    if (i != placeholders.size() - 1) {
      *os << ", ";
    }
  }
  *os << "\n";

  // Print all stages
  for (size_t i = 0; i < state->stages.size(); ++i) {
    const Stage& stage = state->stages[i];
    if (stage->op_type == StageKind::kPlaceholder) {
      continue;
    } else if (stage->op_type == StageKind::kCompute) {
      if (stage->compute_at == ComputeAtKind::kRoot) {
        PrintStage(os, i, state, 0, delete_trivial_loop);
      }
    } else {
      LOG(FATAL) << "Invalid op type";
    }
  }
}

String State::ToStr(bool delete_trivial_loop) const {
  std::ostringstream os;
  PrintState(&os, (*this), delete_trivial_loop);
  return os.str();
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StateNode>([](const ObjectRef& ref, ReprPrinter* p) {
      PrintState(&p->stream, tvm::Downcast<State>(ref), true);
    });

/********** State interface API for ffi **********/
TVM_REGISTER_GLOBAL("auto_scheduler.StateReorder")
    .set_body_typed([](State state, int stage_id, const Array<Iterator>& order) {
      state.reorder(stage_id, order);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateComputeAt")
    .set_body_typed([](State state, int stage_id, int target_stage_id,
                      const Iterator& target_iter) {
      state.compute_at(stage_id, target_stage_id, target_iter);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateComputeRoot")
    .set_body_typed([](State state, int stage_id) {
      state.compute_root(stage_id);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateComputeInline")
    .set_body_typed([](State state, int stage_id) {
      state.compute_inline(stage_id);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateSplit")
    .set_body_typed([](State state, int stage_id, const Iterator& it,
                       const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
      const auto& res = state.split(stage_id, it, lengths, inner_to_outer);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateFuse")
    .set_body_typed([](State state, int stage_id, const Array<Iterator>& iters) {
      const auto& res = state.fuse(stage_id, iters);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateVectorize")
    .set_body_typed([](State state, int stage_id, const Iterator& it) {
      const auto& res = state.vectorize(stage_id, it);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateParallel")
    .set_body_typed([](State state, int stage_id, const Iterator& it) {
      const auto& res = state.parallel(stage_id, it);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateUnroll")
    .set_body_typed([](State state, int stage_id, const Iterator& it,
                      int max_unroll) {
      const auto& res = state.unroll(stage_id, it, max_unroll);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateBindThread")
    .set_body_typed([](State state, int stage_id, const Iterator& it,
                      int thread_type) {
      const auto& res =
          state.bind_thread(stage_id, it, IteratorAnnotation(thread_type));
      return Array<ObjectRef>{state, res};
    });


TVM_REGISTER_GLOBAL("auto_scheduler.StateEqual").set_body_typed([](State state1, State state2) {
  return std::equal_to<State>()(state1, state2);
});

}  // namespace auto_scheduler
}  // namespace tvm
