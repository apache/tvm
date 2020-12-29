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

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

#include <utility>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(StepNode);
TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(StateNode);
TVM_REGISTER_NODE_TYPE(IteratorNode);

/********** Iterator **********/
Iterator::Iterator(String name, Range range, IteratorKind iter_kind, IteratorAnnotation annotation,
                   const std::vector<Iterator>* orig_iters) {
  auto node = make_object<IteratorNode>();
  node->name = std::move(name);
  node->range = std::move(range);
  node->iter_kind = iter_kind;
  node->annotation = annotation;
  if (orig_iters != nullptr) {
    node->orig_iters = *orig_iters;
  }
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
void AttachMap::SetComputeAtIter(int stage_id, int target_stage_id, int target_iter_id) {
  AttachMapNode* pnode = CopyOnWrite();

  // Delete the current entry of this stage
  DeleteStageEntry(pnode, stage_id);

  // Store the new stage/iterator relations to map
  IterKey iter_key(target_stage_id, target_iter_id);
  pnode->stage_to_attach_iter[stage_id] = iter_key;
  pnode->iter_to_attached_stages[iter_key].push_back(stage_id);
}

void AttachMap::DeleteStage(int stage_id) {
  AttachMapNode* pnode = CopyOnWrite();
  // Delete the original stage entry
  DeleteStageEntry(pnode, stage_id);
}

void AttachMap::UpdateIters(const std::vector<IterKey>& original_iters,
                            const std::vector<IterKey>& new_iters) {
  ICHECK_EQ(original_iters.size(), new_iters.size());
  AttachMapNode* pnode = CopyOnWrite();
  std::unordered_map<IterKey, std::vector<StageKey>> new_iter_to_attached_stages;
  for (size_t i = 0; i < original_iters.size(); ++i) {
    auto entry = pnode->iter_to_attached_stages.find(original_iters[i]);
    // We get <IterKey, std::vector<StageKey>> from this map
    if (entry == pnode->iter_to_attached_stages.end()) {
      // Skip if this iterator does not have any attach relations
      continue;
    }

    // Update the attaching target of an stage to the new iter in `stage_to_attach_iter`
    for (const auto& s : entry->second) {
      pnode->stage_to_attach_iter[s] = new_iters[i];
    }

    // Remove the original iterator relation from `iter_to_attached_stages` and add the new
    // iterator to it
    std::vector<int> attached_stages = std::move(entry->second);
    pnode->iter_to_attached_stages.erase(entry);
    new_iter_to_attached_stages[new_iters[i]] = std::move(attached_stages);
  }

  // Update new entries
  for (auto& it : new_iter_to_attached_stages) {
    pnode->iter_to_attached_stages[it.first] = std::move(it.second);
  }
}

void AttachMap::DeleteStageEntry(AttachMapNode* pnode, int stage_id) {
  auto old_entry = pnode->stage_to_attach_iter.find(stage_id);
  // We get <StageKey, IterKey> from this map
  if (old_entry != pnode->stage_to_attach_iter.end()) {
    // Delete the stage in `iter_to_attached_stages`, if the corresponding iterator does not have
    // any attached stage, delete this iterm too
    auto entry2 = pnode->iter_to_attached_stages.find(old_entry->second);
    // We get <IterKey, std::vector<StageKey>> from this map
    FindAndDeleteItem(&entry2->second, stage_id);
    if (entry2->second.size() == 0) {
      pnode->iter_to_attached_stages.erase(entry2);
    }
    // Delete the stage in `stage_to_attach_iter`
    pnode->stage_to_attach_iter.erase(old_entry);
  }
}

AttachMap AttachMap::ApplyStageIdOffset(int start_id, int offset) const {
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
Iterator State::bind(int stage_id, const Iterator& it, IteratorAnnotation thread_type) {
  const Stage& stage = operator->()->stages[stage_id];
  if (thread_type < IteratorAnnotation::kVThread || thread_type > IteratorAnnotation::kThreadZ) {
    LOG(FATAL) << "thread_type error, valid: kVThread, kBlockX, kBlockY, "
               << "kThreadX, kThreadY, kBlockZ, kThreadZ";
  }
  AnnotationStep step = AnnotationStep(stage_id, GetIndex(stage->iters, it), thread_type);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

Iterator State::parallel(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step =
      AnnotationStep(stage_id, GetIndex(stage->iters, it), IteratorAnnotation::kParallel);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

Iterator State::unroll(int stage_id, const Iterator& it, int max_unroll) {
  const Stage& stage = operator->()->stages[stage_id];

  // Don't unroll if the extent is larger than max_unroll
  if (max_unroll != -1 && it->range.defined()) {
    if (auto imm = it->range->extent.as<IntImmNode>()) {
      if (imm->value > max_unroll) {
        return it;
      }
    }
  }

  AnnotationStep step =
      AnnotationStep(stage_id, GetIndex(stage->iters, it), IteratorAnnotation::kUnroll);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

Iterator State::vectorize(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step =
      AnnotationStep(stage_id, GetIndex(stage->iters, it), IteratorAnnotation::kVectorize);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

Iterator State::fuse(int stage_id, const Array<Iterator>& iters) {
  const Stage& stage = operator->()->stages[stage_id];
  Array<Integer> indices;
  GetIndices(stage->iters, iters, &indices);
  FuseStep step = FuseStep(stage_id, indices);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

void State::pragma(int stage_id, const Iterator& it, const String& pragma_type) {
  const Stage& stage = operator->()->stages[stage_id];
  PragmaStep step = PragmaStep(stage_id, GetIndex(stage->iters, it), pragma_type);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

void State::reorder(int stage_id, const Array<Iterator>& order) {
  const Stage& stage = operator->()->stages[stage_id];
  ICHECK_EQ(order.size(), stage->iters.size()) << "The order of all iterators "
                                               << "should be specified";
  Array<Integer> after_ids;
  GetIndices(stage->iters, order, &after_ids);
  ReorderStep step = ReorderStep(stage_id, after_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  step->ApplyToState(this);
}

Array<Iterator> State::split(int stage_id, const Iterator& it,
                             const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];
  SplitStep step =
      SplitStep(stage_id, GetIndex(stage->iters, it),
                it->range.defined() ? it->range->extent : PrimExpr(), lengths, inner_to_outer);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

Array<Iterator> State::follow_split(int stage_id, const Iterator& it, int src_step_id,
                                    int n_split) {
  const Stage& stage = operator->()->stages[stage_id];
  FollowSplitStep step =
      FollowSplitStep(stage_id, GetIndex(stage->iters, it), src_step_id, n_split);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

Array<Iterator> State::follow_fused_split(int stage_id, const Iterator& it,
                                          const Array<Integer>& src_step_ids, int level,
                                          bool factor_or_nparts) {
  const Stage& stage = operator->()->stages[stage_id];
  FollowFusedSplitStep step = FollowFusedSplitStep(stage_id, GetIndex(stage->iters, it),
                                                   src_step_ids, level, factor_or_nparts);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

void State::storage_align(int stage_id, const Iterator& it, int factor, int offset) {
  const Stage& stage = operator->()->stages[stage_id];
  StorageAlignStep step = StorageAlignStep(stage_id, GetIndex(stage->iters, it), factor, offset);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this);
}

void State::compute_at(int stage_id, int target_stage_id, const Iterator& target_iter) {
  const Stage& target_stage = operator->()->stages[target_stage_id];
  ComputeAtStep step =
      ComputeAtStep(stage_id, target_stage_id, GetIndex(target_stage->iters, target_iter));
  CopyOnWrite()->transform_steps.push_back(step);
  step->ApplyToState(this);
}

void State::compute_inline(int stage_id) {
  ComputeInlineStep step = ComputeInlineStep(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  step->ApplyToState(this);
}

void State::compute_root(int stage_id) {
  ComputeRootStep step = ComputeRootStep(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  step->ApplyToState(this);
}

int State::cache_read(int stage_id, const String& scope_name,
                      const Array<Integer>& reader_stage_ids, const ComputeDAG& dag) {
  CacheReadStep step = CacheReadStep(stage_id, scope_name, reader_stage_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this, dag);
}

int State::cache_write(int stage_id, const String& scope_name, const ComputeDAG& dag) {
  CacheWriteStep step = CacheWriteStep(stage_id, scope_name);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this, dag);
}

int State::rfactor(int stage_id, const Iterator& it, int factor_iter_id, const ComputeDAG& dag) {
  const Stage& stage = operator->()->stages[stage_id];
  RfactorStep step = RfactorStep(stage_id, GetIndex(stage->iters, it), factor_iter_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return step->ApplyToState(this, dag);
}

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
      IterKey iter_key(stage_id, i);
      auto pair = state->attach_map->iter_to_attached_stages.find(iter_key);
      if (pair != state->attach_map->iter_to_attached_stages.end()) {
        // Print the attached stage
        for (const auto& attach_stage_id : pair->second) {
          PrintStage(os, attach_stage_id, state, base_indent + indent, delete_trivial_loop);
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
    .set_dispatch<StageNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto& stage = tvm::Downcast<Stage>(ref);
      p->stream << stage->GetTypeKey() << "(" << stage.get() << ": " << stage->op->name << ")";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StateNode>([](const ObjectRef& ref, ReprPrinter* p) {
      PrintState(&p->stream, tvm::Downcast<State>(ref), true);
    });

/********** State interface API for ffi **********/
TVM_REGISTER_GLOBAL("auto_scheduler.StateBind")
    .set_body_typed([](State state, int stage_id, const Iterator& it, int thread_type) {
      const auto& res = state.bind(stage_id, it, IteratorAnnotation(thread_type));
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateParallel")
    .set_body_typed([](State state, int stage_id, const Iterator& it) {
      const auto& res = state.parallel(stage_id, it);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateUnroll")
    .set_body_typed([](State state, int stage_id, const Iterator& it, int max_unroll) {
      const auto& res = state.unroll(stage_id, it, max_unroll);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateVectorize")
    .set_body_typed([](State state, int stage_id, const Iterator& it) {
      const auto& res = state.vectorize(stage_id, it);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateFuse")
    .set_body_typed([](State state, int stage_id, const Array<Iterator>& iters) {
      const auto& res = state.fuse(stage_id, iters);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StatePragma")
    .set_body_typed([](State state, int stage_id, const Iterator& it, const String& pragma_type) {
      state.pragma(stage_id, it, pragma_type);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateReorder")
    .set_body_typed([](State state, int stage_id, const Array<Iterator>& order) {
      state.reorder(stage_id, order);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateSplit")
    .set_body_typed([](State state, int stage_id, const Iterator& it,
                       const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
      const auto& res = state.split(stage_id, it, lengths, inner_to_outer);
      return Array<ObjectRef>{state, res};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateFollowSplit")
    .set_body_typed([](State state, int stage_id, const Iterator& it, int src_step_id,
                       int n_split) {
      const auto& res = state.follow_split(stage_id, it, src_step_id, n_split);
      return Array<ObjectRef>{state, Array<Iterator>(res)};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateFollowFusedSplit")
    .set_body_typed([](State state, int stage_id, const Iterator& it,
                       const Array<Integer>& src_step_ids, int level, bool factor_or_nparts) {
      const auto& res =
          state.follow_fused_split(stage_id, it, src_step_ids, level, factor_or_nparts);
      return Array<ObjectRef>{state, Array<Iterator>(res)};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateStorageAlign")
    .set_body_typed([](State state, int stage_id, const Iterator& it, int factor, int offset) {
      state.storage_align(stage_id, it, factor, offset);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateComputeAt")
    .set_body_typed([](State state, int stage_id, int target_stage_id,
                       const Iterator& target_iter) {
      state.compute_at(stage_id, target_stage_id, target_iter);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateComputeInline")
    .set_body_typed([](State state, int stage_id) {
      state.compute_inline(stage_id);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateComputeRoot")
    .set_body_typed([](State state, int stage_id) {
      state.compute_root(stage_id);
      return state;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateCacheRead")
    .set_body_typed([](State state, int stage_id, const String& scope_name,
                       const Array<Integer>& reader_stage_ids, const ComputeDAG& dag) {
      int res = state.cache_read(stage_id, scope_name, reader_stage_ids, dag);
      return Array<ObjectRef>{state, Integer(res)};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateCacheWrite")
    .set_body_typed([](State state, int stage_id, const String& scope_name,
                       const ComputeDAG& task_dag) {
      int res = state.cache_write(stage_id, scope_name, task_dag);
      return Array<ObjectRef>{state, Integer(res)};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateRfactor")
    .set_body_typed([](State state, int stage_id, const Iterator& it, int factor_iter_id,
                       const ComputeDAG& dag) {
      int res = state.rfactor(stage_id, it, factor_iter_id, dag);
      return Array<ObjectRef>{state, Integer(res)};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.StateEqual").set_body_typed([](State state1, State state2) {
  return std::equal_to<State>()(state1, state2);
});

}  // namespace auto_scheduler
}  // namespace tvm
