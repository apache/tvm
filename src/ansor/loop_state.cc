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
 * \file ansor/loop_state.h
 * \brief  An IR (intermediate representation) for loop structures.
 */

#include "loop_state.h"
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include "transform_step.h"
#include "utils.h"

namespace tvm {
namespace ansor {

TVM_REGISTER_OBJECT_TYPE(StepNode);
TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(StateNode);
TVM_REGISTER_NODE_TYPE(IteratorNode);

// Maker for other classes
Iterator IteratorNode::make(std::string name, Range range,
                            IteratorType iter_type, IteratorAnnotation annotation,
                            const std::vector<Iterator>* ori_iters,
                            std::string attr) {
  auto node = make_object<IteratorNode>();
  node->name = std::move(name);
  node->range = std::move(range);
  node->iter_type = iter_type;
  node->annotation = annotation;
  if (ori_iters != nullptr) {
    node->ori_iters = *ori_iters;
  }
  node->attr = std::move(attr);
  return Iterator(node);
}


Stage StageNode::make(te::Operation op) {
  auto node = make_object<StageNode>();
  if (op->IsInstance<te::ComputeOpNode>()) {
    node->op_type = kCompute;
    auto* pop = op.as<te::ComputeOpNode>();

    for (const auto& axis : pop->axis) {
      node->iters.push_back(IteratorNode::make(CleanName(axis->var->name_hint),
                                               axis->dom, kSpace, kNone));
    }
    for (const auto& axis : pop->reduce_axis) {
      node->iters.push_back(IteratorNode::make(CleanName(axis->var->name_hint),
                                               axis->dom, kReduce, kNone));
    }
  } else if (op->IsInstance<te::PlaceholderOpNode>()) {
    node->op_type = kPlaceholder;
  } else {
    LOG(FATAL) << "Unsupported operator type" << op->_type_key;
  }

  node->compute_at = kRoot;
  node->op = std::move(op);
  node->attrs.auto_unroll_max_step = 0;
  node->attrs.storage_offset = 0;
  return Stage(node);
}

Stage StageNode::make(te::Operation op, StageType op_type,
                      const std::vector<Iterator>& iters,
                      ComputeAtType compute_at, StageAttributes attrs) {
  auto node = make_object<StageNode>();
  node->op = std::move(op);
  node->op_type = op_type;
  node->iters = iters;
  node->compute_at = compute_at;
  node->attrs = attrs;
  return Stage(node);
}

Stage StageNode::make(te::Operation op, StageType op_type,
                      std::vector<Iterator>&& iters, ComputeAtType compute_at,
                      StageAttributes attrs) {
  auto node = make_object<StageNode>();
  node->op = std::move(op);
  node->op_type = op_type;
  node->iters = std::move(iters);
  node->compute_at = compute_at;
  node->attrs = attrs;
  return Stage(node);
}

State StateNode::make_empty_state() {
  auto node = make_object<StateNode>();
  node->attach_map = AttachMapNode::make();
  node->complete = false;
  node->aux_info = ObjectRef();
  return State(node);
}

State StateNode::make(const Array<te::Operation>& ops) {
  auto node = make_object<StateNode>();
  for (const auto& op : ops) {
    node->stages.push_back(StageNode::make(op));
  }
  node->attach_map = AttachMapNode::make();
  node->complete = true;
  node->aux_info = ObjectRef();
  return State(node);
}

State StateNode::make(const std::vector<Stage>& stages,
                      const std::vector<Step>& transform_steps, bool complete,
                      ObjectRef aux_info) {
  auto node = make_object<StateNode>();
  node->stages = stages;
  node->transform_steps = transform_steps;
  node->attach_map = AttachMapNode::make();
  node->complete = complete;
  node->aux_info = std::move(aux_info);
  return State(node);
}

AttachMap AttachMapNode::make() {
  auto node = make_object<AttachMapNode>();
  return AttachMap(node);
}

// Schedule primitives api
void State::reorder(int stage_id, const std::vector<Iterator>& order) {
  const Stage& stage = operator->()->stages[stage_id];

  CHECK_EQ(order.size(), stage->iters.size()) << "The order of all iterators "
                                                 "should be specified";
  std::vector<int> after_ids;
  GetIndices(stage->iters, order, &after_ids);
  ReorderStep step = ReorderStepNode::make(stage_id, after_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  DoReorderStep(step);
}

std::vector<Iterator> State::split(int stage_id, const Iterator& it,
                                   const std::vector<PrimExpr>& lengths,
                                   bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];

  SplitStep step =
      SplitStepNode::make(stage_id, GetIndex(stage->iters, it),
                          it->range.defined() ? it->range->extent : PrimExpr(),
                          lengths, inner_to_outer);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoSplitStep(step);
}

std::vector<Iterator> State::follow_split(int stage_id, const Iterator& it,
                                          int src_step_id, int n_split) {
  const Stage& stage = operator->()->stages[stage_id];

  FollowSplitStep step = FollowSplitStepNode::make(
      stage_id, GetIndex(stage->iters, it), src_step_id, n_split);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFollowSplitStep(step);
}

std::vector<Iterator> State::follow_fused_split(
    int stage_id, const Iterator& it, const std::vector<int>& src_step_ids,
    int level, bool factor_or_nparts) {
  const Stage& stage = operator->()->stages[stage_id];

  FollowFusedSplitStep step =
      FollowFusedSplitStepNode::make(stage_id, GetIndex(stage->iters, it),
                                     src_step_ids, level, factor_or_nparts);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFollowFusedSplitStep(step);
}

Iterator State::fuse(int stage_id, const std::vector<Iterator>& iters) {
  const Stage& stage = operator->()->stages[stage_id];
  std::vector<int> indices;
  GetIndices(stage->iters, iters, &indices);
  FuseStep step = FuseStepNode::make(stage_id, indices);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFuseStep(step);
}

Iterator State::vectorize(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step = AnnotationStepNode::make(
      stage_id, GetIndex(stage->iters, it), kVectorize);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

Iterator State::parallel(int stage_id, const Iterator& it) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step =
      AnnotationStepNode::make(stage_id, GetIndex(stage->iters, it), kParallel);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

Iterator State::unroll(int stage_id, const Iterator& it, int max_unroll) {
  const Stage& stage = operator->()->stages[stage_id];
  AnnotationStep step =
      AnnotationStepNode::make(stage_id, GetIndex(stage->iters, it), kUnroll);

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

void State::compute_at(int stage_id, int target_stage_id,
                       const Iterator& target_iter) {
  const Stage& target_stage = operator->()->stages[target_stage_id];
  ComputeAtStep step = ComputeAtStepNode::make(
      stage_id, target_stage_id, GetIndex(target_stage->iters, target_iter));
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeAtStep(step);
}

void State::compute_root(int stage_id) {
  ComputeRootStep step = ComputeRootStepNode::make(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeRootStep(step);
}

void State::compute_inline(int stage_id) {
  ComputeInlineStep step = ComputeInlineStepNode::make(stage_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoComputeInlineStep(step);
}

Iterator State::bind_thread(int stage_id, const Iterator& it,
                            IteratorAnnotation thread_type) {
  const Stage& stage = operator->()->stages[stage_id];
  if (thread_type < kVThread || thread_type > kThreadY) {
    LOG(FATAL) << "thread_type error, valide: kVThread, kBlockX, kThreadX, "
               << "kThreadY";
  }
  AnnotationStep step = AnnotationStepNode::make(
      stage_id, GetIndex(stage->iters, it), thread_type);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoAnnotationStep(step);
}

int State::cache_read(int stage_id, const std::string& scope_name,
                      const std::vector<int>& reader_stage_ids,
                      const ComputeDAG& task_dag) {
  CacheReadStep step =
      CacheReadStepNode::make(stage_id, scope_name, reader_stage_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoCacheReadStep(step, task_dag);
}

int State::cache_write(int stage_id, const std::string& scope_name,
                       const ComputeDAG& task_dag) {
  CacheWriteStep step = CacheWriteStepNode::make(stage_id, scope_name);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoCacheWriteStep(step, task_dag);
}

void State::pragma(int stage_id, const Iterator& it,
                   const std::string& pragma_type) {
  const Stage& stage = operator->()->stages[stage_id];
  PragmaStep step =
      PragmaStepNode::make(stage_id, GetIndex(stage->iters, it), pragma_type);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoPragmaStep(step);
}

int State::rfactor(int stage_id, const Iterator& it, int factor_iter_id,
                   const ComputeDAG& task_dag) {
  const Stage& stage = operator->()->stages[stage_id];
  RfactorStep step = RfactorStepNode::make(stage_id, GetIndex(stage->iters, it),
                                           factor_iter_id);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoRfactorStep(step, task_dag);
}

void State::storage_align(int stage_id, const Iterator& it, int factor,
                          int offset) {
  const Stage& stage = operator->()->stages[stage_id];
  StorageAlignStep step = StorageAlignStepNode::make(
      stage_id, GetIndex(stage->iters, it), factor, offset);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoStorageAlignStep(step);
}

Iterator State::tensorize(int stage_id, const Iterator& it,
                          std::string ti_func_name) {
  const Stage& stage = operator->()->stages[stage_id];
  TensorizeStep step = TensorizeStepNode::make(
      stage_id, GetIndex(stage->iters, it), ti_func_name);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoTensorizeStep(step);
}

// Steps' implementations
void State::DoReorderStep(const ReorderStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  std::vector<Iterator> iters;
  for (auto x : step->after_ids) {
    iters.push_back(stage->iters[x]);
  }

  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = StageNode::make(
      stage->op, stage->op_type, std::move(iters), stage->compute_at,
      stage->attrs);
}

// common part for DoSplitStep, DoFollowSplitStep, and DoFollowFusedSplitStep
std::vector<Iterator> State::DoSplitStepCommon(
    int stage_id, int iter_id, const std::vector<PrimExpr>& lengths,
    bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];
  const Iterator& it = stage->iters[iter_id];
  size_t old_iter_size = stage->iters.size();

  PrimExpr tosplit_min, tosplit_extent;
  if (it->range.defined()) {
    tosplit_min = it->range->min;
    tosplit_extent = it->range->extent;
  } else {
    tosplit_min = tosplit_extent = PrimExpr();
  }

  std::vector<Iterator> outs;
  for (size_t i = 0; i < lengths.size(); ++i) {
    PrimExpr l;
    std::string name;
    if (inner_to_outer) {
      l = lengths[lengths.size() - i - 1];
      name = it->name + "." + std::to_string(lengths.size() - i);
    } else {
      l = lengths[i];
      name = it->name + "." + std::to_string(i);
    }
    Iterator res;
    if (l.defined() && tosplit_min.defined() && tosplit_extent.defined()) {
      res = IteratorNode::make(name, Range::make_by_min_extent(tosplit_min, l),
                               it->iter_type, kNone);
      tosplit_min = 0;
      tosplit_extent = indexdiv(tosplit_extent + l - 1, l);
    } else {
      res = IteratorNode::make(name, Range(), it->iter_type, kNone);
      tosplit_min = tosplit_extent = PrimExpr();
    }
    outs.push_back(std::move(res));
  }

  Range range;
  if (tosplit_min.defined() && tosplit_extent.defined()) {
    range = Range::make_by_min_extent(tosplit_min, tosplit_extent);
  }
  if (inner_to_outer) {
    outs.push_back(
        IteratorNode::make(it->name + ".0", range, it->iter_type, kNone));
    std::reverse(outs.begin(), outs.end());
  } else {
    outs.push_back(
        IteratorNode::make(it->name + "." + std::to_string(lengths.size()),
                           range, it->iter_type, kNone));
  }

  std::vector<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(),
                   stage->iters.begin() + iter_id);
  new_iters.insert(new_iters.end(), outs.begin(), outs.end());
  new_iters.insert(new_iters.end(), stage->iters.begin() + iter_id + 1,
                   stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages[stage_id] = StageNode::make(
      stage->op, stage->op_type, std::move(new_iters), stage->compute_at,
      stage->attrs);

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

std::vector<Iterator> State::DoSplitStep(const SplitStep& step) {
  return DoSplitStepCommon(step->stage_id, step->iter_id, step->lengths,
                           step->inner_to_outer);
}

std::vector<Iterator> State::DoFollowSplitStep(const FollowSplitStep& step) {
  std::vector<PrimExpr> lengths;
  step->ExtractSplitLengths(operator->()->transform_steps, &lengths);
  return DoSplitStepCommon(step->stage_id, step->iter_id, lengths, true);
}

std::vector<Iterator> State::DoFollowFusedSplitStep(
    const FollowFusedSplitStep& step) {
  const PrimExpr& length =
      step->ExtractSplitLength(operator->()->transform_steps);
  return DoSplitStepCommon(step->stage_id, step->iter_id, {length},
                           step->factor_or_nparts);
}

Iterator State::DoFuseStep(const FuseStep& step) {
  int stage_id = step->stage_id;
  const Stage& stage = operator->()->stages[stage_id];
  int old_iter_size = static_cast<int>(stage->iters.size());

  std::string new_name;
  PrimExpr new_extent = 1;
  IteratorType new_iter_type = kSpecial;

  std::vector<Iterator> ori_iters;
  for (size_t i = 0; i < step->fused_ids.size(); ++i) {
    if (i > 0) {
      CHECK_EQ(step->fused_ids[i], step->fused_ids[i - 1] + 1);
    }

    if (i != step->fused_ids.size() - 1) {
      const auto& iter_to_attached_stage =
      operator->()->attach_map->iter_to_attached_stages;
      if (iter_to_attached_stage.find(std::make_pair(
              stage_id, step->fused_ids[i])) != iter_to_attached_stage.end()) {
        LOG(FATAL) << "Invalid Fuse. Because you want to fuse iterators "
                      "that have been attached by some stages";
      }
    }

    const Iterator& it = stage->iters[step->fused_ids[i]];
    ori_iters.push_back(it);
    new_name += it->name + "@";

    if (it->range.defined() && new_extent.defined()) {
      new_extent = new_extent * it->range->extent;
    } else {
      new_extent = PrimExpr();
    }

    if (i == 0) {
      new_iter_type = it->iter_type;
    } else {
      if (new_iter_type != it->iter_type) {
        new_iter_type = kMixed;
      }
    }
  }

  Range range;
  if (new_extent.defined()) {
    range = Range::make_by_min_extent(0, new_extent);
  }
  Iterator new_it =
      IteratorNode::make(new_name, range, new_iter_type, kNone, &ori_iters);
  std::vector<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(),
                   stage->iters.begin() + step->fused_ids.front());
  new_iters.push_back(new_it);
  new_iters.insert(new_iters.end(),
                   stage->iters.begin() + step->fused_ids.back() + 1,
                   stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages[stage_id] = StageNode::make(
      stage->op, stage->op_type, std::move(new_iters), stage->compute_at,
      stage->attrs);

  // we have to replace the iterators in attach map,
  // these two vectors keep the replacement mapping
  std::vector<AttachMap::IterKey> from_iters;
  std::vector<AttachMap::IterKey> to_iters;
  const int begin_id = step->fused_ids.front(), end_id = step->fused_ids.back();
  for (int i = 0; i < old_iter_size; ++i) {
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

  CHECK_EQ(it->annotation, IteratorAnnotation::kNone);
  Iterator new_it = IteratorNode::make(it->name, it->range, it->iter_type,
                                       step->annotation, &it->ori_iters,
                                       it->attr);
  Stage new_stage = stage;
  new_stage.CopyOnWrite()->iters[step->iter_id] = new_it;
  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = std::move(new_stage);
  return new_it;
}

void State::DoComputeAtStep(const ComputeAtStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  // after compute_at, we don't know the accurate length information any more
  // If we do want to know the accurate lengths, we can call
  // ComputeDAG::ReplayAndInferBound
  std::vector<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    size_t s = it->name.size();
    if (s >= 2 && it->name[s - 2] == '.' && it->name[s - 1] >= '1' &&
        it->name[s - 1] <= '4') {
      // We use a dangerous heuristic rule here : For multi level splitted
      // iterators, we assume their length does not change after compute_at.
      // Reason: These iterators are generated in MultiStagePolicy by multi
      // level tiling, they will be carefully compute_at their consumers.
      // In this case, their lengths do not change.
      // We do this to keep the AnnotateCPU pass to annotate more efficiently.
      new_iters.push_back(it);
    } else {
      new_iters.push_back(IteratorNode::make(it->name, Range(), it->iter_type,
                                             it->annotation, &it->ori_iters,
                                             it->attr));
    }
  }

  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] =
      StageNode::make(stage->op, stage->op_type, std::move(new_iters), kIter,
                      stage->attrs);
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
    new_iters.push_back(IteratorNode::make(it->name, Range(), it->iter_type,
                                           it->annotation, &it->ori_iters,
                                           it->attr));
  }

  // update attach map
  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] =
      StageNode::make(stage->op, stage->op_type, std::move(new_iters), kRoot,
                      stage->attrs);
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

  pstate->stages[step->stage_id].CopyOnWrite()->compute_at = kInlined;
  pstate->attach_map.DeleteStage(step->stage_id);
}

// Common part for steps that add new stages
// (e.g. CacheReadStep, CacheWriteStep, RfactorStep)
void AddStageModificationSteps(size_t step_id,
                               const std::vector<Step>& transform_steps,
                               std::vector<Step>* replay_steps) {
  const Step& step = transform_steps[step_id];
  if (step->IsInstance<CacheWriteStepNode>() ||
      step->IsInstance<CacheReadStepNode>()) {
    replay_steps->push_back(step);
  } else if (step->IsInstance<RfactorStepNode>()) {
    // add FuseStepNode required by rfactor
    if (step_id >= 2 &&
        transform_steps[step_id - 2]->IsInstance<FuseStepNode>()) {
      const Step& fuse_step = transform_steps[step_id - 2];
      if (fuse_step->stage_id == step->stage_id) {
        replay_steps->push_back(fuse_step);
      }
    }
    // add SplitStepNode required by rfactor
    CHECK_GE(step_id, 1);
    CHECK(transform_steps[step_id - 1]->IsInstance<SplitStepNode>());
    const Step& split_step = transform_steps[step_id - 1];
    CHECK_EQ(split_step->stage_id, step->stage_id);
    replay_steps->push_back(split_step);
    // add RfactorStepNode
    replay_steps->push_back(step);
  }
}

int State::DoCacheReadStep(const CacheReadStep& step, const ComputeDAG& dag) {
  StateNode* pstate = CopyOnWrite();
  std::vector<Step> replay_steps;
  for (size_t i = 0; i < pstate->transform_steps.size(); ++i) {
    AddStageModificationSteps(i, pstate->transform_steps, &replay_steps);
    if (pstate->transform_steps[i].same_as(step)) {
      break;
    }
  }
  dag.ReplayAndGetDAG(replay_steps, &(pstate->task_dag));

  // target -> target + target_store
  // Should update target's op, insert new stage, update the later stage's op
  pstate->stages[step->stage_id].CopyOnWrite()->op =
  operator->()->task_dag->ops[step->stage_id];
  pstate->stages.insert(
      pstate->stages.begin() + step->stage_id + 1,
      StageNode::make(operator->()->task_dag->ops[step->stage_id + 1]));
  for (size_t i = step->stage_id + 2; i < operator->()->stages.size(); ++i) {
    pstate->stages[i].CopyOnWrite()->op = operator->()->task_dag->ops[i];
  }
  pstate->attach_map = operator->()->attach_map.ApplyStageIdOfffset(
      step->stage_id + 1, 1);

  return step->stage_id + 1;
}

int State::DoCacheWriteStep(const CacheWriteStep& step, const ComputeDAG& dag) {
  StateNode* pstate = CopyOnWrite();
  std::vector<Step> replay_steps;
  for (size_t i = 0; i < pstate->transform_steps.size(); ++i) {
    AddStageModificationSteps(i, pstate->transform_steps, &replay_steps);
    if (pstate->transform_steps[i].same_as(step)) {
      break;
    }
  }

  int last_dag_op_size = pstate->task_dag.defined()
                             ? pstate->task_dag->ops.size()
                             : dag->ops.size();
  dag.ReplayAndGetDAG(replay_steps, &(pstate->task_dag));
  int added_ops = pstate->task_dag->ops.size() - last_dag_op_size;
  CHECK_GE(added_ops, 1);

  // target -> target_compute + target
  // Assume target stage has never been applied any steps before cache_write
  // Should insert new stage, update target stage, update the later stage's op
  pstate->stages.insert(
      pstate->stages.begin() + step->stage_id,
      StageNode::make(operator->()->task_dag->ops[step->stage_id]));
  pstate->stages[step->stage_id + 1] =
      StageNode::make(operator->()->task_dag->ops[step->stage_id + 1]);
  int next_stage_id = step->stage_id + 2;
  // Notice: added_ops should actually assert to be 1
  // branch of 2 here is somehow a hack to TVM's cache_write bug with
  // multi outputs, see test/cpp/ansor_test.cc: CacheReadWrite test
  // for more information
  // TODO(jcf94): Fix this
  if (added_ops == 2) {
    pstate->stages.insert(
        pstate->stages.begin() + next_stage_id,
        StageNode::make(operator->()->task_dag->ops[next_stage_id]));
    next_stage_id++;
  } else if (added_ops > 2) {
    LOG(ERROR) << "Unexpected behavior of CacheWrite.";
  }
  for (size_t i = next_stage_id; i < operator->()->task_dag->ops.size(); ++i) {
    pstate->stages[i].CopyOnWrite()->op = operator->()->task_dag->ops[i];
  }
  pstate->attach_map = operator->()->attach_map.ApplyStageIdOfffset(
      step->stage_id, added_ops);

  return step->stage_id;
}

void State::DoPragmaStep(const PragmaStep& step) {
  if (step->pragma_type == "debug_skip_region") {
    StateNode* pstate = CopyOnWrite();
    pstate->attach_map.DeleteStage(step->stage_id);
  } else if (StrStartsWith(step->pragma_type, "auto_unroll_max_step")) {
    StateNode* pstate = CopyOnWrite();
    StageNode* stage = pstate->stages[step->stage_id].CopyOnWrite();
    size_t pos = step->pragma_type.find('$');
    stage->attrs.auto_unroll_max_step = atoi(step->pragma_type.c_str() + pos + 1);
  } else if (step->pragma_type == "tensor_core") {
    // Nothing needs to be done here
  } else {
    LOG(FATAL) << "Invalid pragma: " << step->pragma_type;
  }
}

int State::DoRfactorStep(const RfactorStep& step, const ComputeDAG& dag) {
  StateNode* pstate = CopyOnWrite();
  const auto compute_at_type = pstate->stages[step->stage_id]->compute_at;
  std::vector<Step> replay_steps;
  for (size_t i = 0; i < pstate->transform_steps.size(); ++i) {
    AddStageModificationSteps(i, pstate->transform_steps, &replay_steps);
    if (pstate->transform_steps[i].same_as(step)) {
      break;
    }
  }
  dag.ReplayAndGetDAG(replay_steps, &(pstate->task_dag));

  // target -> target_compute + target
  // Should insert new stage, update target stage, update the later stage's op
  pstate->stages.insert(
      pstate->stages.begin() + step->stage_id,
      StageNode::make(operator->()->task_dag->ops[step->stage_id]));
  // maintain the compute_at type of target stage
  Stage target_stage =
      StageNode::make(operator->()->task_dag->ops[step->stage_id + 1]);
  target_stage.CopyOnWrite()->compute_at = compute_at_type;
  pstate->stages[step->stage_id + 1] = target_stage;

  for (size_t i = step->stage_id + 2; i < operator->()->stages.size(); ++i) {
    pstate->stages[i].CopyOnWrite()->op = operator->()->task_dag->ops[i];
  }
  pstate->attach_map = operator->()->attach_map.ApplyStageIdOfffset(
      step->stage_id, 1);

  return step->stage_id;
}

void State::DoStorageAlignStep(const StorageAlignStep& step) {
  StateNode* pstate = CopyOnWrite();
  StageNode* stage = pstate->stages[step->stage_id].CopyOnWrite();
  stage->attrs.storage_offset = step->offset;
}

Iterator State::DoTensorizeStep(const TensorizeStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];
  Iterator it = stage->iters[step->iter_id];
  Iterator new_it = IteratorNode::make(it->name, it->range, it->iter_type,
      IteratorAnnotation::kTensorized, &it->ori_iters, step->ti_func_name);
  Stage new_stage = stage;
  new_stage.CopyOnWrite()->iters[step->iter_id] = new_it;
  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = std::move(new_stage);
  return new_it;
}

void State::DoStep(const Step& step, const ComputeDAG& dag) {
  if (auto ps = step.as<ReorderStepNode>()) {
    DoReorderStep(GetRef<ReorderStep>(ps));
  } else if (auto ps = step.as<SplitStepNode>()) {
    DoSplitStep(GetRef<SplitStep>(ps));
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    DoFollowSplitStep(GetRef<FollowSplitStep>(ps));
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    DoFollowFusedSplitStep(GetRef<FollowFusedSplitStep>(ps));
  } else if (auto ps = step.as<FuseStepNode>()) {
    DoFuseStep(GetRef<FuseStep>(ps));
  } else if (auto ps = step.as<AnnotationStepNode>()) {
    DoAnnotationStep(GetRef<AnnotationStep>(ps));
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    DoComputeAtStep(GetRef<ComputeAtStep>(ps));
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    DoComputeRootStep(GetRef<ComputeRootStep>(ps));
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    DoComputeInlineStep(GetRef<ComputeInlineStep>(ps));
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    DoCacheReadStep(GetRef<CacheReadStep>(ps), dag);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    DoCacheWriteStep(GetRef<CacheWriteStep>(ps), dag);
  } else if (auto ps = step.as<PragmaStepNode>()) {
    DoPragmaStep(GetRef<PragmaStep>(ps));
  } else if (auto ps = step.as<RfactorStepNode>()) {
    DoRfactorStep(GetRef<RfactorStep>(ps), dag);
  } else if (auto ps = step.as<StorageAlignStepNode>()) {
    DoStorageAlignStep(GetRef<StorageAlignStep>(ps));
  } else if (auto ps = step.as<TensorizeStepNode>()) {
    DoTensorizeStep(GetRef<TensorizeStep>(ps));
  } else {
    LOG(FATAL) << "Invalid step: " << step;
  }
}

void State::DoSteps(const std::vector<Step>& steps, const ComputeDAG& dag) {
  // Use complete rate for the study in the paper
  const char* complete_rate_str = getenv("ANSOR_PROGRAM_COMPLETE_RATE");
  double complete_rate = -1.0;
  if (complete_rate_str) {
    complete_rate = std::stod(complete_rate_str);
  }
  size_t ct = 0;

  for (const auto& step : steps) {
    if (complete_rate >= 0 && ct++ > steps.size() * complete_rate) {
      break;
    }
    DoStep(step, dag);
  }
}

void PrintStage(std::ostream* os, int stage_id, const StateNode* state,
                size_t base_indent, bool delete_trivial_loop) {
  const Stage& stage = state->stages[stage_id];

  if (stage->attrs.auto_unroll_max_step != 0) {
    for (size_t j = 0; j < base_indent; ++j) {
      *os << " ";
    }
    *os << stage->op->func_name()
        << " auto_unroll: " << stage->attrs.auto_unroll_max_step << "\n";
  }
  if (stage->attrs.storage_offset != 0) {
    for (size_t j = 0; j < base_indent; ++j) {
      *os << " ";
    }
    *os << stage->op->func_name()
        << " storage_offset: " << stage->attrs.storage_offset << "\n";
  }

  size_t indent = 0;
  for (size_t i = 0; i < stage->iters.size(); ++i) {
    const Iterator& iter = stage->iters[i];

    if (!(delete_trivial_loop && iter->range.defined() &&
          is_one(iter->range->extent))) {
      for (size_t j = 0; j < base_indent + indent; ++j) {
        *os << " ";
      }
      switch (iter->annotation) {
        case kNone:
          *os << "for ";
          break;
        case kUnroll:
          *os << "unroll ";
          break;
        case kParallel:
          *os << "parallel ";
          break;
        case kVectorize:
          *os << "vectorize ";
          break;
        case kVThread:
          *os << "vthread ";
          break;
        case kBlockX:
          *os << "gpu.blockIdx.x ";
          break;
        case kBlockY:
          *os << "gpu.blockIdx.y ";
          break;
        case kThreadX:
          *os << "gpu.threadIdx.x ";
          break;
        case kThreadY:
          *os << "gpu.threadIdx.y ";
          break;
        case kTensorized:
          *os << "tensorize ";
          break;
        default:
          LOG(FATAL) << "Invalid Annotation " << iter->annotation; break;
      }
      if (iter->range.defined()) {
        *os << iter->name << " (" << iter->range->min << ","
            << iter->range->extent << ")";
      } else {
        *os << iter->name << " (None)";
      }
      if (!iter->attr.empty()) {
        *os << " " << iter->attr;
      }
      *os << "\n";

      indent += 2;
    }

    if (state != nullptr) {
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
  *os << stage->op->func_name() << " = ...\n";
}

void PrintState(std::ostream* os, const StateNode* node,
                bool delete_trivial_loop) {
  // Gather placeholders
  std::vector<std::string> placeholders;
  for (const auto& stage : node->stages) {
    if (stage->op_type == kPlaceholder) {
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
  for (size_t i = 0; i < node->stages.size(); ++i) {
    const Stage& stage = node->stages[i];
    if (stage->op_type == kPlaceholder) {
      continue;
    } else if (stage->op_type == kCompute) {
      if (stage->compute_at == kRoot) {
        PrintStage(os, i, node, 0, delete_trivial_loop);
      }
    } else {
      LOG(FATAL) << "Invalid op type";
    }
  }
}

std::string State::ToStr(bool delete_trivial_loop) const {
  std::ostringstream os;
  PrintState(&os, operator->(), delete_trivial_loop);
  return os.str();
}

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
  AttachMap map = AttachMapNode::make();
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<StateNode>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const StateNode*>(ref.get());
  PrintState(&p->stream, node, true);
});

TVM_REGISTER_GLOBAL("ansor.StageGetIterators").set_body_typed([](const Stage& stage) {
  return Array<Iterator>(stage->iters);
});

TVM_REGISTER_GLOBAL("ansor.StateGetStages").set_body_typed([](const State& state) {
  return Array<Stage>(state->stages);
});

TVM_REGISTER_GLOBAL("ansor.StateGetTransformStepsSize").set_body_typed([](const State& state) {
  return static_cast<int64_t>(state->transform_steps.size());
});

TVM_REGISTER_GLOBAL("ansor.StateReorder")
.set_body_typed([](State state, int stage_id, const Array<Iterator>& order) {
  std::vector<Iterator> ord;
  for (const auto& i : order) {
    ord.push_back(i);
  }
  state.reorder(stage_id, ord);
  return state;
});

TVM_REGISTER_GLOBAL("ansor.StateSplit")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   const Array<PrimExpr>& lengths, bool inner_to_outer) {
  std::vector<PrimExpr> len;
  for (const auto& i : lengths) {
    len.push_back(i);
  }
  const auto& res = state.split(stage_id, it, len, inner_to_outer);
  return Array<ObjectRef>{state, Array<Iterator>(res)};
});

TVM_REGISTER_GLOBAL("ansor.StateFollowSplit")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   int src_step_id, int n_split) {
  const auto& res = state.follow_split(stage_id, it, src_step_id, n_split);
  return Array<ObjectRef>{state, Array<Iterator>(res)};
});

TVM_REGISTER_GLOBAL("ansor.StateFollowFusedSplit")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   const Array<IntImm>& src_step_ids, int level,
                   bool factor_or_nparts) {
  std::vector<int> array_src_step_ids;
  for (const auto& i : src_step_ids) {
    array_src_step_ids.push_back(i->value);
  }
  const auto& res = state.follow_fused_split(
      stage_id, it, array_src_step_ids, level, factor_or_nparts);
  return Array<ObjectRef>{state, Array<Iterator>(res)};
});

TVM_REGISTER_GLOBAL("ansor.StateFuse")
.set_body_typed([](State state, int stage_id,
                   const Array<Iterator>& iters) {
  std::vector<Iterator> its;
  for (const auto& i : iters) {
    its.push_back(i);
  }
  const auto& res = state.fuse(stage_id, its);
  return Array<ObjectRef>{state, res};
});

TVM_REGISTER_GLOBAL("ansor.StateVectorize")
.set_body_typed([](State state, int stage_id, const Iterator& it) {
  const auto& res = state.vectorize(stage_id, it);
  return Array<ObjectRef>{state, res};
});

TVM_REGISTER_GLOBAL("ansor.StateParallel")
.set_body_typed([](State state, int stage_id, const Iterator& it) {
  const auto& res = state.parallel(stage_id, it);
  return Array<ObjectRef>{state, res};
});

TVM_REGISTER_GLOBAL("ansor.StateUnroll")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   int max_unroll) {
  const auto& res = state.unroll(stage_id, it, max_unroll);
  return Array<ObjectRef>{state, res};
});

TVM_REGISTER_GLOBAL("ansor.StateBindThread")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   int thread_type) {
  const auto& res =
      state.bind_thread(stage_id, it, IteratorAnnotation(thread_type));
  return Array<ObjectRef>{state, res};
});

TVM_REGISTER_GLOBAL("ansor.StateComputeAt")
.set_body_typed([](State state, int stage_id, int target_stage_id,
                   const Iterator& target_iter) {
  state.compute_at(stage_id, target_stage_id, target_iter);
  return state;
});

TVM_REGISTER_GLOBAL("ansor.StateComputeRoot")
.set_body_typed([](State state, int stage_id) {
  state.compute_root(stage_id);
  return state;
});

TVM_REGISTER_GLOBAL("ansor.StateComputeInline")
.set_body_typed([](State state, int stage_id) {
  state.compute_inline(stage_id);
  return state;
});

TVM_REGISTER_GLOBAL("ansor.StateCacheRead")
.set_body_typed([](State state, int stage_id, const std::string& scope_name,
                   const Array<IntImm>& reader_stage_ids,
                   const ComputeDAG& task_dag) {
  std::vector<int> array_reader_stage_ids;
  for (const auto& i : reader_stage_ids) {
    array_reader_stage_ids.push_back(i->value);
  }
  int res = state.cache_read(stage_id, scope_name, array_reader_stage_ids,
                             task_dag);
  return Array<ObjectRef>{state, IntImm(DataType::Int(32), res)};
});

TVM_REGISTER_GLOBAL("ansor.StateCacheWrite")
.set_body_typed([](State state, int stage_id, const std::string& scope_name,
                   const ComputeDAG& task_dag) {
  int res = state.cache_write(stage_id, scope_name, task_dag);
  return Array<ObjectRef>{state, IntImm(DataType::Int(32), res)};
});

TVM_REGISTER_GLOBAL("ansor.StatePragma")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   const std::string& pragma_type) {
  state.pragma(stage_id, it, pragma_type);
  return state;
});

TVM_REGISTER_GLOBAL("ansor.StateRfactor")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   int factor_iter_id, const ComputeDAG& task_dag) {
  int res = state.rfactor(stage_id, it, factor_iter_id, task_dag);
  return Array<ObjectRef>{state, IntImm(DataType::Int(32), res)};
});

TVM_REGISTER_GLOBAL("ansor.StateStorageAlign")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   int factor, int offset) {
  state.storage_align(stage_id, it, factor, offset);
  return state;
});

TVM_REGISTER_GLOBAL("ansor.StateTensorize")
.set_body_typed([](State state, int stage_id, const Iterator& it,
                   std::string ti_func) {
  const auto& res = state.tensorize(stage_id, it, ti_func);
  return Array<ObjectRef>{state, res};
});

TVM_REGISTER_GLOBAL("ansor.StateEqual")
.set_body_typed([](State state1, State state2) {
  return std::equal_to<State>()(state1, state2);
});

}  // namespace ansor
}  // namespace tvm
