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
 * \file auto_scheduler/transform_step.cc
 * \brief Transformation steps. These steps are used to manipulate the LoopState.
 *        They are similar to the schedule primitives in te::Stage.
 */

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

#include <string>
#include <utility>
#include <vector>

#include "utils.h"

namespace dmlc {
namespace json {

template <>
struct Handler<::tvm::Array<::tvm::Integer>> {
  inline static void Write(dmlc::JSONWriter* writer, const ::tvm::Array<::tvm::Integer>& array) {
    writer->BeginArray(false);
    for (const auto& i : array) {
      ICHECK(i.defined());
      writer->WriteArrayItem(i->value);
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader, ::tvm::Array<::tvm::Integer>* array) {
    array->clear();
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      int value;
      Handler<int>::Read(reader, &value);
      array->push_back(value);
    }
  }
};

template <>
struct Handler<::tvm::Array<::tvm::Optional<::tvm::Integer>>> {
  inline static void Write(dmlc::JSONWriter* writer,
                           const ::tvm::Array<::tvm::Optional<::tvm::Integer>>& array) {
    writer->BeginArray(false);
    for (const auto& i : array) {
      ICHECK(i);
      writer->WriteArrayItem(i.value()->value);
    }
    writer->EndArray();
  }
  inline static void Read(dmlc::JSONReader* reader,
                          ::tvm::Array<::tvm::Optional<::tvm::Integer>>* array) {
    array->clear();
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      int value;
      Handler<int>::Read(reader, &value);
      array->push_back(::tvm::Integer(value));
    }
  }
};

}  // namespace json
}  // namespace dmlc

namespace tvm {
namespace auto_scheduler {

// Update the te::stage to tir::IterVar axis mapping
void UpdateStageToAxesMap(const te::Stage& stage, StageToAxesMap* stage_to_axes) {
  if (auto pop = stage->op.as<te::ComputeOpNode>()) {
    Array<IterVar> axes;
    for (const auto& axis : pop->axis) {
      axes.push_back(axis);
    }
    for (const auto& axis : pop->reduce_axis) {
      axes.push_back(axis);
    }
    stage_to_axes->Set(stage, std::move(axes));
  } else if (stage->op->IsInstance<te::PlaceholderOpNode>()) {
    {}  // do nothing on Placeholder
  } else {
    LOG(FATAL) << "Invalid op " << stage->op;
  }
}

const char* IteratorAnnotationString[] = {
    "for",          // kNone = 0
    "unroll",       // kUnroll = 1
    "vectorize",    // kVectorize = 2
    "parallel",     // kParallel = 3
    "vthread",      // kVThread = 4
    "blockIdx.x",   // kBlockX = 5
    "threadIdx.x",  // kThreadX = 6
    "blockIdx.y",   // kBlockY = 7
    "threadIdx.y",  // kThreadY = 8
    "blockIdx.z",   // kBlockZ = 9
    "threadIdx.z",  // kThreadZ = 10
    "tensorize"     // kTensorized = 11
};

StepNode* Step::CopyOnWrite() {
  CHECK(data_ != nullptr);
  if (!data_.unique()) {
    if (const auto& ps = as<AnnotationStepNode>()) {
      auto n = make_object<AnnotationStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<FuseStepNode>()) {
      auto n = make_object<FuseStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<PragmaStepNode>()) {
      auto n = make_object<PragmaStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<ReorderStepNode>()) {
      auto n = make_object<ReorderStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<SplitStepNode>()) {
      auto n = make_object<SplitStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<FollowSplitStepNode>()) {
      auto n = make_object<FollowSplitStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<FollowFusedSplitStepNode>()) {
      auto n = make_object<FollowFusedSplitStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<StorageAlignStepNode>()) {
      auto n = make_object<StorageAlignStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<ComputeAtStepNode>()) {
      auto n = make_object<ComputeAtStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<ComputeInlineStepNode>()) {
      auto n = make_object<ComputeInlineStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<ComputeRootStepNode>()) {
      auto n = make_object<ComputeRootStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<CacheReadStepNode>()) {
      auto n = make_object<CacheReadStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<CacheWriteStepNode>()) {
      auto n = make_object<CacheWriteStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else if (const auto& ps = as<RfactorStepNode>()) {
      auto n = make_object<RfactorStepNode>(*ps);
      ObjectPtr<Object>(std::move(n)).swap(data_);
    } else {
      LOG(FATAL) << "Invalid step: " << (*this);
    }
  }
  return static_cast<StepNode*>(data_.get());
}

Step StepReadFromRecord(dmlc::JSONReader* reader) {
  std::string name;
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&name);
  if (name == AnnotationStepNode::record_prefix_str) {
    return AnnotationStep(reader);
  } else if (name == FuseStepNode::record_prefix_str) {
    return FuseStep(reader);
  } else if (name == PragmaStepNode::record_prefix_str) {
    return PragmaStep(reader);
  } else if (name == ReorderStepNode::record_prefix_str) {
    return ReorderStep(reader);
  } else if (name == SplitStepNode::record_prefix_str) {
    return SplitStep(reader);
  } else if (name == FollowSplitStepNode::record_prefix_str) {
    return FollowSplitStep(reader);
  } else if (name == FollowFusedSplitStepNode::record_prefix_str) {
    return FollowFusedSplitStep(reader);
  } else if (name == StorageAlignStepNode::record_prefix_str) {
    return StorageAlignStep(reader);
  } else if (name == ComputeAtStepNode::record_prefix_str) {
    return ComputeAtStep(reader);
  } else if (name == ComputeInlineStepNode::record_prefix_str) {
    return ComputeInlineStep(reader);
  } else if (name == ComputeRootStepNode::record_prefix_str) {
    return ComputeRootStep(reader);
  } else if (name == CacheReadStepNode::record_prefix_str) {
    return CacheReadStep(reader);
  } else if (name == CacheWriteStepNode::record_prefix_str) {
    return CacheWriteStep(reader);
  } else if (name == RfactorStepNode::record_prefix_str) {
    return RfactorStep(reader);
  } else {
    LOG(FATAL) << "Invalid step format: " << name;
  }
  return Step();
}

void StepApplyToState(const Step& step, State* state, const ComputeDAG& dag) {
  // We need this runtime dispatcher because different steps have different function signatures
  if (auto ps = step.as<AnnotationStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<FuseStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<PragmaStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ReorderStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<SplitStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<StorageAlignStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    ps->ApplyToState(state);
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    ps->ApplyToState(state, dag);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    ps->ApplyToState(state, dag);
  } else if (auto ps = step.as<RfactorStepNode>()) {
    ps->ApplyToState(state, dag);
  } else {
    LOG(FATAL) << "Invalid step: " << step;
  }
}

void StepApplyToSchedule(const Step& step, Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                         te::Schedule* schedule, const Array<Step>& transform_steps) {
  if (auto ps = step.as<AnnotationStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<FuseStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<PragmaStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ReorderStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<SplitStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, transform_steps);
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, transform_steps);
  } else if (auto ps = step.as<StorageAlignStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes);
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<RfactorStepNode>()) {
    ps->ApplyToSchedule(stages, stage_to_axes, schedule);
  } else {
    LOG(FATAL) << "Invalid Step: " << step;
  }
}

String StepPrintAsPythonAPI(const Step& step, Array<te::Stage>* stages,
                            StageToAxesMap* stage_to_axes, te::Schedule* schedule,
                            const Array<Step>& transform_steps) {
  if (auto ps = step.as<AnnotationStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<FuseStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<PragmaStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ReorderStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<SplitStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes, transform_steps);
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes, transform_steps);
  } else if (auto ps = step.as<StorageAlignStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes);
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes, schedule);
  } else if (auto ps = step.as<RfactorStepNode>()) {
    return ps->PrintAsPythonAPI(stages, stage_to_axes, schedule);
  } else {
    LOG(FATAL) << "Invalid Step: " << step;
  }
  return "";
}

/********** Steps working on single stage **********/

/********** Annotation **********/
AnnotationStep::AnnotationStep(int stage_id, int iter_id, IteratorAnnotation ann) {
  auto node = make_object<AnnotationStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->annotation = ann;
  data_ = std::move(node);
}

AnnotationStep::AnnotationStep(dmlc::JSONReader* reader) {
  auto node = make_object<AnnotationStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->iter_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  int int_val;
  reader->Read(&int_val);
  node->annotation = IteratorAnnotation(int_val);
  data_ = std::move(node);
}

void AnnotationStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(static_cast<int>(annotation));
}

Iterator AnnotationStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];
  Iterator it = stage->iters[iter_id];

  ICHECK(it->annotation == IteratorAnnotation::kNone);
  Iterator new_it = Iterator(it->name, it->range, it->iter_kind, annotation, &it->orig_iters);
  Stage new_stage = stage;
  new_stage.CopyOnWrite()->iters.Set(iter_id, new_it);
  state->CopyOnWrite()->stages.Set(stage_id, std::move(new_stage));
  return new_it;
}

void AnnotationStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                         StageToAxesMap* stage_to_axes) const {
  te::Stage stage = (*stages)[stage_id];
  const Array<IterVar>& axes = (*stage_to_axes)[stage];

  switch (annotation) {
    case IteratorAnnotation::kUnroll:
      stage.unroll(axes[iter_id]);
      break;
    case IteratorAnnotation::kVectorize:
      stage.vectorize(axes[iter_id]);
      break;
    case IteratorAnnotation::kParallel:
      stage.parallel(axes[iter_id]);
      break;
    case IteratorAnnotation::kVThread:
    case IteratorAnnotation::kBlockX:
    case IteratorAnnotation::kBlockY:
    case IteratorAnnotation::kBlockZ:
    case IteratorAnnotation::kThreadX:
    case IteratorAnnotation::kThreadY:
    case IteratorAnnotation::kThreadZ:
      stage.bind(axes[iter_id],
                 te::thread_axis(Range(), IteratorAnnotationString[static_cast<int>(annotation)]));
      break;
    case IteratorAnnotation::kNone:
      break;
    default:
      LOG(FATAL) << "Invalid Annotation " << static_cast<int>(annotation);
      break;
  }

  stages->Set(stage_id, std::move(stage));
}

String AnnotationStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                            StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& iter = (*stage_to_axes)[stage][iter_id];
  const auto& op_name = CleanName(stage->op->name);

  ss << "s[" << op_name << "].";
  switch (annotation) {
    case IteratorAnnotation::kUnroll:
      ss << "unroll(";
      break;
    case IteratorAnnotation::kVectorize:
      ss << "vectorize(";
      break;
    case IteratorAnnotation::kParallel:
      ss << "parallel(";
      break;
    case IteratorAnnotation::kVThread:
    case IteratorAnnotation::kBlockX:
    case IteratorAnnotation::kBlockY:
    case IteratorAnnotation::kBlockZ:
    case IteratorAnnotation::kThreadX:
    case IteratorAnnotation::kThreadY:
    case IteratorAnnotation::kThreadZ:
      ss << "bind(";
      break;
    case IteratorAnnotation::kNone:
      break;
    default:
      LOG(FATAL) << "Invalid annotation " << static_cast<int>(annotation);
      break;
  }
  ss << CleanName(iter->var->name_hint, op_name);
  switch (annotation) {
    case IteratorAnnotation::kVThread:
    case IteratorAnnotation::kBlockX:
    case IteratorAnnotation::kBlockY:
    case IteratorAnnotation::kBlockZ:
    case IteratorAnnotation::kThreadX:
    case IteratorAnnotation::kThreadY:
    case IteratorAnnotation::kThreadZ:
      ss << ", te.thread_axis(\"" << IteratorAnnotationString[static_cast<int>(annotation)]
         << "\")";
      break;
    default:
      break;
  }
  ss << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Fuse **********/
FuseStep::FuseStep(int stage_id, const Array<Integer>& fused_ids) {
  auto node = make_object<FuseStepNode>();
  node->stage_id = stage_id;
  for (const auto& x : fused_ids) {
    ICHECK(x->IsInstance<IntImmNode>());
  }
  node->fused_ids = fused_ids;
  data_ = std::move(node);
}

FuseStep::FuseStep(dmlc::JSONReader* reader) {
  auto node = make_object<FuseStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->fused_ids);
  data_ = std::move(node);
}

void FuseStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(fused_ids);
}

Iterator FuseStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];
  size_t old_iter_size = static_cast<int>(stage->iters.size());

  String new_name;
  PrimExpr new_extent = 1;
  IteratorKind new_iter_kind = IteratorKind::kSpecial;
  std::vector<Iterator> orig_iters;

  for (size_t i = 0; i < fused_ids.size(); ++i) {
    if (i > 0) {
      ICHECK_EQ(fused_ids[i]->value, fused_ids[i - 1]->value + 1);
    }
    if (i != fused_ids.size() - 1) {
      const auto& iter_to_attached_stage = (*state)->attach_map->iter_to_attached_stages;
      if (iter_to_attached_stage.find(std::make_pair(stage_id, fused_ids[i].IntValue())) !=
          iter_to_attached_stage.end()) {
        LOG(FATAL) << "Invalid Fuse. Trying to fuse iterators that have been attached by some "
                   << "stages. State before fusion:\n"
                   << (*state);
      }
    }

    const Iterator& it = stage->iters[fused_ids[i].IntValue()];
    orig_iters.push_back(it);
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
  Iterator new_it =
      Iterator(new_name, range, new_iter_kind, IteratorAnnotation::kNone, &orig_iters);
  Array<Iterator> new_iters;

  if (fused_ids.empty()) {
    new_iters.push_back(new_it);
  } else {
    new_iters.insert(new_iters.end(), stage->iters.begin(),
                     stage->iters.begin() + fused_ids.front().IntValue());
    new_iters.push_back(new_it);
    new_iters.insert(new_iters.end(), stage->iters.begin() + fused_ids.back().IntValue() + 1,
                     stage->iters.end());
  }

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id,
                     Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));

  if (fused_ids.empty()) {
    return new_it;
  }

  // Two vectors are used to represent the iterator relation before and after fuse
  // The original iterators in AttachMap will be updated with the new iterators
  std::vector<IterKey> from_iters;
  std::vector<IterKey> to_iters;
  const size_t begin_id = fused_ids.front().IntValue(), end_id = fused_ids.back().IntValue();
  for (size_t i = 0; i < old_iter_size; ++i) {
    if (i <= begin_id) {
      continue;
    } else if (i > end_id) {
      // move forward
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, i - end_id + begin_id);
    } else {
      // move to the fused id
      from_iters.emplace_back(stage_id, i);
      to_iters.emplace_back(stage_id, begin_id);
    }
  }
  pstate->attach_map.UpdateIters(from_iters, to_iters);

  return new_it;
}

IterVar FuseStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);

  Array<IterVar> to_fuse;
  for (const auto& i : fused_ids) {
    to_fuse.push_back(axes[i.IntValue()]);
  }
  IterVar fused_axis;
  stage.fuse(to_fuse, &fused_axis);

  Array<IterVar> new_axes;
  if (fused_ids.empty()) {
    new_axes.push_back(fused_axis);
  } else {
    new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + fused_ids.front().IntValue());
    new_axes.push_back(fused_axis);
    new_axes.insert(new_axes.end(), axes.begin() + fused_ids.back().IntValue() + 1, axes.end());
  }

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
  return fused_axis;
}

String FuseStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  const auto& stage = (*stages)[stage_id];
  const auto& op_name = CleanName(stage->op->name);
  std::stringstream to_fuse;

  for (size_t i = 0; i < fused_ids.size(); ++i) {
    to_fuse << CleanName(stage_to_axes->at(stage)[fused_ids[i].IntValue()]->var->name_hint,
                         op_name);
    if (i != fused_ids.size() - 1) {
      to_fuse << ", ";
    }
  }

  std::stringstream ss;
  const auto& fused = ApplyToSchedule(stages, stage_to_axes);

  ss << CleanName(fused->var->name_hint, op_name) << " = s[" << op_name << "].fuse("
     << to_fuse.str() << ")\n";

  return ss.str();
}

/********** Pragma **********/
PragmaStep::PragmaStep(int stage_id, int iter_id, String pragma_type) {
  auto node = make_object<PragmaStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->pragma_type = std::move(pragma_type);
  data_ = std::move(node);
}

PragmaStep::PragmaStep(dmlc::JSONReader* reader) {
  auto node = make_object<PragmaStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->iter_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  std::string string_value;
  reader->Read(&string_value);
  node->pragma_type = std::move(string_value);
  data_ = std::move(node);
}

void PragmaStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArraySeperator();
  writer->WriteString(pragma_type);
}

void PragmaStepNode::ApplyToState(State* state) const {
  if (pragma_type == "debug_skip_region") {
    StateNode* pstate = state->CopyOnWrite();
    pstate->attach_map.DeleteStage(stage_id);
  } else if (StrStartsWith(pragma_type, "auto_unroll_max_step")) {
    StateNode* pstate = state->CopyOnWrite();
    Stage stage = pstate->stages[stage_id];
    size_t pos = 0;
    for (; pos < pragma_type.size(); ++pos) {
      if ((*(pragma_type.c_str() + pos)) == '$') {
        break;
      }
    }
    ICHECK_LT(pos, pragma_type.size()) << "max step value not found.";
    stage.CopyOnWrite()->attrs.auto_unroll_max_step = atoi(pragma_type.c_str() + pos + 1);
    pstate->stages.Set(stage_id, std::move(stage));
  } else {
    LOG(FATAL) << "Unsupported pragma: " << pragma_type;
  }
}

void PragmaStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                     StageToAxesMap* stage_to_axes) const {
  te::Stage stage = (*stages)[stage_id];
  const Array<IterVar>& axes = (*stage_to_axes)[stage];
  if (StrStartsWith(pragma_type, "auto_unroll_max_step")) {
    size_t pos = 0;
    for (; pos < pragma_type.size(); ++pos) {
      if ((*(pragma_type.c_str() + pos)) == '$') {
        break;
      }
    }
    ICHECK_LT(pos, pragma_type.size()) << "max step value not found.";
    int value = atoi(pragma_type.c_str() + pos + 1);
    if (iter_id < static_cast<int>(axes.size())) {
      stage.pragma(axes[iter_id], "auto_unroll_max_step", value);
      stage.pragma(axes[iter_id], "unroll_explicit", true);
    }
  } else {
    ICHECK_LT(iter_id, axes.size());
    stage.pragma(axes[iter_id], pragma_type);
  }
  stages->Set(stage_id, std::move(stage));
}

String PragmaStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                        StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& op_name = CleanName(stage->op->name);

  if (StrStartsWith(pragma_type, "auto_unroll_max_step")) {
    size_t pos = 0;
    for (; pos < pragma_type.size(); ++pos) {
      if ((*(pragma_type.c_str() + pos)) == '$') {
        break;
      }
    }
    ICHECK_LT(pos, pragma_type.size()) << "max step value not found.";
    int value = atoi(pragma_type.c_str() + pos + 1);
    ss << "s[" << op_name << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint, op_name)
       << ", \"auto_unroll_max_step\", " << value << ")\n";
    ss << "s[" << op_name << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint, op_name)
       << ", \"unroll_explicit\", True)\n";
  } else {
    ss << "s[" << op_name << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint, op_name) << ", \""
       << pragma_type << "\")\n";
  }

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Reorder **********/
ReorderStep::ReorderStep(int stage_id, const Array<Integer>& after_ids) {
  auto node = make_object<ReorderStepNode>();
  node->stage_id = stage_id;
  for (const auto& x : after_ids) {
    ICHECK(x->IsInstance<IntImmNode>());
  }
  node->after_ids = after_ids;
  data_ = std::move(node);
}

ReorderStep::ReorderStep(dmlc::JSONReader* reader) {
  auto node = make_object<ReorderStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->after_ids);
  data_ = std::move(node);
}

void ReorderStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(after_ids);
}

void ReorderStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];
  Array<Iterator> iters;
  for (auto x : after_ids) {
    iters.push_back(stage->iters[x.IntValue()]);
  }
  state->CopyOnWrite()->stages.Set(
      stage_id, Stage(stage->op, stage->op_type, iters, stage->compute_at, stage->attrs));
}

void ReorderStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);
  ICHECK_EQ(after_ids.size(), axes.size());

  Array<IterVar> new_axes;
  new_axes.reserve(axes.size());
  for (auto i : after_ids) {
    new_axes.push_back(axes[i.IntValue()]);
  }
  stage.reorder(new_axes);

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
}

String ReorderStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                         StageToAxesMap* stage_to_axes) const {
  const auto& stage = (*stages)[stage_id];
  const auto& op_name = CleanName(stage->op->name);
  std::stringstream ss;

  ss << "s[" << op_name << "].reorder(";
  for (size_t i = 0; i < after_ids.size(); ++i) {
    ss << CleanName((*stage_to_axes)[stage][after_ids[i].IntValue()]->var->name_hint, op_name);
    if (i != after_ids.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Split **********/
// common part for SplitStep, FollowSplitStep, and FollowFusedSplitStep
Array<Iterator> ApplySplitToState(State* state, int stage_id, int iter_id,
                                  const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  const Stage& stage = (*state)->stages[stage_id];
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
      if (!l.defined()) {
        concrete = false;
      }
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

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id,
                     Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));
  pstate->concrete &= concrete;

  // Two vectors are used to represent the iterator relation before and after split
  // The original iterators in AttachMap will be updated with the new iterators
  std::vector<IterKey> from_iters;
  std::vector<IterKey> to_iters;
  for (size_t i = iter_id; i < old_iter_size; ++i) {
    from_iters.emplace_back(stage_id, i);
    to_iters.emplace_back(stage_id, i + lengths.size());
  }
  pstate->attach_map.UpdateIters(from_iters, to_iters);

  return outs;
}

Array<IterVar> ApplySplitToSchedule(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                    int stage_id, int iter_id,
                                    const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);

  Array<IterVar> outs;
  if (inner_to_outer) {
    IterVar outer = axes[iter_id], inner;
    for (int i = static_cast<int>(lengths.size()) - 1; i >= 0; i--) {
      IterVar to_split = outer;
      stage.split(to_split, lengths[i].value(), &outer, &inner);
      outs.push_back(inner);
    }
    outs.push_back(outer);
  } else {
    IterVar outer, inner = axes[iter_id];
    for (size_t i = 0; i < lengths.size(); i++) {
      IterVar to_split = inner;
      stage.split_by_nparts(to_split, lengths[i].value(), &outer, &inner);
      outs.push_back(outer);
    }
    outs.push_back(inner);
  }

  Array<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + iter_id);
  if (inner_to_outer) {
    for (auto x = outs.rbegin(); x != outs.rend(); ++x) {
      new_axes.push_back((*x));
    }
  } else {
    for (const auto& x : outs) {
      new_axes.push_back(x);
    }
  }
  new_axes.insert(new_axes.end(), axes.begin() + iter_id + 1, axes.end());

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
  return outs;
}

String PrintSplitAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes, int stage_id,
                             int iter_id, const Array<Optional<Integer>>& lengths,
                             bool inner_to_outer) {
  const auto& stage = (*stages)[stage_id];
  auto to_split = stage_to_axes->at(stage)[iter_id];
  const auto& func_name = CleanName(stage->op->name);
  const auto& outs =
      ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
  ICHECK_EQ(outs.size(), lengths.size() + 1);

  std::stringstream ss;
  int size = static_cast<int>(lengths.size());
  if (inner_to_outer) {
    for (int i = size - 1; i >= 0; i--) {
      ss << CleanName(outs[size - i]->var->name_hint, func_name) << ", "
         << CleanName(outs[size - i - 1]->var->name_hint, func_name) << " = s[" << func_name
         << "].split(" << CleanName(to_split->var->name_hint, func_name)
         << ", factor=" << lengths[i] << ")\n";
      to_split = outs[size - i];
    }
  } else {
    for (int i = 0; i < size; i++) {
      ss << CleanName(outs[i]->var->name_hint, func_name) << ", "
         << CleanName(outs[i + 1]->var->name_hint, func_name) << " = s[" << func_name << "].split("
         << CleanName(to_split->var->name_hint, func_name) << ", nparts=" << lengths[i] << ")\n";
      to_split = outs[i + 1];
    }
  }

  return ss.str();
}

SplitStep::SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
                     const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  auto node = make_object<SplitStepNode>();
  node->stage_id = stage_id;
  // Extent can be a irreducible expression in some special cases
  if (extent && extent.value()->IsInstance<IntImmNode>()) {
    node->extent = tvm::Downcast<Integer>(extent.value());
  }
  node->iter_id = iter_id;
  node->lengths = lengths;
  node->inner_to_outer = inner_to_outer;
  data_ = std::move(node);
}

SplitStep::SplitStep(dmlc::JSONReader* reader) {
  auto node = make_object<SplitStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->iter_id);
  int int_val;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&int_val);
  if (int_val) {
    node->extent = Integer(int_val);
  }
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->lengths);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->inner_to_outer);
  data_ = std::move(node);
}

void SplitStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(extent ? GetIntImm(extent.value()) : 0);
  writer->WriteArrayItem(lengths);
  writer->WriteArrayItem(static_cast<int>(inner_to_outer));
}

Array<Iterator> SplitStepNode::ApplyToState(State* state) const {
  return ApplySplitToState(state, stage_id, iter_id, lengths, inner_to_outer);
}

Array<IterVar> SplitStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                              StageToAxesMap* stage_to_axes) const {
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
}

String SplitStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                       StageToAxesMap* stage_to_axes) const {
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
}

/********** Follow Split **********/
FollowSplitStep::FollowSplitStep(int stage_id, int iter_id, int src_step_id, int n_split) {
  auto node = make_object<FollowSplitStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->src_step_id = src_step_id;
  node->n_split = n_split;
  data_ = std::move(node);
}

void FollowSplitStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(src_step_id);
  writer->WriteArrayItem(n_split);
}

Array<Optional<Integer>> FollowSplitStepNode::ExtractSplitLengths(
    const Array<Step>& transform_steps) const {
  // Make sure src_step_id is within the range of transform_steps.
  ICHECK_LT(src_step_id, transform_steps.size());
  auto ps = transform_steps[src_step_id].as<SplitStepNode>();
  ICHECK(ps != nullptr);

  // Make sure the size of ps->lengths is not smaller than n_split-1.
  // Note that the number of actual splitting factors of src_step is ps->lengths.size()+1.
  ICHECK_LE(n_split, ps->lengths.size() + 1);
  ICHECK(ps != nullptr);

  Array<Optional<Integer>> lengths;
  lengths.reserve(n_split);
  int j = 0;
  // Get the first (n_split-1) split factors of followed src_step.
  for (; j < n_split - 1; ++j) {
    lengths.push_back(ps->lengths[j]);
  }

  // Get the last split factor of src_step for splitting level if n_split is smaller than
  // ps->lengths.size()+1.
  PrimExpr last_factor = 1;
  for (; j < static_cast<int>(ps->lengths.size()); ++j) {
    if (ps->lengths[j]) {
      last_factor *= ps->lengths[j].value();
    } else {
      last_factor = PrimExpr();
      break;
    }
  }
  if (last_factor.defined()) {
    lengths.push_back(Downcast<Integer>(last_factor));
  } else {
    lengths.push_back(NullOpt);
  }

  return lengths;
}

FollowSplitStep::FollowSplitStep(dmlc::JSONReader* reader) {
  auto node = make_object<FollowSplitStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->iter_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->src_step_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->n_split);
  data_ = std::move(node);
}

Array<Iterator> FollowSplitStepNode::ApplyToState(State* state) const {
  return ApplySplitToState(state, stage_id, iter_id, ExtractSplitLengths((*state)->transform_steps),
                           true);
}

Array<IterVar> FollowSplitStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                                    StageToAxesMap* stage_to_axes,
                                                    const Array<Step>& transform_steps) const {
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              ExtractSplitLengths(transform_steps), true);
}

String FollowSplitStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                             StageToAxesMap* stage_to_axes,
                                             const Array<Step>& transform_steps) const {
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                               ExtractSplitLengths(transform_steps), true);
}

/********** Follow Fused Split **********/
FollowFusedSplitStep::FollowFusedSplitStep(int stage_id, int iter_id,
                                           const Array<Integer>& src_step_ids, int level,
                                           bool factor_or_nparts) {
  auto node = make_object<FollowFusedSplitStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->src_step_ids = src_step_ids;
  node->level = level;
  node->factor_or_nparts = factor_or_nparts;
  data_ = std::move(node);
}

FollowFusedSplitStep::FollowFusedSplitStep(dmlc::JSONReader* reader) {
  auto node = make_object<FollowFusedSplitStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->iter_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->src_step_ids);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->level);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->factor_or_nparts);
  data_ = std::move(node);
}

void FollowFusedSplitStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(src_step_ids);
  writer->WriteArrayItem(level);
  writer->WriteArrayItem(static_cast<int>(factor_or_nparts));
}

Optional<Integer> FollowFusedSplitStepNode::ExtractSplitLength(
    const Array<Step>& transform_steps) const {
  PrimExpr ret(1);

  for (auto src_step_id : src_step_ids) {
    // Make sure the src_step_id is within the range of transform_steps.
    ICHECK_LT(src_step_id.IntValue(), transform_steps.size());
    auto ps = transform_steps[src_step_id.IntValue()].as<SplitStepNode>();
    ICHECK(ps != nullptr);
    // Multiple the splitting factor on corresponding splitting level of src_steps.
    if (ps->lengths[level] && ret.defined()) {
      ret *= ps->lengths[level].value();
    } else {
      return NullOpt;
    }
  }
  return Downcast<Integer>(ret);
}

Array<Iterator> FollowFusedSplitStepNode::ApplyToState(State* state) const {
  return ApplySplitToState(state, stage_id, iter_id,
                           {ExtractSplitLength((*state)->transform_steps)}, factor_or_nparts);
}

Array<IterVar> FollowFusedSplitStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                                         StageToAxesMap* stage_to_axes,
                                                         const Array<Step>& transform_steps) const {
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              {ExtractSplitLength(transform_steps)}, factor_or_nparts);
}

String FollowFusedSplitStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                                  StageToAxesMap* stage_to_axes,
                                                  const Array<Step>& transform_steps) const {
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                               {ExtractSplitLength(transform_steps)}, factor_or_nparts);
}

/********** Storage Align **********/
StorageAlignStep::StorageAlignStep(int stage_id, int iter_id, int factor, int offset) {
  auto node = make_object<StorageAlignStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->factor = factor;
  node->offset = offset;
  data_ = std::move(node);
}

StorageAlignStep::StorageAlignStep(dmlc::JSONReader* reader) {
  auto node = make_object<StorageAlignStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->iter_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->factor);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->offset);
  data_ = std::move(node);
}

void StorageAlignStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(factor);
  writer->WriteArrayItem(offset);
}

void StorageAlignStepNode::ApplyToState(State* state) const {
  StateNode* pstate = state->CopyOnWrite();
  Stage stage = pstate->stages[stage_id];
  stage.CopyOnWrite()->attrs.storage_offset = offset;
  pstate->stages.Set(stage_id, std::move(stage));
}

void StorageAlignStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                           StageToAxesMap* stage_to_axes) const {
  te::Stage stage = (*stages)[stage_id];
  const Array<IterVar>& axes = (*stage_to_axes)[stage];
  stage.storage_align(axes[iter_id], factor, offset);
  stages->Set(stage_id, std::move(stage));
}

String StorageAlignStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                              StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& op_name = CleanName(stage->op->name);
  ss << "s[" << op_name << "].storage_align("
     << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint, op_name) << ", " << factor
     << ", " << offset << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Steps working on multiple stages **********/

/********** Compute At **********/
ComputeAtStep::ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id) {
  auto node = make_object<ComputeAtStepNode>();
  node->stage_id = stage_id;
  node->target_stage_id = target_stage_id;
  node->target_iter_id = target_iter_id;
  data_ = std::move(node);
}

ComputeAtStep::ComputeAtStep(dmlc::JSONReader* reader) {
  auto node = make_object<ComputeAtStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->target_stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->target_iter_id);
  data_ = std::move(node);
}

void ComputeAtStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(target_stage_id);
  writer->WriteArrayItem(target_iter_id);
}
void ComputeAtStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];

  // Remove the bound information of each iterator since they may not be accurate after
  // compute at
  Array<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    new_iters.push_back(
        Iterator(it->name, Range(), it->iter_kind, it->annotation, &it->orig_iters));
  }

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id, Stage(stage->op, stage->op_type, std::move(new_iters),
                                     ComputeAtKind::kIter, stage->attrs));
  // Update attach map
  pstate->attach_map.SetComputeAtIter(stage_id, target_stage_id, target_iter_id);
}

void ComputeAtStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                        StageToAxesMap* stage_to_axes) const {
  te::Stage stage = (*stages)[stage_id];
  const auto& target_stage = (*stages)[target_stage_id];
  const auto& target_axis = (*stage_to_axes)[target_stage][target_iter_id];
  stage.compute_at(target_stage, target_axis);

  stages->Set(stage_id, std::move(stage));
}

String ComputeAtStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                           StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& target_stage = (*stages)[target_stage_id];
  const auto& op_name = CleanName(stage->op->name);
  const auto& target_op_name = CleanName(target_stage->op->name);
  ss << "s[" << op_name << "].compute_at(s[" << target_op_name << "], "
     << CleanName((*stage_to_axes)[target_stage][target_iter_id]->var->name_hint, target_op_name)
     << ")\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute Inline **********/
ComputeInlineStep::ComputeInlineStep(int stage_id) {
  auto node = make_object<ComputeInlineStepNode>();
  node->stage_id = stage_id;
  data_ = std::move(node);
}

ComputeInlineStep::ComputeInlineStep(dmlc::JSONReader* reader) {
  auto node = make_object<ComputeInlineStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  data_ = std::move(node);
}

void ComputeInlineStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
}

void ComputeInlineStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];

  // Check the validity of compute_inline
  for (size_t i = 0; i < stage->iters.size(); ++i) {
    ICHECK_EQ((*state)->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, i)), 0)
        << "Invalid compute_inline: There are some other stages that are attached to the "
        << "target stage";
  }

  StateNode* pstate = state->CopyOnWrite();
  auto new_stage = pstate->stages[stage_id];
  new_stage.CopyOnWrite()->compute_at = ComputeAtKind::kInlined;
  pstate->stages.Set(stage_id, std::move(new_stage));
  // Update attach map
  pstate->attach_map.DeleteStage(stage_id);
}

void ComputeInlineStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                            StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  stage.compute_inline();
  stages->Set(stage_id, std::move(stage));
}

String ComputeInlineStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                               StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  ss << "s[" << CleanName(stage->op->name) << "].compute_inline()\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute Root **********/
ComputeRootStep::ComputeRootStep(int stage_id) {
  auto node = make_object<ComputeRootStepNode>();
  node->stage_id = stage_id;
  data_ = std::move(node);
}

ComputeRootStep::ComputeRootStep(dmlc::JSONReader* reader) {
  auto node = make_object<ComputeRootStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  data_ = std::move(node);
}

void ComputeRootStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
}

void ComputeRootStepNode::ApplyToState(State* state) const {
  const Stage& stage = (*state)->stages[stage_id];

  // Remove the bound information of each iterator since they may not be accurate after
  // compute root
  Array<Iterator> new_iters;
  for (const Iterator& it : stage->iters) {
    new_iters.push_back(
        Iterator(it->name, Range(), it->iter_kind, it->annotation, &it->orig_iters));
  }

  StateNode* pstate = state->CopyOnWrite();
  pstate->stages.Set(stage_id, Stage(stage->op, stage->op_type, std::move(new_iters),
                                     ComputeAtKind::kRoot, stage->attrs));
  // Update attach map
  pstate->attach_map.DeleteStage(stage_id);
}

void ComputeRootStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                          StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  stage.compute_root();
  stages->Set(stage_id, std::move(stage));
}

String ComputeRootStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                             StageToAxesMap* stage_to_axes) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  ss << "s[" << CleanName(stage->op->name) << "].compute_root()\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Steps adding new stages **********/

/*!
 * \brief Common part for steps that add new stages(e.g. CacheReadStep, CacheWriteStep,
 * RfactorStep). This will return all steps that can change the number of stages in a ComputeDAG,
 * and stop by the current step.
 */
Array<Step> GetFormerStageModifiableSteps(Step current_step, const Array<Step>& transform_steps) {
  Array<Step> ret_steps;
  for (size_t i = 0; i < transform_steps.size(); ++i) {
    const Step& step = transform_steps[i];
    if (step->IsInstance<CacheWriteStepNode>() || step->IsInstance<CacheReadStepNode>()) {
      ret_steps.push_back(step);
    } else if (step->IsInstance<RfactorStepNode>()) {
      // add FuseStepNode required by rfactor
      if (i >= 2 && transform_steps[i - 2]->IsInstance<FuseStepNode>()) {
        const Step& fuse_step = transform_steps[i - 2];
        if (fuse_step->stage_id == step->stage_id) {
          ret_steps.push_back(fuse_step);
        }
      }
      // add SplitStepNode required by rfactor
      ICHECK_GE(i, 1);
      ICHECK(transform_steps[i - 1]->IsInstance<SplitStepNode>());
      const Step& split_step = transform_steps[i - 1];
      ICHECK_EQ(split_step->stage_id, step->stage_id);
      ret_steps.push_back(split_step);
      // add RfactorStepNode
      ret_steps.push_back(step);
    }
    // A state may have multiple stage modifiable steps, stop by the current step to avoid
    // replaying excess steps
    if (step.same_as(current_step)) {
      break;
    }
  }
  return ret_steps;
}

/********** Cache Read **********/
CacheReadStep::CacheReadStep(int stage_id, String scope_name,
                             const Array<Integer>& reader_stage_ids) {
  auto node = make_object<CacheReadStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  node->reader_stage_ids = reader_stage_ids;
  data_ = std::move(node);
}

CacheReadStep::CacheReadStep(dmlc::JSONReader* reader) {
  auto node = make_object<CacheReadStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  std::string string_value;
  reader->Read(&string_value);
  node->scope_name = std::move(string_value);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->reader_stage_ids);
  data_ = std::move(node);
}

void CacheReadStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArraySeperator();
  writer->WriteString(scope_name);
  writer->WriteArrayItem(reader_stage_ids);
}

int CacheReadStepNode::ApplyToState(State* state, const ComputeDAG& dag) const {
  StateNode* pstate = state->CopyOnWrite();
  const ComputeDAG& current_compute_dag = dag.ReplayAndGetDAG(
      GetFormerStageModifiableSteps(GetRef<Step>(this), (*state)->transform_steps));

  // target_stage -> target_stage + target_store
  // Update the op of the target stage, insert a new cache read stage behind, update the op of
  // later stages, then update the stage_id mapping in AttachMap
  int added_stage_id = stage_id + 1;
  Stage tmp_stage = pstate->stages[stage_id];
  tmp_stage.CopyOnWrite()->op = current_compute_dag->ops[stage_id];
  pstate->stages.Set(stage_id, std::move(tmp_stage));
  pstate->stages.insert(pstate->stages.begin() + added_stage_id,
                        Stage(current_compute_dag->ops[added_stage_id]));
  for (size_t i = added_stage_id + 1; i < pstate->stages.size(); ++i) {
    tmp_stage = pstate->stages[i];
    tmp_stage.CopyOnWrite()->op = current_compute_dag->ops[i];
    pstate->stages.Set(i, std::move(tmp_stage));
  }
  pstate->attach_map = pstate->attach_map.ApplyStageIdOffset(added_stage_id);
  pstate->current_compute_dag = std::move(current_compute_dag);

  return added_stage_id;
}

te::Tensor CacheReadStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                              StageToAxesMap* stage_to_axes,
                                              te::Schedule* schedule) const {
  const te::Stage& stage = (*stages)[stage_id];
  Array<te::Operation> readers;
  for (const auto& i : reader_stage_ids) {
    readers.push_back((*stages)[i.IntValue()]->origin_op);
  }
  auto out = schedule->cache_read(stage->origin_op.output(0), scope_name, readers);

  const auto& new_stage = (*schedule)[out->op];
  UpdateStageToAxesMap(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id + 1, new_stage);

  return out;
}

String CacheReadStepNode::PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                           te::Schedule* schedule) const {
  std::stringstream ss;
  // Since the original stage will be changed after schedule apply, keep a copy here
  // These information will be used to print Python API string later
  auto stage = (*stages)[stage_id];
  Array<te::Stage> reader_stages;
  for (size_t i = 0; i < reader_stage_ids.size(); ++i) {
    reader_stages.push_back((*stages)[reader_stage_ids[i].IntValue()]);
  }
  auto out = ApplyToSchedule(stages, stage_to_axes, schedule);

  const auto& op_name = CleanName(out->op->name);
  ss << op_name << " = "
     << "s.cache_read(" << CleanName(stage->op->name) << ", \"" << scope_name << "\", ["
     << CleanName(reader_stages[0]->op->name);
  for (size_t i = 1; i < reader_stage_ids.size(); ++i) {
    ss << ", " << CleanName(reader_stages[i]->op->name);
  }
  ss << "])\n";

  // Print the iterators of the new added stage
  const auto& iters = out->op->root_iter_vars();
  for (size_t i = 0; i < iters.size(); ++i) {
    ss << CleanName(iters[i]->var->name_hint, op_name);
    if (i != iters.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = "
     << "tuple(" << op_name << ".op.axis)\n";

  return ss.str();
}

/********** Cache Write **********/
CacheWriteStep::CacheWriteStep(int stage_id, String scope_name) {
  auto node = make_object<CacheWriteStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  data_ = std::move(node);
}

CacheWriteStep::CacheWriteStep(dmlc::JSONReader* reader) {
  auto node = make_object<CacheWriteStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  std::string string_value;
  reader->Read(&string_value);
  node->scope_name = std::move(string_value);
  data_ = std::move(node);
}

void CacheWriteStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArraySeperator();
  writer->WriteString(scope_name);
}

int CacheWriteStepNode::ApplyToState(State* state, const ComputeDAG& dag) const {
  StateNode* pstate = state->CopyOnWrite();
  int last_dag_op_size = pstate->current_compute_dag
                             ? pstate->current_compute_dag.value().as<ComputeDAGNode>()->ops.size()
                             : dag->ops.size();
  const ComputeDAG& current_compute_dag = dag.ReplayAndGetDAG(
      GetFormerStageModifiableSteps(GetRef<Step>(this), (*state)->transform_steps));
  int added_ops = current_compute_dag->ops.size() - last_dag_op_size;
  // TODO(jcf94): Update this check to equal after fixing the cache write bug in TVM
  ICHECK_GE(added_ops, 1);

  // target_stage -> cache_write_stage + target_stage
  // Assume no step has been applied to the target stage before cache write.
  // Insert a new cache write stage ahead, update the op of the target stage and later stages, then
  // update the stage_id mapping in AttachMap
  pstate->stages.insert(pstate->stages.begin() + stage_id,
                        Stage(current_compute_dag->ops[stage_id]));
  pstate->stages.Set(stage_id + 1, Stage(current_compute_dag->ops[stage_id + 1]));
  int next_stage_id = stage_id + 2;
  // TODO(jc94): Fix the cache write bug in TVM and remove added_op == 2 support.
  // TVM's cache_write has a bug with multi outputs. See
  // `tests/python/auto_scheduler/test_auto_scheduler_loop_state.py::test_cache_read_write` test
  // for more details
  if (added_ops == 2) {
    pstate->stages.insert(pstate->stages.begin() + next_stage_id,
                          Stage(current_compute_dag->ops[next_stage_id]));
    next_stage_id++;
  } else if (added_ops > 2) {
    LOG(ERROR) << "Unexpected behavior of CacheWrite.";
  }
  for (size_t i = next_stage_id; i < current_compute_dag->ops.size(); ++i) {
    Stage tmp_stage = pstate->stages[i];
    tmp_stage.CopyOnWrite()->op = current_compute_dag->ops[i];
    pstate->stages.Set(i, std::move(tmp_stage));
  }
  pstate->attach_map = pstate->attach_map.ApplyStageIdOffset(stage_id, added_ops);
  pstate->current_compute_dag = std::move(current_compute_dag);

  return stage_id;
}

Array<te::Tensor> CacheWriteStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                                      StageToAxesMap* stage_to_axes,
                                                      te::Schedule* schedule) const {
  const te::Stage& stage = (*stages)[stage_id];
  Array<te::Tensor> tensor_array;
  // If the target stage has multi outputs, TVM requires to cache_write
  // all of them or schedule.cache_write will raise an error
  for (auto i = 0; i < stage->op->num_outputs(); ++i) {
    tensor_array.push_back(stage->origin_op.output(i));
  }
  auto outs = schedule->cache_write(tensor_array, scope_name);

  UpdateStageToAxesMap(stage, stage_to_axes);
  // Even if there is multi outputs, TVM schedule only generate one
  // new stage
  const auto& new_stage = (*schedule)[outs[0]->op];
  UpdateStageToAxesMap(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id, new_stage);

  return outs;
}

String CacheWriteStepNode::PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                            te::Schedule* schedule) const {
  std::stringstream ss;
  // Since the original stage will be changed after schedule apply, keep a copy here
  // These information will be used to print Python API string later
  te::Stage stage = (*stages)[stage_id];
  auto outs = ApplyToSchedule(stages, stage_to_axes, schedule);

  for (size_t i = 0; i < outs.size(); ++i) {
    ss << CleanName(outs[i]->op->name) << ", ";
  }
  ss << "= "
     << "s.cache_write([" << CleanName(stage->op.output(0)->op->name);
  for (auto i = 1; i < stage->op->num_outputs(); ++i) {
    ss << ", " << CleanName(stage->op.output(i)->op->name);
  }
  ss << "], \"" << scope_name << "\")\n";

  // Print the iterators of the new added stage
  for (const auto& out : outs) {
    const auto& iters = out->op->root_iter_vars();
    const auto& op_name = CleanName(out->op->name);
    for (size_t i = 0; i < iters.size(); ++i) {
      ss << CleanName(iters[i]->var->name_hint, op_name);
      if (i != iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << " = "
       << "tuple(" << op_name << ".op.axis)"
       << " + "
       << "tuple(" << op_name << ".op.reduce_axis)\n";
  }

  return ss.str();
}

/********** Rfactor **********/
RfactorStep::RfactorStep(int stage_id, int iter_id, int factor_iter_id) {
  auto node = make_object<RfactorStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->factor_iter_id = factor_iter_id;
  data_ = std::move(node);
}

RfactorStep::RfactorStep(dmlc::JSONReader* reader) {
  auto node = make_object<RfactorStepNode>();
  bool s;
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->stage_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->iter_id);
  s = reader->NextArrayItem();
  ICHECK(s);
  reader->Read(&node->factor_iter_id);
  data_ = std::move(node);
}

void RfactorStepNode::WriteToRecord(dmlc::JSONWriter* writer) const {
  writer->WriteArraySeperator();
  writer->WriteString(record_prefix_str);
  writer->WriteArrayItem(stage_id);
  writer->WriteArrayItem(iter_id);
  writer->WriteArrayItem(factor_iter_id);
}

int RfactorStepNode::ApplyToState(State* state, const ComputeDAG& dag) const {
  StateNode* pstate = state->CopyOnWrite();
  const auto& compute_at_type = pstate->stages[stage_id]->compute_at;
  const ComputeDAG& current_compute_dag = dag.ReplayAndGetDAG(
      GetFormerStageModifiableSteps(GetRef<Step>(this), (*state)->transform_steps));

  // target_stage -> rfactor_compute + target_stage
  // Insert a new compute stage, update the target stage and later stage, then update the stage_id
  // mapping in AttachMap
  pstate->stages.insert(pstate->stages.begin() + stage_id,
                        Stage(current_compute_dag->ops[stage_id]));
  // Maintain the compute_at type of the target stage
  Stage target_stage = Stage(current_compute_dag->ops[stage_id + 1]);
  target_stage.CopyOnWrite()->compute_at = compute_at_type;
  pstate->stages.Set(stage_id + 1, std::move(target_stage));
  for (size_t i = stage_id + 2; i < pstate->stages.size(); ++i) {
    Stage stage = pstate->stages[i];
    stage.CopyOnWrite()->op = current_compute_dag->ops[i];
    pstate->stages.Set(i, std::move(stage));
  }
  pstate->attach_map = pstate->attach_map.ApplyStageIdOffset(stage_id);
  pstate->current_compute_dag = std::move(current_compute_dag);

  return stage_id;
}

Array<te::Tensor> RfactorStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                                   StageToAxesMap* stage_to_axes,
                                                   te::Schedule* schedule) const {
  const auto& stage = (*stages)[stage_id];
  const Array<IterVar>& axes = (*stage_to_axes)[stage];

  const te::Tensor& tensor = stage->origin_op.output(0);
  const IterVar& axis = axes[iter_id];
  auto outs = schedule->rfactor(tensor, axis, factor_iter_id);

  UpdateStageToAxesMap(stage, stage_to_axes);
  const auto& new_stage = (*schedule)[outs[0]->op];
  UpdateStageToAxesMap(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id, new_stage);

  return outs;
}

String RfactorStepNode::PrintAsPythonAPI(Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
                                         te::Schedule* schedule) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  const auto& tensor_name = CleanName(stage->origin_op.output(0)->op->name);
  const auto& axis_name = CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint);

  const auto& outs = ApplyToSchedule(stages, stage_to_axes, schedule);

  for (size_t i = 0; i < outs.size(); ++i) {
    ss << CleanName(outs[i]->op->name);
    if (i != outs.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = "
     << "s.rfactor(" << tensor_name << ", " << axis_name << ", " << factor_iter_id << ")\n";

  for (const auto& out : outs) {
    const auto& iters = out->op->root_iter_vars();
    const auto& op_name = CleanName(out->op->name);
    for (size_t i = 0; i < iters.size(); ++i) {
      ss << CleanName(iters[i]->var->name_hint, op_name);
      if (i != iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << " = "
       << "tuple(" << op_name << ".op.axis)"
       << " + "
       << "tuple(" << op_name << ".op.reduce_axis)\n";
  }

  const auto& output = (*stages)[stage_id + 1]->op.output(0);
  const auto& iters = output->op->root_iter_vars();
  const auto& op_name = CleanName(output->op->name);
  for (size_t i = 0; i < iters.size(); ++i) {
    ss << CleanName(iters[i]->var->name_hint, op_name);
    if (i != iters.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = "
     << "tuple(s[" << op_name << "].op.axis)"
     << " + "
     << "tuple(s[" << op_name << "].op.reduce_axis)\n";

  return ss.str();
}

}  // namespace auto_scheduler
}  // namespace tvm
