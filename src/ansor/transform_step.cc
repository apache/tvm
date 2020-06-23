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
 * \file ansor/transform_step.cc
 * \brief  Transformation steps. For each schedule primitive, there is a corresponding transform step.
 *
 * See the note in transform_step.h on how to add a new step
 */

#include "transform_step.h"
#include <tvm/te/operation.h>
#include <tvm/runtime/registry.h>
#include <utility>
#include "utils.h"

namespace tvm {
namespace ansor {

/********** Reorder **********/
ReorderStep::ReorderStep(int stage_id, const std::vector<int>& after_ids) {
  auto node = make_object<ReorderStepNode>();
  node->stage_id = stage_id;
  node->after_ids = after_ids;
  data_ = std::move(node);
}

void ReorderStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                      StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];
  CHECK_EQ(after_ids.size(), axes.size());

  std::vector<IterVar> new_axes;
  new_axes.reserve(axes.size());
  for (auto i : after_ids) {
    new_axes.push_back(axes[i]);
  }
  stage.reorder(new_axes);
  (*stage_to_axes)[stage] = std::move(new_axes);
}

std::string ReorderStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                              StageToAxesMap *stage_to_axes,
                                              te::Schedule *schedule,
                                              const std::vector<Step>& transform_steps) const {
  const te::Stage& stage = (*stages)[stage_id];
  std::stringstream ss;

  ss << "s[" << CleanName(stage->op->name) << "].reorder(";
  for (size_t i = 0; i < after_ids.size(); ++i) {
    ss << CleanName((*stage_to_axes)[stage][after_ids[i]]->var->name_hint);
    if (i != after_ids.size() - 1) {
      ss << ", ";
    }
  }
  ss << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Split **********/
std::vector<IterVar> ApplySplitToSchedule(std::vector<te::Stage> *stages,
                                          StageToAxesMap *stage_to_axes,
                                          int stage_id,
                                          int iter_id,
                                          const std::vector<PrimExpr>& lengths,
                                          bool inner_to_outer) {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  std::vector<IterVar> outs;
  if (inner_to_outer) {
    IterVar outer = axes[iter_id], inner;
    for (int i = static_cast<int>(lengths.size()) - 1; i >= 0; i--) {
      IterVar to_split = outer;
      stage.split(to_split, lengths[i], &outer, &inner);
      outs.push_back(inner);
    }
    outs.push_back(outer);
  } else {
    IterVar outer, inner = axes[iter_id];
    for (size_t i = 0; i < lengths.size(); i++) {
      IterVar to_split = inner;
      stage.split_by_nparts(to_split, lengths[i], &outer, &inner);
      outs.push_back(outer);
    }
    outs.push_back(inner);
  }

  std::vector<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + iter_id);
  if (inner_to_outer) {
    new_axes.insert(new_axes.end(), outs.rbegin(), outs.rend());
  } else {
    new_axes.insert(new_axes.end(), outs.begin(), outs.end());
  }
  new_axes.insert(new_axes.end(), axes.begin() + iter_id + 1, axes.end());
  (*stage_to_axes)[stage] = std::move(new_axes);

  return outs;
}

std::string PrintSplitAsPythonAPI(std::vector<te::Stage> *stages,
                                  StageToAxesMap *stage_to_axes,
                                  int stage_id,
                                  int iter_id,
                                  const std::vector<PrimExpr>& lengths,
                                  bool inner_to_outer) {
  te::Stage& stage = (*stages)[stage_id];
  auto to_split = (*stage_to_axes)[stage][iter_id];
  const auto& func_name = CleanName(stage->op->name);
  const auto& outs = ApplySplitToSchedule(stages, stage_to_axes, stage_id,
                                          iter_id, lengths, inner_to_outer);

  std::stringstream ss;
  int size = static_cast<int>(lengths.size());
  if (inner_to_outer) {
    for (int i = size - 1; i >= 0; i--) {
      ss << CleanName(outs[size - i]->var->name_hint) << ", "
        << CleanName(outs[size - i - 1]->var->name_hint)
        << " = s[" << func_name << "].split("
        << CleanName(to_split->var->name_hint)
        << ", factor=" << lengths[i] << ")\n";
      to_split = outs[size - i];
    }
  } else {
    for (int i = 0; i < size; i++) {
      ss << CleanName(outs[i]->var->name_hint) << ", "
        << CleanName(outs[i + 1]->var->name_hint)
        << " = s[" << func_name << "].split("
        << CleanName(to_split->var->name_hint)
        << ", nparts=" << lengths[i] << ")\n";
      to_split = outs[i + 1];
    }
  }

  return ss.str();
}

SplitStep::SplitStep(int stage_id, int iter_id, PrimExpr extent,
                     const std::vector<PrimExpr>& lengths,
                     bool inner_to_outer) {
  auto node = make_object<SplitStepNode>();
  node->stage_id = stage_id;
  // Extent can be a unreducible expression in some special cases
  if (extent->IsInstance<IntImmNode>()) {
    node->extent = std::move(extent);
  }
  node->iter_id = iter_id;
  node->lengths = lengths;
  node->inner_to_outer = inner_to_outer;
  data_ = std::move(node);
}

std::vector<IterVar> SplitStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes) const {
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              lengths, inner_to_outer);
}

std::string SplitStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                               lengths, inner_to_outer);
}

/********** Follow Split **********/
FollowSplitStep::FollowSplitStep(int stage_id, int iter_id,
                                 int src_step_id, int n_split) {
  auto node = make_object<FollowSplitStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->src_step_id = src_step_id;
  node->n_split = n_split;
  data_ = std::move(node);
}

void FollowSplitStepNode::ExtractSplitLengths(
    const std::vector<Step>& transform_steps,
    std::vector<PrimExpr>* lengths) const {
  CHECK_LT(src_step_id, transform_steps.size());
  auto ps = transform_steps[src_step_id].as<SplitStepNode>();
  CHECK(ps != nullptr);

  // get lengths from src step
  lengths->reserve(n_split);
  int j = 0;
  for (; j < n_split - 1; ++j) {
    lengths->push_back(ps->lengths[j]);
  }
  PrimExpr last_factor = 1;
  for (; j < static_cast<int>(ps->lengths.size()); ++j) {
    if (ps->lengths[j].defined()) {
      last_factor *= ps->lengths[j];
    } else {
      last_factor = PrimExpr();
      break;
    }
  }
  lengths->push_back(std::move(last_factor));
}

std::vector<IterVar> FollowSplitStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    const std::vector<Step>& transform_steps) const {
  std::vector<PrimExpr> lengths;
  ExtractSplitLengths(transform_steps, &lengths);
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              lengths, true);
}

std::string FollowSplitStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  std::vector<PrimExpr> lengths;
  ExtractSplitLengths(transform_steps, &lengths);
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                               lengths, true);
}

/********** Follow Fused Split **********/
FollowFusedSplitStep::FollowFusedSplitStep(int stage_id, int iter_id,
    const std::vector<int>& src_step_ids, int level, bool factor_or_nparts) {
  auto node = make_object<FollowFusedSplitStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->src_step_ids = src_step_ids;;
  node->level = level;
  node->factor_or_nparts = factor_or_nparts;
  data_ = std::move(node);
}

PrimExpr FollowFusedSplitStepNode::ExtractSplitLength(
    const std::vector<Step>& transform_steps) const {
  PrimExpr ret(1);

  for (int src_step_id : src_step_ids) {
    CHECK_LT(src_step_id, transform_steps.size());
    auto ps = transform_steps[src_step_id].as<SplitStepNode>();
    CHECK(ps != nullptr);
    if (ps->lengths[level].defined() && ret.defined()) {
      ret *= ps->lengths[level];
    } else {
      return PrimExpr();
    }
  }

  return ret;
}

std::vector<IterVar> FollowFusedSplitStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    const std::vector<Step>& transform_steps) const {
  const PrimExpr& length = ExtractSplitLength(transform_steps);
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id,
                              {length}, factor_or_nparts);
}

std::string FollowFusedSplitStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  const PrimExpr& length = ExtractSplitLength(transform_steps);
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id,
                               {length}, factor_or_nparts);
}


/********** Fuse **********/
FuseStep::FuseStep(int stage_id, const std::vector<int>& fused_ids) {
  auto node = make_object<FuseStepNode>();
  node->stage_id = stage_id;
  node->fused_ids = fused_ids;
  data_ = std::move(node);
}

IterVar FuseStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                      StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  Array<IterVar> to_fuse;
  for (auto i : fused_ids) {
    to_fuse.push_back(axes[i]);
  }
  IterVar fused_axis;
  stage.fuse(to_fuse, &fused_axis);
  std::vector<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + fused_ids[0]);
  new_axes.push_back(fused_axis);
  new_axes.insert(new_axes.end(), axes.begin() + fused_ids.back() + 1,
                  axes.end());
  (*stage_to_axes)[stage] = std::move(new_axes);

  return fused_axis;
}

std::string FuseStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                           StageToAxesMap *stage_to_axes,
                                           te::Schedule *schedule,
                                           const std::vector<Step>& transform_steps) const {
  const auto& stage = (*stages)[stage_id];
  std::stringstream to_fuse;

  for (size_t i = 0; i < fused_ids.size(); ++i) {
    to_fuse << CleanName((*stage_to_axes)[stage][fused_ids[i]]->var->name_hint);
    if (i != fused_ids.size() - 1) {
      to_fuse << ", ";
    }
  }

  std::stringstream ss;
  const auto& fused = ApplyToSchedule(stages, stage_to_axes);

  ss << CleanName(fused->var->name_hint) << " = s["
     << CleanName(stage->op->name) << "].fuse("
     << to_fuse.str() << ")\n";

  return ss.str();
}

/********** Annotation **********/
AnnotationStep::AnnotationStep(int stage_id, int iter_id,
                               IteratorAnnotation ann) {
  auto node = make_object<AnnotationStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->annotation = ann;
  data_ = std::move(node);
}

void AnnotationStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                         StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  switch (annotation) {
    case kUnroll:    stage.unroll(axes[iter_id]); break;
    case kVectorize: stage.vectorize(axes[iter_id]); break;
    case kParallel:  stage.parallel(axes[iter_id]); break;
    case kVThread:   stage.bind(axes[iter_id], te::thread_axis(Range(), "vthread")); break;
    case kBlockX:    stage.bind(axes[iter_id], te::thread_axis(Range(), "blockIdx.x")); break;
    case kBlockY:    stage.bind(axes[iter_id], te::thread_axis(Range(), "blockIdx.y")); break;
    case kThreadX:
      if (axes[iter_id]->iter_type == kCommReduce) {
        const auto &thread_x = te::thread_axis(Range(), "threadIdx.x");
        stage.bind(axes[iter_id], thread_x);
        stage.set_store_predicate(thread_x->var == 0);
      } else {
        stage.bind(axes[iter_id], te::thread_axis(Range(), "threadIdx.x"));
      }
      break;
    case kThreadY:   stage.bind(axes[iter_id], te::thread_axis(Range(), "threadIdx.y")); break;
    case kNone: break;
    default: LOG(FATAL) << "Invalid Annotation " << annotation; break;
  }
}

std::string AnnotationStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                 StageToAxesMap *stage_to_axes,
                                                 te::Schedule *schedule,
                                                 const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& iter = (*stage_to_axes)[stage][iter_id];

  bool bind_reduce_iter = iter->iter_type == kCommReduce && annotation == kThreadX;
  if (bind_reduce_iter) {
    ss << "thread_x = tvm.thread_axis(\"threadIdx.x\")\n";
  }

  ss << "s[" << CleanName(stage->op->name) << "].";
  switch (annotation) {
    case kUnroll:    ss << "unroll("; break;
    case kVectorize: ss << "vectorize("; break;
    case kParallel:  ss << "parallel("; break;
    case kVThread:
    case kBlockX:
    case kBlockY:
    case kThreadX:
    case kThreadY:   ss << "bind("; break;
    case kNone:      break;
    default:
      LOG(FATAL) << "Invalid annotation " << annotation; break;
  }
  ss << CleanName(iter->var->name_hint);
  switch (annotation) {
    case kVThread:   ss << ", tvm.thread_axis(\"vthread\")"; break;
    case kBlockX:    ss << ", tvm.thread_axis(\"blockIdx.x\")"; break;
    case kBlockY:    ss << ", tvm.thread_axis(\"blockIdy.y\")"; break;
    case kThreadX:
      if (bind_reduce_iter) {
        ss << ", thread_x";
      } else {
        ss << ", tvm.thread_axis(\"threadIdx.x\")";
      }
      break;
    case kThreadY:   ss << ", tvm.thread_axis(\"threadIdx.y\")"; break;
    default:         break;
  }
  ss << ")\n";

  if (bind_reduce_iter) {
    ss << "s[" << CleanName(stage->op->name) << "]"
       << ".set_store_predicate(thread_x.var.equal(0))\n";
  }

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute At **********/
ComputeAtStep::ComputeAtStep(int stage_id, int target_stage_id, int target_iter_id) {
  auto node = make_object<ComputeAtStepNode>();
  node->stage_id = stage_id;
  node->target_stage_id = target_stage_id;
  node->target_iter_id = target_iter_id;
  data_ = std::move(node);
}

void ComputeAtStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                        StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const IterVar& target_axis =
      (*stage_to_axes)[(*stages)[target_stage_id]][target_iter_id];
  stage.compute_at((*stages)[target_stage_id], target_axis);
}

std::string ComputeAtStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                StageToAxesMap *stage_to_axes,
                                                te::Schedule *schedule,
                                                const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  const auto& target_stage = (*stages)[target_stage_id];

  ss << "s[" << CleanName(stage->op->name) << "].compute_at(s["
      << CleanName(target_stage->op->name) << "], "
      << CleanName((*stage_to_axes)[target_stage][target_iter_id]->var->name_hint);

  ss << ")\n";
  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Compute Root **********/
ComputeRootStep::ComputeRootStep(int stage_id) {
  auto node = make_object<ComputeRootStepNode>();
  node->stage_id = stage_id;
  data_ = std::move(node);
}

void ComputeRootStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                          StageToAxesMap *stage_to_axes) const {
  (*stages)[stage_id].compute_root();
}

std::string ComputeRootStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                  StageToAxesMap *stage_to_axes,
                                                  te::Schedule *schedule,
                                                  const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  ss << "s[" << CleanName(stage->op->name) << "].compute_root()\n";
  ApplyToSchedule(stages, stage_to_axes);

  return ss.str();
}

/********** Compute Inline **********/
ComputeInlineStep::ComputeInlineStep(int stage_id) {
  auto node = make_object<ComputeInlineStepNode>();
  node->stage_id = stage_id;
  data_ = std::move(node);
}

void ComputeInlineStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                            StageToAxesMap *stage_to_axes) const {
  (*stages)[stage_id].compute_inline();
}

std::string ComputeInlineStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes,
    te::Schedule *schedule,
    const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  ss << "s[" << CleanName(stage->op->name) << "].compute_inline()\n";
  ApplyToSchedule(stages, stage_to_axes);

  return ss.str();
}

/********** Cache Read **********/
CacheReadStep::CacheReadStep(int stage_id, std::string scope_name,
                             const std::vector<int>& reader_stage_ids) {
  auto node = make_object<CacheReadStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  node->reader_stage_ids = reader_stage_ids;
  data_ = std::move(node);
}

te::Tensor CacheReadStepNode::ApplyToSchedule(std::vector<te::Stage>* stages,
    StageToAxesMap *stage_to_axes, te::Schedule *schedule) const {
  te::Stage& stage = (*stages)[stage_id];

  Array<te::Operation> readers;
  for (const auto& i : reader_stage_ids) {
    readers.push_back((*stages)[i]->origin_op);
  }
  auto out = schedule->cache_read(stage->origin_op.output(0), scope_name, readers);

  const auto& new_stage = (*schedule)[out->op];
  UpdateStageAxis(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id + 1, new_stage);

  return out;
}

std::string CacheReadStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                StageToAxesMap *stage_to_axes,
                                                te::Schedule *schedule,
                                                const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  // copy stage here, for the original stage will change after apply
  auto stage = (*stages)[stage_id];
  std::vector<te::Stage> reader_stages;
  for (size_t i = 0; i < reader_stage_ids.size(); ++i) {
    reader_stages.push_back((*stages)[reader_stage_ids[i]]);
  }

  auto out = ApplyToSchedule(stages, stage_to_axes, schedule);

  ss << CleanName(out->op->name) << " = "
      << "s.cache_read(" << CleanName(stage->op->name) << ", \""
      << scope_name << "\", ["
      << CleanName(reader_stages[0]->op->name);
  for (size_t i = 1; i < reader_stage_ids.size(); ++i) {
    ss << ", " << CleanName(reader_stages[i]->op->name);
  }
  ss << "])\n";

  const auto& iters = out->op->root_iter_vars();
  for (size_t i = 0; i < iters.size(); ++i) {
    ss << CleanName(iters[i]->var->name_hint);
    if (i != iters.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = " << "tuple(" << CleanName(out->op->name)
      << ".op.axis)\n";

  return ss.str();
}

/********** Cache Write **********/
CacheWriteStep::CacheWriteStep(int stage_id, std::string scope_name) {
  auto node = make_object<CacheWriteStepNode>();
  node->stage_id = stage_id;
  node->scope_name = std::move(scope_name);
  data_ = std::move(node);
}

Array<te::Tensor> CacheWriteStepNode::ApplyToSchedule(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule) const {
  te::Stage& stage = (*stages)[stage_id];

  Array<te::Tensor> tensor_array;
  // If the target stage has multi outputs, TVM requires to cache_write
  // all of them or schedule.cache_write will raise an error
  for (auto i = 0; i < stage->op->num_outputs(); ++i) {
    tensor_array.push_back(stage->origin_op.output(i));
  }
  auto outs = schedule->cache_write(tensor_array, scope_name);

  UpdateStageAxis(stage, stage_to_axes);
  // Even if there is multi outputs, TVM schedule only generate one
  // new stage
  const auto& new_stage = (*schedule)[outs[0]->op];
  UpdateStageAxis(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id, new_stage);

  return outs;
}

std::string CacheWriteStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                                 StageToAxesMap *stage_to_axes,
                                                 te::Schedule *schedule,
                                                 const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  // copy stage here, for the original stage will change after apply
  te::Stage stage = (*stages)[stage_id];

  auto outs = ApplyToSchedule(stages, stage_to_axes, schedule);

  for (size_t i = 0; i < outs.size(); ++i) {
    ss << CleanName(outs[i]->op->name) << ", ";
  }
  ss << "= " << "s.cache_write(["
     << CleanName(stage->op.output(0)->op->name);
  for (auto i = 1; i < stage->op->num_outputs(); ++i) {
    ss << ", " << CleanName(stage->op.output(i)->op->name);
  }
  ss << "], \"" << scope_name << "\")\n";

  for (const auto& out : outs) {
    const auto& iters = out->op->root_iter_vars();
    for (size_t i = 0; i < iters.size(); ++i) {
      ss << CleanName(iters[i]->var->name_hint);
      if (i != iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << " = " << "tuple(" << CleanName(out->op->name)
      << ".op.axis)"
      << " + " << "tuple(" << CleanName(out->op->name)
      << ".op.reduce_axis)\n";
  }

  return ss.str();
}

/********** Pragma **********/
PragmaStep::PragmaStep(int stage_id, int iter_id, std::string pragma_type) {
  auto node = make_object<PragmaStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->pragma_type = std::move(pragma_type);
  data_ = std::move(node);
}

void PragmaStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
                                     StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];
  if (StrStartsWith(pragma_type, "auto_unroll_max_step")) {
    size_t pos = pragma_type.find('$');
    int value = atoi(pragma_type.c_str() + pos + 1);
    stage.pragma(axes[iter_id], "auto_unroll_max_step", value);
    stage.pragma(axes[iter_id], "unroll_explicit", true);
  } else {
    stage.pragma(axes[iter_id], pragma_type);
  }
}

std::string PragmaStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                             StageToAxesMap *stage_to_axes,
                                             te::Schedule *schedule,
                                             const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];

  if (StrStartsWith(pragma_type, "auto_unroll_max_step")) {
    size_t pos = pragma_type.find('$');
    int value = atoi(pragma_type.c_str() + pos + 1);
    ss << "s[" << CleanName(stage->op->name) << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint)
       << ", \"auto_unroll_max_step\", " << value << ")\n";
    ss << "s[" << CleanName(stage->op->name) << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint)
       << ", \"unroll_explicit\", True)\n";
  } else {
    ss << "s[" << CleanName(stage->op->name) << "].pragma("
       << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint) << ", \""
       << pragma_type << "\")\n";
  }

  ApplyToSchedule(stages, stage_to_axes);
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

Array<te::Tensor> RfactorStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes, te::Schedule *schedule) const {
  const auto& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];

  const te::Tensor& tensor = stage->origin_op.output(0);
  const IterVar& axis = axes[iter_id];
  auto outs = schedule->rfactor(tensor, axis, factor_iter_id);

  UpdateStageAxis(stage, stage_to_axes);

  const auto& new_stage = (*schedule)[outs[0]->op];
  UpdateStageAxis(new_stage, stage_to_axes);
  stages->insert(stages->begin() + stage_id, new_stage);

  return outs;
}

std::string RfactorStepNode::PrintAsPythonAPI(std::vector<te::Stage> *stages,
                                              StageToAxesMap *stage_to_axes,
                                              te::Schedule *schedule,
                                              const std::vector<Step>& transform_steps) const {
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
  ss << " = " << "s.rfactor("
     << tensor_name << ", "
     << axis_name << ", "
     << factor_iter_id << ")\n";

  for (const auto& out : outs) {
    const auto& iters = out->op->root_iter_vars();
    for (size_t i = 0; i < iters.size(); ++i) {
      ss << CleanName(iters[i]->var->name_hint);
      if (i != iters.size() - 1) {
        ss << ", ";
      }
    }
    ss << " = " << "tuple(" << CleanName(out->op->name)
      << ".op.axis)"
      << " + " << "tuple(" << CleanName(out->op->name)
      << ".op.reduce_axis)\n";
  }

  const auto& output = (*stages)[stage_id + 1]->op.output(0);
  const auto& iters = output->op->root_iter_vars();
  for (size_t i = 0; i < iters.size(); ++i) {
    ss << CleanName(iters[i]->var->name_hint);
    if (i != iters.size() - 1) {
      ss << ", ";
    }
  }
  ss << " = " << "tuple(s[" << CleanName(output->op->name)
    << "].op.axis)"
    << " + " << "tuple(s[" << CleanName(output->op->name)
    << "].op.reduce_axis)\n";

  return ss.str();
}

/********** Storage Align **********/
StorageAlignStep::StorageAlignStep(int stage_id, int iter_id,
                                   int factor, int offset) {
  auto node = make_object<StorageAlignStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->factor = factor;
  node->offset = offset;
  data_ = std::move(node);
}

void StorageAlignStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];
  stage.storage_align(axes[iter_id], factor, offset);
}

std::string StorageAlignStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  ss << "s[" << CleanName(stage->op->name) << "].storage_align("
     << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint) << ", "
     << factor << ", " << offset << ")\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

/********** Tensorize **********/
TensorizeStep::TensorizeStep(int stage_id, int iter_id,
                             std::string ti_func_name) {
  auto node = make_object<TensorizeStepNode>();
  node->stage_id = stage_id;
  node->iter_id = iter_id;
  node->ti_func_name = ti_func_name;
  data_ = std::move(node);
}

void TensorizeStepNode::ApplyToSchedule(std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes) const {
  te::Stage& stage = (*stages)[stage_id];
  const std::vector<IterVar>& axes = (*stage_to_axes)[stage];
  auto func = tvm::runtime::Registry::Get(ti_func_name);
  CHECK(func != nullptr) << "Cannot find the tensorize intrinsic func";
  tvm::te::TensorIntrin res = (*func)();
  CHECK(res.defined()) << "Tensorize intrinsic func must return a "
                       << "tvm::te::TensorIntrin object";
  stage.tensorize(axes[iter_id], res);
}

std::string TensorizeStepNode::PrintAsPythonAPI(
    std::vector<te::Stage> *stages, StageToAxesMap *stage_to_axes,
    te::Schedule *schedule, const std::vector<Step>& transform_steps) const {
  std::stringstream ss;
  const auto& stage = (*stages)[stage_id];
  ss << "s[" << CleanName(stage->op->name) << "].tensorize("
     << CleanName((*stage_to_axes)[stage][iter_id]->var->name_hint) << ", "
     << ti_func_name << "())\n";

  ApplyToSchedule(stages, stage_to_axes);
  return ss.str();
}

}  // namespace ansor
}  // namespace tvm
