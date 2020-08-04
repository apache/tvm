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
 * \brief Transformation steps. For each schedule primitive, there is a corresponding transform
 * step.
 */

#include "transform_step.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>

#include <utility>

#include "loop_state.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

/********** Reorder **********/
ReorderStep::ReorderStep(int stage_id, const Array<Integer>& after_ids) {
  auto node = make_object<ReorderStepNode>();
  node->stage_id = stage_id;
  for (const auto& x : after_ids) {
    CHECK(x->IsInstance<IntImmNode>());
  }
  node->after_ids = after_ids;
  data_ = std::move(node);
}

void ReorderStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);
  CHECK_EQ(after_ids.size(), axes.size());

  Array<IterVar> new_axes;
  new_axes.reserve(axes.size());
  for (auto i : after_ids) {
    new_axes.push_back(axes[i]);
  }
  stage.reorder(new_axes);

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
}

String ReorderStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                         StageToAxesMap* stage_to_axes) const {
  const auto& stage = (*stages)[stage_id];
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
  CHECK_EQ(outs.size(), lengths.size() + 1);

  std::stringstream ss;
  int size = static_cast<int>(lengths.size());
  if (inner_to_outer) {
    for (int i = size - 1; i >= 0; i--) {
      ss << CleanName(outs[size - i]->var->name_hint) << ", "
         << CleanName(outs[size - i - 1]->var->name_hint) << " = s[" << func_name << "].split("
         << CleanName(to_split->var->name_hint) << ", factor=" << lengths[i] << ")\n";
      to_split = outs[size - i];
    }
  } else {
    for (int i = 0; i < size; i++) {
      ss << CleanName(outs[i]->var->name_hint) << ", " << CleanName(outs[i + 1]->var->name_hint)
         << " = s[" << func_name << "].split(" << CleanName(to_split->var->name_hint)
         << ", nparts=" << lengths[i] << ")\n";
      to_split = outs[i + 1];
    }
  }

  return ss.str();
}

SplitStep::SplitStep(int stage_id, int iter_id, Optional<PrimExpr> extent,
                     const Array<Optional<Integer>>& lengths, bool inner_to_outer) {
  auto node = make_object<SplitStepNode>();
  node->stage_id = stage_id;
  // Extent can be a unreducible expression in some special cases
  if (extent && extent.value()->IsInstance<IntImmNode>()) {
    node->extent = tvm::Downcast<Integer>(extent.value());
  }
  node->iter_id = iter_id;
  node->lengths = lengths;
  node->inner_to_outer = inner_to_outer;
  data_ = std::move(node);
}

Array<IterVar> SplitStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                              StageToAxesMap* stage_to_axes) const {
  return ApplySplitToSchedule(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
}

String SplitStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                       StageToAxesMap* stage_to_axes) const {
  return PrintSplitAsPythonAPI(stages, stage_to_axes, stage_id, iter_id, lengths, inner_to_outer);
}

/********** Fuse **********/
FuseStep::FuseStep(int stage_id, const Array<Integer>& fused_ids) {
  auto node = make_object<FuseStepNode>();
  node->stage_id = stage_id;
  for (const auto& x : fused_ids) {
    CHECK(x->IsInstance<IntImmNode>());
  }
  node->fused_ids = fused_ids;
  data_ = std::move(node);
}

IterVar FuseStepNode::ApplyToSchedule(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  auto stage = (*stages)[stage_id];
  const Array<IterVar>& axes = stage_to_axes->at(stage);

  Array<IterVar> to_fuse;
  for (const auto& i : fused_ids) {
    to_fuse.push_back(axes[i]);
  }
  IterVar fused_axis;
  stage.fuse(to_fuse, &fused_axis);

  Array<IterVar> new_axes;
  new_axes.insert(new_axes.end(), axes.begin(), axes.begin() + fused_ids.front());
  new_axes.push_back(fused_axis);
  new_axes.insert(new_axes.end(), axes.begin() + fused_ids.back() + 1, axes.end());

  stage_to_axes->Set(stage, std::move(new_axes));
  stages->Set(stage_id, std::move(stage));
  return fused_axis;
}

String FuseStepNode::PrintAsPythonAPI(Array<te::Stage>* stages,
                                      StageToAxesMap* stage_to_axes) const {
  const auto& stage = (*stages)[stage_id];
  std::stringstream to_fuse;

  for (size_t i = 0; i < fused_ids.size(); ++i) {
    to_fuse << CleanName(stage_to_axes->at(stage)[fused_ids[i]]->var->name_hint);
    if (i != fused_ids.size() - 1) {
      to_fuse << ", ";
    }
  }

  std::stringstream ss;
  const auto& fused = ApplyToSchedule(stages, stage_to_axes);

  ss << CleanName(fused->var->name_hint) << " = s[" << CleanName(stage->op->name) << "].fuse("
     << to_fuse.str() << ")\n";

  return ss.str();
}

}  // namespace auto_scheduler
}  // namespace tvm
