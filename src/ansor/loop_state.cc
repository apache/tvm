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
 * \file ansor/loop_state.cc
 * \brief  An lightweight IR (intermediate representation) for loop structures.
 * see ansor/loop_state.h for more explanation.
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
Iterator::Iterator(std::string name, Range range, IteratorType iter_type,
                   IteratorAnnotation annotation,
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
  data_ = std::move(node);
}

Stage::Stage(te::Operation op) {
  auto node = make_object<StageNode>();
  if (op->IsInstance<te::ComputeOpNode>()) {
    node->op_type = kCompute;
    auto* pop = op.as<te::ComputeOpNode>();

    for (const auto& axis : pop->axis) {
      node->iters.push_back(Iterator(CleanName(axis->var->name_hint),
                                     axis->dom, kSpace, kNone));
    }
    for (const auto& axis : pop->reduce_axis) {
      node->iters.push_back(Iterator(CleanName(axis->var->name_hint),
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
  data_ = std::move(node);
}

Stage::Stage(te::Operation op, StageType op_type,
             const std::vector<Iterator>& iters, ComputeAtType compute_at,
             StageAttributes attrs) {
  auto node = make_object<StageNode>();
  node->op = std::move(op);
  node->op_type = op_type;
  node->iters = iters;
  node->compute_at = compute_at;
  node->attrs = attrs;
  data_ = std::move(node);
}

Stage::Stage(te::Operation op, StageType op_type, std::vector<Iterator>&& iters,
             ComputeAtType compute_at, StageAttributes attrs) {
  auto node = make_object<StageNode>();
  node->op = std::move(op);
  node->op_type = op_type;
  node->iters = std::move(iters);
  node->compute_at = compute_at;
  node->attrs = attrs;
  data_ = std::move(node);
}

State::State(const Array<te::Operation>& ops) {
  auto node = make_object<StateNode>();
  for (const auto& op : ops) {
    node->stages.push_back(Stage(op));
  }
  node->complete = true;
  node->aux_info = ObjectRef();
  data_ = std::move(node);
}

State::State(const std::vector<Stage>& stages,
             const std::vector<Step>& transform_steps, bool complete,
             ObjectRef aux_info) {
  auto node = make_object<StateNode>();
  node->stages = stages;
  node->transform_steps = transform_steps;
  node->complete = complete;
  node->aux_info = std::move(aux_info);
  data_ = std::move(node);
}

// Schedule primitives api
void State::reorder(int stage_id, const std::vector<Iterator>& order) {
  const Stage& stage = operator->()->stages[stage_id];

  CHECK_EQ(order.size(), stage->iters.size()) << "The order of all iterators "
                                                 "should be specified";
  std::vector<int> after_ids;
  GetIndices(stage->iters, order, &after_ids);
  ReorderStep step = ReorderStep(stage_id, after_ids);
  CopyOnWrite()->transform_steps.push_back(step);
  DoReorderStep(step);
}

std::vector<Iterator> State::split(int stage_id, const Iterator& it,
                                   const std::vector<PrimExpr>& lengths,
                                   bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];

  SplitStep step =
      SplitStep(stage_id, GetIndex(stage->iters, it),
                it->range.defined() ? it->range->extent : PrimExpr(),
                lengths, inner_to_outer);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoSplitStep(step);
}

Iterator State::fuse(int stage_id, const std::vector<Iterator>& iters) {
  const Stage& stage = operator->()->stages[stage_id];
  std::vector<int> indices;
  GetIndices(stage->iters, iters, &indices);
  FuseStep step = FuseStep(stage_id, indices);
  CopyOnWrite()->transform_steps.push_back(step);
  return DoFuseStep(step);
}

// Steps' implementations
void State::DoReorderStep(const ReorderStep& step) {
  const Stage& stage = operator->()->stages[step->stage_id];

  std::vector<Iterator> iters;
  for (auto x : step->after_ids) {
    iters.push_back(stage->iters[x]);
  }

  StateNode* pstate = CopyOnWrite();
  pstate->stages[step->stage_id] = Stage(
      stage->op, stage->op_type, std::move(iters), stage->compute_at,
      stage->attrs);
}

// common part for DoSplitStep, DoFollowSplitStep, and DoFollowFusedSplitStep
std::vector<Iterator> State::DoSplitStepCommon(
    int stage_id, int iter_id, const std::vector<PrimExpr>& lengths,
    bool inner_to_outer) {
  const Stage& stage = operator->()->stages[stage_id];
  const Iterator& it = stage->iters[iter_id];

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
      res = Iterator(name, Range::make_by_min_extent(tosplit_min, l),
                     it->iter_type, kNone);
      tosplit_min = 0;
      tosplit_extent = indexdiv(tosplit_extent + l - 1, l);
    } else {
      res = Iterator(name, Range(), it->iter_type, kNone);
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
        Iterator(it->name + ".0", range, it->iter_type, kNone));
    std::reverse(outs.begin(), outs.end());
  } else {
    outs.push_back(
        Iterator(it->name + "." + std::to_string(lengths.size()),
                 range, it->iter_type, kNone));
  }

  std::vector<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(),
                   stage->iters.begin() + iter_id);
  new_iters.insert(new_iters.end(), outs.begin(), outs.end());
  new_iters.insert(new_iters.end(), stage->iters.begin() + iter_id + 1,
                   stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages[stage_id] = Stage(
      stage->op, stage->op_type, std::move(new_iters), stage->compute_at,
      stage->attrs);

  return outs;
}

std::vector<Iterator> State::DoSplitStep(const SplitStep& step) {
  return DoSplitStepCommon(step->stage_id, step->iter_id, step->lengths,
                           step->inner_to_outer);
}

Iterator State::DoFuseStep(const FuseStep& step) {
  int stage_id = step->stage_id;
  const Stage& stage = operator->()->stages[stage_id];

  std::string new_name;
  PrimExpr new_extent = 1;
  IteratorType new_iter_type = kSpecial;

  std::vector<Iterator> ori_iters;
  for (size_t i = 0; i < step->fused_ids.size(); ++i) {
    if (i > 0) {
      CHECK_EQ(step->fused_ids[i], step->fused_ids[i - 1] + 1);
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
      Iterator(new_name, range, new_iter_type, kNone, &ori_iters);
  std::vector<Iterator> new_iters;
  new_iters.insert(new_iters.end(), stage->iters.begin(),
                   stage->iters.begin() + step->fused_ids.front());
  new_iters.push_back(new_it);
  new_iters.insert(new_iters.end(),
                   stage->iters.begin() + step->fused_ids.back() + 1,
                   stage->iters.end());

  StateNode* pstate = CopyOnWrite();
  pstate->stages[stage_id] = Stage(
      stage->op, stage->op_type, std::move(new_iters), stage->compute_at,
      stage->attrs);

  return new_it;
}

void State::DoStep(const Step& step, const ComputeDAG& dag) {
  if (auto ps = step.as<ReorderStepNode>()) {
    DoReorderStep(GetRef<ReorderStep>(ps));
  } else if (auto ps = step.as<SplitStepNode>()) {
    DoSplitStep(GetRef<SplitStep>(ps));
  } else if (auto ps = step.as<FuseStepNode>()) {
    DoFuseStep(GetRef<FuseStep>(ps));
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
    *os << stage->op->name
        << " auto_unroll: " << stage->attrs.auto_unroll_max_step << "\n";
  }
  if (stage->attrs.storage_offset != 0) {
    for (size_t j = 0; j < base_indent; ++j) {
      *os << " ";
    }
    *os << stage->op->name
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
  }

  for (size_t j = 0; j < base_indent + indent; ++j) {
    *os << " ";
  }
  *os << stage->op->name << " = ...\n";
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

TVM_REGISTER_GLOBAL("ansor.StateEqual")
.set_body_typed([](State state1, State state2) {
  return std::equal_to<State>()(state1, state2);
});

}  // namespace ansor
}  // namespace tvm
