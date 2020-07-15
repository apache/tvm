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
 * \file auto_scheduler/compute_dag.cc
 * \brief Compute declaration graph and its related analysis tools.
 */

#include "compute_dag.h"

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "loop_state.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;

TVM_REGISTER_NODE_TYPE(ComputeDAGNode);

// Topo-sort ops from tensors according to their read-write relations.
Array<te::Operation> TopoSortOps(const Array<te::Tensor>& tensors) {
  std::unordered_map<const te::OperationNode*, int> degree;
  std::unordered_map<const te::OperationNode*, std::vector<const te::OperationNode*>> edge_set;
  std::unordered_map<const te::OperationNode*, int> priority;
  std::unordered_set<const te::OperationNode*> visited;

  // traverse to build edge_set and count degree
  std::vector<const te::OperationNode*> stack;
  stack.reserve(tensors.size());
  for (const auto& x : tensors) {
    stack.push_back(x->op.operator->());
  }

  int ct = 0;
  while (!stack.empty()) {
    const te::OperationNode* op = stack.back();
    stack.pop_back();
    if (visited.count(op)) {
      continue;
    }

    priority[op] = ct;
    ct++;
    visited.insert(op);

    if (op->IsInstance<te::PlaceholderOpNode>()) {
      degree[op] = 0;
    } else if (auto cop = GetRef<te::Operation>(op).as<te::ComputeOpNode>()) {
      const Array<te::Tensor>& input_tensors = cop->InputTensors();
      degree[op] = input_tensors.size();
      for (const auto& ten : input_tensors) {
        edge_set[ten->op.operator->()].push_back(op);
        stack.push_back(ten->op.operator->());
      }
    } else {
      LOG(FATAL) << "Unsupported op " << GetRef<te::Operation>(op);
    }
  }

  // topo sort
  Array<te::Operation> ops;

  using Item = std::pair<const te::OperationNode*, int>;
  auto cmp = [](const Item& left, const Item& right) { return left.second < right.second; };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp)> queue(cmp);
  for (const auto& iter : degree) {
    if (iter.second == 0) {
      queue.push(Item(iter.first, priority[iter.first]));
    }
  }

  ops.reserve(degree.size());
  while (!queue.empty()) {
    Item item = queue.top();
    queue.pop();
    ops.push_back(GetRef<te::Operation>(item.first));
    for (const auto& dst : edge_set[item.first]) {
      degree[dst] -= 1;
      if (degree[dst] == 0) {
        queue.push(Item(dst, priority[dst]));
      }
    }
  }

  return ops;
}

// Estimate number of float operations in an expression
class FlopEstimator : public ExprFunctor<double(const PrimExpr& n)> {
 public:
  double EstimateFlop(const Array<te::Operation>& ops) {
    double ret = 0;
    for (const auto& op : ops) {
      if (auto pop = op.as<te::ComputeOpNode>()) {
        double num_element = AxisLengthProd(pop->axis);
        if (num_element == -1) {
          fail_ = true;
          break;
        }
        double op_per_element = 0;
        for (const auto& x : pop->body) {
          op_per_element += VisitExpr(x);
        }
        ret += num_element * op_per_element;
      } else if (op->IsInstance<te::PlaceholderOpNode>()) {
        {}  // do nothing
      } else {
        LOG(FATAL) << "Invalid op type " << op;
      }
    }

    return fail_ ? -1 : ret;
  }

  double VisitExpr_(const ReduceNode* op) final {
    uint64_t num_iter = 1;
    for (const auto& x : op->axis) {
      if (auto imm = x->dom->extent.as<IntImmNode>()) {
        num_iter *= imm->value;
      } else {
        fail_ = true;
        num_iter = -1;
      }
    }
    double body_flop = 0;
    for (size_t i = 0; i < op->combiner->result.size(); ++i) {
      body_flop += VisitExpr(op->combiner->result[i]);
      body_flop += VisitExpr(op->source[i]);
    }
    return num_iter * body_flop;
  }

  double VisitExpr_(const FloatImmNode* op) final { return 0.0; }
  double VisitExpr_(const IntImmNode* op) final { return 0.0; }
  double VisitExpr_(const ProducerLoadNode* op) final { return 0.0; }

  double VisitExpr_(const CastNode* op) final { return VisitExpr(op->value); }
  double VisitExpr_(const VarNode* op) final { return 0.0; }

  double VisitExpr_(const SelectNode* op) final {
    return VisitExpr(op->condition) +
           std::max(VisitExpr(op->true_value), VisitExpr(op->false_value));
  }

#define VisitBinary(Node) \
  double VisitExpr_(const Node* op) final { return 1.0 + VisitExpr(op->a) + VisitExpr(op->b); }
#define VisitUnary(Node) \
  double VisitExpr_(const Node* op) final { return 1.0 + VisitExpr(op->a); }

  VisitBinary(AddNode);
  VisitBinary(SubNode);
  VisitBinary(MulNode);
  VisitBinary(DivNode);
  VisitBinary(ModNode);
  VisitBinary(FloorDivNode);
  VisitBinary(FloorModNode);
  VisitBinary(MaxNode);
  VisitBinary(MinNode);
  VisitBinary(EQNode);
  VisitBinary(NENode);
  VisitBinary(LTNode);
  VisitBinary(LENode);
  VisitBinary(GTNode);
  VisitBinary(GENode);
  VisitBinary(AndNode);
  VisitBinary(OrNode);
  VisitUnary(NotNode);

  double VisitExpr_(const CallNode* op) final {
    double ret = 0.0;
    for (const auto& x : op->args) {
      ret += VisitExpr(x);
    }
    return ret;
  }

  double VisitExprDefault_(const Object* op) final {
    fail_ = true;
    return -1.0;
  }

 private:
  bool fail_{false};
};

ComputeDAG::ComputeDAG(Array<te::Tensor> tensors) {
  auto node = make_object<ComputeDAGNode>();
  node->tensors = std::move(tensors);
  node->ops = TopoSortOps(node->tensors);
  node->flop_ct = FlopEstimator().EstimateFlop(node->ops);
  node->init_state = State(node->ops);
  data_ = std::move(node);
}

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

std::pair<te::Schedule, Array<te::Tensor>> ComputeDAG::ApplySteps(
    const Array<Step>& transform_steps, Array<te::Stage>* stages,
    StageToAxesMap* stage_to_axes) const {
  // Temporal object to be used if the input pointer is nullptr
  Array<te::Stage> temp_stages;
  StageToAxesMap temp_stage_to_axes;
  if (stages == nullptr) {
    stages = &temp_stages;
  }
  if (stage_to_axes == nullptr) {
    stage_to_axes = &temp_stage_to_axes;
  }
  Array<te::Operation> ops;
  for (const auto& op : operator->()->ops) {
    if (!op->IsInstance<te::PlaceholderOpNode>()) {
      ops.push_back(op);
    }
  }
  // Create the initial schedule
  // TODO(jcf94): Currently we only checked single output dag for TVM Auto-scheduler,
  // update this after testing with multiple outputs.
  te::Schedule schedule = te::create_schedule({ops.back()});

  // init axes
  for (const auto& x : operator->()->ops) {
    const te::Stage& stage = schedule[x];
    stages->push_back(stage);
    UpdateStageToAxesMap(stage, stage_to_axes);
  }

  // Apply the history steps to TVM schedule
  for (const auto& step : transform_steps) {
    // Call each step's ApplyToSchedule method
    // Note: some steps have extra parameters that must be passed and they may need different
    // return value, so the ApplyToSchedule is not able to be merged to single interface
    if (auto ps = step.as<ReorderStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<SplitStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<FuseStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else {
      LOG(FATAL) << "Invalid Step";
    }
  }

  return std::make_pair(schedule, operator->()->tensors);
}

String ComputeDAG::PrintStepsAsPython(const Array<Step>& transform_steps) const {
  Array<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  Array<te::Operation> ops;
  for (const auto& op : operator->()->ops) {
    if (!op->IsInstance<te::PlaceholderOpNode>()) {
      ops.push_back(op);
    }
  }
  // Create the initial schedule
  // TODO(jcf94): Currently we only checked single output dag for TVM Auto-scheduler,
  // update this after testing with multiple outputs.
  te::Schedule schedule = te::create_schedule({ops.back()});

  // init axes
  for (const auto& x : operator->()->ops) {
    const te::Stage& stage = schedule[x];
    stages.push_back(stage);
    UpdateStageToAxesMap(stage, &stage_to_axes);
  }

  std::stringstream ss;
  for (const auto& stage : stages) {
    if (stage->op->IsInstance<te::ComputeOpNode>()) {
      for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
        ss << stage->leaf_iter_vars[i]->var->name_hint;
        if (i != stage->leaf_iter_vars.size() - 1) {
          ss << ", ";
        }
      }
      ss << " = "
         << "tuple(" << stage->op->name << ".op.axis)"
         << " + "
         << "tuple(" << stage->op->name << ".op.reduce_axis)\n";
    }
  }
  // Call each step's PrintAsPythonAPI method
  for (const auto& step : transform_steps) {
    if (auto ps = step.as<ReorderStepNode>()) {
      ss << ps->PrintAsPythonAPI(&stages, &stage_to_axes);
    } else if (auto ps = step.as<SplitStepNode>()) {
      ss << ps->PrintAsPythonAPI(&stages, &stage_to_axes);
    } else if (auto ps = step.as<FuseStepNode>()) {
      ss << ps->PrintAsPythonAPI(&stages, &stage_to_axes);
    } else {
      LOG(FATAL) << "Invalid Step";
    }
  }

  return ss.str();
}

State ComputeDAG::InferBound(const State& state) const {
  CHECK(state->concrete) << "Only concrete state can be processed to get bound info.";

  State ret_state;
  StateNode* pstate;

  if (state->stages.empty()) {
    // If the input state is incomplete with empty operation stage
    // create a new state from init_state and update it first
    ret_state = operator->()->init_state;
    pstate = ret_state.CopyOnWrite();
    pstate->transform_steps = state->transform_steps;
    ret_state.DoSteps(*this);
  } else {
    ret_state = state;
    pstate = ret_state.CopyOnWrite();
  }

  Array<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  te::Schedule sch;
  Array<te::Tensor> tensors;
  // Replay steps to tvm::Schedule
  std::tie(sch, tensors) = ApplySteps(pstate->transform_steps, &stages, &stage_to_axes);
  sch = sch.normalize();
  // Get bound information from TVM schedule
  Map<IterVar, Range> bounds = te::InferBound(sch);

  // Update the state bound information
  for (size_t i = 0; i < pstate->stages.size(); ++i) {
    const Stage& stage = pstate->stages[i];

    if (stage->compute_at == ComputeAtKind::kInlined) {
      continue;
    }

    Array<Iterator> new_iters;
    new_iters.reserve(stage->iters.size());
    // Get bound information from schedule
    // the StageToAxesMap is used to find the corresponding IterVar in TVM schedule result
    for (size_t j = 0; j < stage->iters.size(); ++j) {
      const Iterator& iter = stage->iters[j];
      const IterVar& axis = stage_to_axes.at(stages[i])[j];

      auto find_res = bounds.find(axis);
      if (find_res != bounds.end()) {
        new_iters.push_back(
            Iterator(iter->name, (*find_res).second, iter->iter_kind, iter->annotation));
      } else {
        LOG(FATAL) << "Infer bound fails";
      }
    }

    pstate->stages.Set(
        i, Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));
  }

  return ret_state;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComputeDAGNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ComputeDAGNode*>(ref.get());
      std::stringstream ss;

      for (const auto& op : node->ops) {
        if (op->IsInstance<te::PlaceholderOpNode>()) {
          ss << op->name << " = PLACEHOLDER " << op.output(0)->shape << "\n";
        } else if (auto pop = op.as<te::ComputeOpNode>()) {
          for (size_t k = 0; k < pop->body.size(); ++k) {
            ss << op->name << "(";
            for (size_t i = 0; i < pop->axis.size(); i++) {
              ss << pop->axis[i]->var->name_hint;
              if (i != pop->axis.size() - 1) {
                ss << ", ";
              }
            }
            ss << ")";
            if (pop->body.size() > 1) {
              ss << ".v" << k;
            }
            if (auto preduce = pop->body[k].as<ReduceNode>()) {
              CHECK_LT(k, preduce->combiner->result.size());
              PrimExpr combiner = preduce->combiner->result[k];
              if (combiner->IsInstance<AddNode>()) {
                ss << " += " << preduce->source[0] << "\n";
              } else if (combiner->IsInstance<MaxNode>()) {
                ss << " max= " << preduce->source[0] << "\n";
              } else if (combiner->IsInstance<MinNode>()) {
                ss << " min= " << preduce->source[0] << "\n";
              } else if (combiner->IsInstance<SelectNode>()) {
                const auto& select = combiner.as<SelectNode>();
                ss << " select(" << select->condition << ", " << select->true_value << ", "
                   << select->false_value << ")= " << '(' << preduce->source[0] << ','
                   << preduce->source[1] << ")\n";
              } else {
                LOG(FATAL) << "Unsupported reduction operator" << combiner;
              }
            } else {
              ss << " = " << pop->body[k] << "\n";
            }
          }
        } else {
          LOG(FATAL) << "Invalid op";
        }
      }

      p->stream << ss.str();
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAG").set_body_typed([](Array<te::Tensor> tensors) {
  return ComputeDAG(tensors);
});

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGApplyStepsFromState")
    .set_body_typed([](const ComputeDAG& dag, const State& state) {
      te::Schedule sch;
      Array<te::Tensor> return_tensors;
      std::tie(sch, return_tensors) = dag.ApplySteps(state->transform_steps);
      return Array<ObjectRef>{sch, return_tensors};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGPrintPythonCodeFromState")
    .set_body_typed([](const ComputeDAG& dag, const State& state) {
      return dag.PrintStepsAsPython(state->transform_steps);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGInferBoundFromState")
    .set_body_typed([](const ComputeDAG& dag, const State& state) {
      return dag.InferBound(state);
    });

}  // namespace auto_scheduler
}  // namespace tvm
