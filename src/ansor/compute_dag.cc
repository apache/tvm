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
 * \file ansor/compute_dag.cc
 * \brief Compute declaration graph and its related analysis tools
 */

#include "compute_dag.h"
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/te/operation.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/registry.h>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <utility>
#include <string>
#include <set>
#include <vector>
#include "transform_step.h"

namespace tvm {
namespace ansor {

using namespace tvm::tir;

TVM_REGISTER_NODE_TYPE(ComputeDAGNode);

// Topo-sort ops from tensors according to their read-write relations.
// Results are stored in ops
void TopoSortOps(const Array<te::Tensor>& tensors,
                 std::vector<te::Operation>* ops) {
  std::unordered_map<const te::OperationNode*, int> degree;
  std::unordered_map<const te::OperationNode*, std::vector<const te::OperationNode*> > edge_set;
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
  ops->clear();

  using Item = std::pair<const te::OperationNode*, int>;
  auto cmp = [](const Item& left, const Item& right) {
    return left.second < right.second;
  };
  std::priority_queue<Item, std::vector<Item>, decltype(cmp)> queue(cmp);
  for (const auto& iter : degree) {
    if (iter.second == 0) {
      queue.push(Item(iter.first, priority[iter.first]));
    }
  }

  ops->reserve(degree.size());
  while (!queue.empty()) {
    Item item = queue.top();
    queue.pop();
    ops->push_back(GetRef<te::Operation>(item.first));
    for (const auto& dst : edge_set[item.first]) {
      degree[dst] -= 1;
      if (degree[dst] == 0) {
        queue.push(Item(dst, priority[dst]));
      }
    }
  }
}

// Estimate number of float operations in an expression
class FlopEstimator: public ExprFunctor<double(const PrimExpr& n)> {
 public:
  double EstimateFlop(const Array<te::Operation>& ops) {
    double ret = 0;
    for (const auto& op : ops) {
      if (auto pop = op.as<te::ComputeOpNode>()) {
        double num_element = AxisLengthProd(pop->axis);
        if (num_element == -1) {
          fail = true;
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

    return fail ? -1 : ret;
  }

  double VisitExpr_(const ReduceNode* op) final {
    uint64_t num_iter = 1;
    for (const auto& x : op->axis) {
      if (auto imm = x->dom->extent.as<IntImmNode>()) {
        num_iter *= imm->value;
      } else {
        fail = true;
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
    return VisitExpr(op->condition) + std::max(VisitExpr(op->true_value),
        VisitExpr(op->false_value));
  }

#define VisitBinary(Node)                            \
  double VisitExpr_(const Node* op) final {             \
    return 1.0 + VisitExpr(op->a) + VisitExpr(op->b);  \
  }
#define VisitUnary(Node)                             \
  double VisitExpr_(const Node* op) final {             \
    return 1.0 + VisitExpr(op->a);                     \
  }

  VisitBinary(AddNode); VisitBinary(SubNode); VisitBinary(MulNode)
  VisitBinary(DivNode); VisitBinary(ModNode); VisitBinary(FloorDivNode)
  VisitBinary(FloorModNode); VisitBinary(MaxNode); VisitBinary(MinNode);
  VisitBinary(EQNode); VisitBinary(NENode); VisitBinary(LTNode);
  VisitBinary(LENode); VisitBinary(GTNode); VisitBinary(GENode);
  VisitBinary(AndNode); VisitBinary(OrNode); VisitUnary(NotNode);

  double VisitExpr_(const CallNode* op) final {
    double ret = 0.0;
    for (const auto&x : op->args) {
      ret += VisitExpr(x);
    }
    return ret;
  }

  double VisitExprDefault_(const Object* op) final {
    fail = true;
    return -1.0;
  }

  bool fail{false};
};

State ComputeDAG::GetInitState() const {
  return Downcast<State>(operator->()->init_state);
}

ComputeDAG::ComputeDAG(Array<te::Tensor> tensors) {
  auto node = make_object<ComputeDAGNode>();
  FlopEstimator estimator;
  node->tensors = std::move(tensors);
  std::vector<te::Operation> ops;
  TopoSortOps(node->tensors, &ops);
  node->ops = Array<te::Operation>(ops);
  node->flop_ct = estimator.EstimateFlop(node->ops);
  node->init_state = State(node->ops);
  data_ = std::move(node);
}

ComputeDAG::ComputeDAG(const std::string& workload_key) {
  Array<te::Tensor> tens;
  // Call python function to decode the workload_key and get the I/O tensors
  if (const auto* f = runtime::Registry::Get("ansor.workload_key_to_tensors")) {
    tens = (*f)(workload_key);
  } else {
    LOG(FATAL) << "ansor.workload_key_to_tensors is not registered";
  }

  auto node = make_object<ComputeDAGNode>();
  FlopEstimator estimator;
  node->tensors = std::move(tens);
  std::vector<te::Operation> ops;
  TopoSortOps(node->tensors, &ops);
  node->ops = Array<te::Operation>(ops);
  node->flop_ct = estimator.EstimateFlop(node->ops);
  node->init_state = State(node->ops);
  data_ = std::move(node);
}

std::string BaseName(const std::string& str) {
  return str.substr(0, str.rfind("_"));
}

void UpdateStageAxis(const te::Stage& stage, StageToAxesMap *stage_to_axes) {
  if (auto pop = stage->op.as<te::ComputeOpNode>()) {
    std::vector<IterVar>& axes = (*stage_to_axes)[stage];
    axes.clear();
    for (const auto& axis : pop->axis) {
      axes.push_back(axis);
    }
    for (const auto& axis : pop->reduce_axis) {
      axes.push_back(axis);
    }
  } else if (stage->op->IsInstance<te::PlaceholderOpNode>()) {
    {}  // do nothing
  } else {
    LOG(FATAL) << "Invalid op " << stage->op;
  }
}

std::pair<te::Schedule, Array<te::Tensor> > ComputeDAG::ApplySteps(
    const std::vector<Step>& transform_steps,
    LayoutRewriteLevel layout_rewrite_level) const {
  std::vector<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  return ReplaySteps(transform_steps, &stages, &stage_to_axes);
}

std::string ComputeDAG::PrintStepsAsPython(const std::vector<Step>& transform_steps) const {
  std::vector<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  Array<te::Operation> ops;
  for (const auto& op : operator->()->ops) {
    if (!op->IsInstance<te::PlaceholderOpNode>()) {
      ops.push_back(op);
    }
  }
  te::Schedule schedule = te::create_schedule({ops.back()});

  // init axes
  for (const auto& x : operator->()->ops) {
    const te::Stage& stage = schedule.operator[](x);
    stages.push_back(stage);
    UpdateStageAxis(stage, &stage_to_axes);
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
      ss << " = " << "tuple(" << stage->op->name << ".op.axis)"
         << " + " << "tuple(" << stage->op->name << ".op.reduce_axis)\n";
    }
  }

  for (const auto& step : transform_steps) {
    ss << step->PrintAsPythonAPI(&stages, &stage_to_axes, &schedule,
                                 transform_steps);
  }

  return ss.str();
}

State ComputeDAG::ReplayAndInferBound(
    const std::vector<Step>& transform_steps) const {
  State ret_state = GetInitState();
  StateNode* pstate = ret_state.CopyOnWrite();
  pstate->transform_steps = transform_steps;
  ret_state.DoSteps(transform_steps, *this);

  InferBoundCommon(pstate);

  return ret_state;
}

State ComputeDAG::InferBound(const State& state) const {
  State ret_state = state;
  StateNode* pstate = ret_state.CopyOnWrite();

  InferBoundCommon(pstate);

  return ret_state;
}

void ComputeDAG::InferBound(std::vector<State>* states) const {
  std::vector<State> out_states(states->size(), State());

  auto worker_func = [&states, &out_states, this](int idx) {
    try {
      out_states[idx] = this->InferBound((*states)[idx]);
    } catch (dmlc::Error &e) {
      LOG(WARNING) << "InferBound fails on the state:\n" << (*states)[idx]
                   << "\n" << e.what() << std::endl;
    }
  };

  // Lower states in parallel
  ThreadPool& pool = ThreadPool::Global();
  pool.BeginBatch(states->size());
  for (size_t i = 0; i < states->size(); ++i) {
    pool.Enqueue(worker_func, i);
  }
  pool.WaitBatch();

  *states = std::move(out_states);
}

void ComputeDAG::InferBoundCommon(StateNode* pstate) const {
  std::vector<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  te::Schedule sch;
  Array<te::Tensor> tensors;
  Map<IterVar, Range> bounds;

  std::tie(sch, tensors) = ReplaySteps(pstate->transform_steps, &stages,
                                       &stage_to_axes);
  sch = sch.normalize();
  bounds = te::InferBound(sch);

  for (size_t i = 0; i < pstate->stages.size(); ++i) {
    const Stage& stage = pstate->stages[i];

    if (stage->compute_at == kInlined) {
      continue;
    }

    std::vector<Iterator> new_iters;
    new_iters.reserve(stage->iters.size());
    for (size_t j = 0; j < stage->iters.size(); ++j) {
      const Iterator& iter = stage->iters[j];
      const IterVar& axis = stage_to_axes.at(stages[i])[j];

      auto find_res = bounds.find(axis);
      if (find_res != bounds.end()) {
        new_iters.push_back(Iterator(iter->name, (*find_res).second,
                                     iter->iter_type, iter->annotation,
                                     &iter->ori_iters, iter->attr));
      } else {
        LOG(FATAL) << "Infer bound fails";
      }
    }

    pstate->stages[i] = Stage(stage->op, stage->op_type, std::move(new_iters),
                              stage->compute_at, stage->attrs);
  }
}

std::pair<te::Schedule, Array<te::Tensor> > ComputeDAG::ReplaySteps(
    const std::vector<Step> &transform_steps,
    std::vector<te::Stage> *stages,
    StageToAxesMap *stage_to_axes) const {
  std::vector<te::Operation> ops;
  for (const auto& op : operator->()->ops) {
    if (!op->IsInstance<te::PlaceholderOpNode>()) {
      ops.push_back(op);
    }
  }

  te::Schedule schedule = te::create_schedule({ops.back()});

  // init axes
  stages->reserve(operator->()->ops.size());
  for (const auto& x : operator->()->ops) {
    const te::Stage& stage = schedule.operator[](x);
    stages->push_back(stage);
    UpdateStageAxis(stage, stage_to_axes);
  }

  // Use complete rate for the study in the paper
  const char* complete_rate_str = getenv("ANSOR_PROGRAM_COMPLETE_RATE");
  double complete_rate = -1.0;
  if (complete_rate_str) {
    complete_rate = std::stod(complete_rate_str);
  }
  size_t ct = 0;

  // replay history
  for (const auto& step : transform_steps) {
    if (complete_rate >= 0 && ct++ > transform_steps.size() * complete_rate) {
      break;
    }

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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ComputeDAGNode>([](const ObjectRef& ref, ReprPrinter *p) {
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
            ss << " select(" << select->condition << ", " << select->true_value
               << ", " << select->false_value << ")= " << '('
               << preduce->source[0] << ',' << preduce->source[1] << ")\n";
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

TVM_REGISTER_GLOBAL("ansor.ComputeDAG")
.set_body_typed([](Array<te::Tensor> tensors) {
  return ComputeDAG(tensors);
});

TVM_REGISTER_GLOBAL("ansor.ComputeDAGGetInitState")
.set_body_method(&ComputeDAG::GetInitState);

TVM_REGISTER_GLOBAL("ansor.ComputeDAGApplyStepsFromState")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  ComputeDAG dag = args[0];
  State state = args[1];
  LayoutRewriteLevel layout_rewrite_level = kNoRewrite;
  if (args.size() >= 3) {
    layout_rewrite_level = LayoutRewriteLevel(static_cast<int>((args[2])));
  }

  te::Schedule sch;
  Array<te::Tensor> return_tensors;
  std::tie(sch, return_tensors) = dag.ApplySteps(state->transform_steps, layout_rewrite_level);
  *ret = Array<ObjectRef>{sch, return_tensors};
});

TVM_REGISTER_GLOBAL("ansor.ComputeDAGPrintPythonCodeFromState")
.set_body_typed([](const ComputeDAG& dag, const State& state) {
  return dag.PrintStepsAsPython(state->transform_steps);
});

TVM_REGISTER_GLOBAL("ansor.ComputeDAGInferBoundFromState")
.set_body_typed([](const ComputeDAG& dag, const State& state) {
  return dag.ReplayAndInferBound(state->transform_steps);
});

}  // namespace ansor
}  // namespace tvm
