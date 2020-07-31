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

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../arith/pattern_match.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;

template <class T>
using OperationMap = AccessAnalyzerNode::OperationMap<T>;
using OperationSet = std::unordered_set<te::Operation, ObjectHash, ObjectEqual>;

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

// Extract all tensor accesses in an expr
class ReadAccessExtractor : public StmtExprVisitor {
 public:
  void Extract(PrimExpr expr) { this->VisitExpr(expr); }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      has_branch = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ProducerLoadNode* op) final {
    read_access[Downcast<te::Tensor>(op->producer)->op].emplace_back(op->indices.begin(),
                                                                     op->indices.end());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    has_branch = true;
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const SelectNode* op) final {
    has_branch = true;
    StmtExprVisitor::VisitExpr_(op);
  }

  // All read accesses to all operations
  // The innermost vector stores mulit-dimentional indices.
  // The middle vector stores possible multiple accesses
  OperationMap<std::vector<std::vector<PrimExpr>>> read_access;
  // Whether this expression has branch
  bool has_branch{false};
};

// Returns whether the expr equals to the var with an optional const shift
bool IsConstShiftEqual(const Var& var, const PrimExpr& expr) {
  arith::PVar<PrimExpr> x;
  arith::PVar<IntImm> c;

  if (((x + c).Match(expr) || (x - c).Match(expr) || (c + x).Match(expr) || x.Match(expr)) &&
      x.Eval().same_as(var)) {
    return true;
  }
  return false;
}

// Return whether the access to an operation is a simple access
// (i.e. all index is just a variable with an optional constant shift)
// For example, A[i][j], A[i+1][j] are simple accesses but A[i][j+i] is not.
bool IsSimpleAccess(const te::Operation& op, const std::vector<PrimExpr>& indices,
                    bool* axis_missing, bool* axis_duplicated, bool* same_order) {
  auto cop = op.as<te::ComputeOpNode>();
  if (cop == nullptr) {
    return false;
  }

  std::vector<int> index_to_var_idx;
  std::vector<int> var_idx_ct(cop->axis.size(), 0);

  for (const auto& expr : indices) {
    if (!is_const_int(expr)) {
      bool found = false;
      for (size_t i = 0; i < cop->axis.size(); ++i) {
        if (IsConstShiftEqual(cop->axis[i]->var, expr)) {
          index_to_var_idx.push_back(i);
          var_idx_ct[i]++;
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }

  *axis_missing = false;     // Some axes are missing
  *axis_duplicated = false;  // Some axes appear more than once
  *same_order = true;        // The axis order is the same as op->axis
  for (int ct : var_idx_ct) {
    if (ct == 0) {
      *axis_missing = true;
    } else if (ct > 1) {
      *axis_duplicated = true;
    }
  }
  for (size_t i = 1; i < index_to_var_idx.size(); ++i) {
    if (index_to_var_idx[i] < index_to_var_idx[i - 1]) {
      *same_order = false;
      break;
    }
  }

  return true;
}

// Gather all VarNodes in an expr
void GatherVars(const PrimExpr& expr, std::unordered_set<const VarNode*>* vars) {
  PostOrderVisit(expr, [&vars](const ObjectRef& node) {
    if (const VarNode* op = node.as<VarNode>()) {
      vars->insert(op);
    }
  });
}

// Check whether an expr has expensive operations (e.g. exp)
bool HasExpensiveOp(const PrimExpr& expr) {
  bool found = false;
  PostOrderVisit(expr, [&found](const ObjectRef& node) {
    if (const CallNode* op = node.as<CallNode>()) {
      if (op->op.as<OpNode>()->name == "tir.exp") {
        found = true;
      }
    }
  });
  return found;
}

AccessAnalyzer::AccessAnalyzer(const Array<te::Tensor>& tensors) {
  auto node = make_object<AccessAnalyzerNode>();
  OperationMap<bool> has_branch;

  // Get all ops in topological order
  node->ops_topo_order = TopoSortOps(tensors);

  arith::Analyzer analyzer;

  // Build read & write access map
  for (const auto& op : node->ops_topo_order) {
    if (op->IsInstance<te::PlaceholderOpNode>()) {
      node->read_from[op] = OperationMap<std::vector<std::vector<PrimExpr>>>();
    } else if (auto cop = op.as<te::ComputeOpNode>()) {
      ReadAccessExtractor extractor;
      for (const auto& exp : cop->body) {
        extractor.Extract(exp);
      }

      // read_by and read_from map
      for (const auto& iter : extractor.read_access) {
        std::vector<std::vector<PrimExpr>>& accesses = node->read_by[iter.first][op];
        accesses.insert(accesses.begin(), iter.second.begin(), iter.second.end());
      }

      node->read_from[op] = std::move(extractor.read_access);
      has_branch[op] = extractor.has_branch;

      // compute number of common outer iterators
      for (const auto& pair : node->read_from[op]) {
        const te::Operation& producer = pair.first;
        const std::vector<std::vector<PrimExpr>>& access_list = pair.second;
        const Array<PrimExpr>& output_shape = op->output_shape(0);
        const Array<PrimExpr>& producer_shape = producer->output_shape(0);

        int n_common;
        for (n_common = 0;
             n_common < static_cast<int>(std::min(output_shape.size(), producer_shape.size()));
             n_common++) {
          if (!is_zero(analyzer.Simplify(output_shape[n_common] - producer_shape[n_common]))) {
            break;
          }

          bool injective = true;
          for (const auto& access : access_list) {
            if (!IsConstShiftEqual(cop->axis[n_common]->var, access[n_common])) {
              injective = false;
              break;
            }
          }

          if (!injective) {
            break;
          }
        }

        node->num_common_outer_iterators[op][producer] = n_common;
        node->num_common_outer_iterators[producer][op] = n_common;
      }
    } else {
      LOG(FATAL) << "Invalid op: " << op;
    }
  }

  // Do some static analysis on ComputeOps
  for (const auto& op : node->ops_topo_order) {
    if (op->IsInstance<te::PlaceholderOpNode>()) {
      node->is_simple_access[op] = true;
      node->needs_multi_level_tiling[op] = false;
      node->is_strict_inlineable[op] = false;
      node->is_output[op] = false;
    } else if (auto cop = op.as<te::ComputeOpNode>()) {
      // check whether this op is element-wise and strict-inlineable
      bool is_simple_access = true;
      bool is_strict_inlineable = true;

      bool axis_missing, axis_duplicated, same_order;
      for (const auto& pair : node->read_from[op]) {
        const std::vector<std::vector<PrimExpr>>& access_list = pair.second;
        for (const auto& access : access_list) {
          if (!auto_scheduler::IsSimpleAccess(op, access, &axis_missing, &axis_duplicated,
                                              &same_order)) {
            is_simple_access = false;
            is_strict_inlineable = false;
            break;
          }
          if (!same_order || axis_duplicated) {
            // do not strictly inline transpose
            is_strict_inlineable = false;
          }
        }
        if (!is_simple_access) {
          break;
        }
      }

      // don't strictly inline expensive op (e.g. exp)
      bool has_expensive_op = false;
      for (const auto& expr : cop->body) {
        has_expensive_op |= HasExpensiveOp(expr);
      }
      if (has_expensive_op || has_branch[op]) {
        is_strict_inlineable = false;
      }

      node->is_simple_access[op] = is_simple_access;
      node->is_strict_inlineable[op] = is_strict_inlineable;

      // check whether the op needs multi-level tiling
      bool needs_multi_level_tiling = false;
      int n_missing = 0;

      for (const auto& pair : node->read_from[op]) {
        const std::vector<std::vector<PrimExpr>>& access_list = pair.second;
        std::unordered_set<const VarNode*> vars;
        for (const std::vector<PrimExpr>& access : access_list) {
          for (const PrimExpr& expr : access) {
            GatherVars(expr, &vars);
          }
        }

        for (const auto& axis : cop->axis) {
          if (GetIntImm(axis->dom->extent) > 1 && vars.count(axis->var.get()) == 0) {
            n_missing++;
            break;
          }
        }

        if (n_missing >= 2 || (n_missing >= 1 && !cop->reduce_axis.empty())) {
          needs_multi_level_tiling = true;
          break;
        }
      }

      node->needs_multi_level_tiling[op] = needs_multi_level_tiling;

      // check whether the op is output
      node->is_output[op] = node->read_by[op].empty();
    } else {
      LOG(FATAL) << "Invalid op" << op;
    }
  }

  data_ = std::move(node);
}

bool AccessAnalyzer::NeedsMultiLevelTiling(const te::Operation& op) const {
  return operator->()->needs_multi_level_tiling.at(op);
}

bool AccessAnalyzer::IsOutput(const te::Operation& op) const {
  return operator->()->is_output.at(op);
}

bool AccessAnalyzer::IsSimpleAccess(const te::Operation& op) const {
  return operator->()->is_simple_access.at(op);
}

bool AccessAnalyzer::IsStrictInlineable(const te::Operation& op) const {
  return operator->()->is_strict_inlineable.at(op);
}

OperationSet AccessAnalyzer::GetConsumers(const State& state, const te::Operation& op) const {
  OperationSet inlined_ops;
  for (const auto& stage : state->stages) {
    if (stage->compute_at == ComputeAtKind::kInlined) {
      inlined_ops.insert(stage->op);
    }
  }

  OperationSet consumers;
  std::function<void(const te::Operation&)> collect;
  collect = [this, &collect, &inlined_ops, &consumers](const te::Operation& op) {
    for (const auto& iter : operator->()->read_by.at(op)) {
      if (inlined_ops.count(iter.first)) {
        collect(iter.first);
      } else {
        consumers.insert(iter.first);
      }
    }
  };

  collect(op);
  return consumers;
}

OperationSet AccessAnalyzer::GetDirectProducers(const te::Operation& op) const {
  OperationSet producers;
  for (const auto& iter : operator->()->read_from.at(op)) {
    producers.insert(iter.first);
  }
  return producers;
}

OperationSet AccessAnalyzer::GetProducers(const State& state, const te::Operation& op) const {
  OperationSet inlined_ops;
  for (const auto& stage : state->stages) {
    if (stage->compute_at == ComputeAtKind::kInlined) {
      inlined_ops.insert(stage->op);
    }
  }

  OperationSet producers;
  std::function<void(const te::Operation&)> collect;
  collect = [this, &collect, &inlined_ops, &producers](const te::Operation& op) {
    for (const auto& iter : operator->()->read_from.at(op)) {
      if (inlined_ops.count(iter.first)) {
        collect(iter.first);
      } else {
        producers.insert(iter.first);
      }
    }
  };

  collect(op);
  return producers;
}

int AccessAnalyzer::GetNumCommonOuterIterator(const te::Operation& op,
                                              const te::Operation& target_op) const {
  int ret = INT32_MAX;
  bool meet = false;

  std::function<void(const te::Operation&, int)> traverse;
  traverse = [this, &traverse, &target_op, &ret, &meet](const te::Operation& cur_op, int cur_num) {
    if (cur_op == target_op) {
      ret = std::min(ret, cur_num);
      meet = true;
      return;
    }

    for (const auto& iter : operator->()->read_by.at(cur_op)) {
      traverse(
          iter.first,
          std::min(cur_num, operator->()->num_common_outer_iterators.at(cur_op).at(iter.first)));
    }
  };

  traverse(op, op->output_shape(0).size());
  return meet ? ret : 0;
}

bool AccessAnalyzer::ElementWiseMatch(const te::Operation& op,
                                      const te::Operation& target_op) const {
  te::Operation cur_op = op;
  while (cur_op != target_op) {
    const AccessAnalyzerNode::OperationMap<std::vector<std::vector<PrimExpr>>>& map =
    operator->()->read_by.at(cur_op);

    if (map.size() != 1) {
      return false;
    }
    te::Operation next_op = map.begin()->first;

    // Check condition 1: They have the same output size
    auto p_cur = cur_op.as<te::ComputeOpNode>();
    auto p_next = next_op.as<te::ComputeOpNode>();
    if (p_cur == nullptr || p_next == nullptr) {
      return false;
    }

    Array<PrimExpr> output_shape = p_cur->output_shape(0);
    for (int i = 1; i < p_cur->num_outputs(); ++i) {
      if (!IntArrayEqual(p_cur->output_shape(i), output_shape)) {
        return false;
      }
    }
    for (int i = 0; i < p_next->num_outputs(); ++i) {
      if (!IntArrayEqual(p_next->output_shape(i), output_shape)) {
        return false;
      }
    }

    // Check condition 2: The read is elementwise
    const std::vector<std::vector<PrimExpr>> reads = map.begin()->second;
    bool is_simple_access, axis_missing, axis_duplicated, same_order;
    for (const auto& read : reads) {
      is_simple_access = auto_scheduler::IsSimpleAccess(next_op, read, &axis_missing,
                                                        &axis_duplicated, &same_order);
      if (!is_simple_access || axis_missing || axis_duplicated || !same_order) {
        return false;
      }
    }

    cur_op = std::move(next_op);
  }
  return true;
}

// Estimate the number of float operations in an expression
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
        cur_type_code_ = pop->output_dtype(0).code();
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

#define VisitBinary(Node)                                         \
  double VisitExpr_(const Node* op) final {                       \
    double base = op->dtype.code() == cur_type_code_ ? 1.0 : 0.0; \
    return base + VisitExpr(op->a) + VisitExpr(op->b);            \
  }

#define VisitUnary(Node)                                          \
  double VisitExpr_(const Node* op) final {                       \
    double base = op->dtype.code() == cur_type_code_ ? 1.0 : 0.0; \
    return base + VisitExpr(op->a);                               \
  }

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
  int cur_type_code_;
};

ComputeDAG::ComputeDAG(Array<te::Tensor> tensors) {
  auto node = make_object<ComputeDAGNode>();
  node->tensors = std::move(tensors);
  node->access_analyzer = AccessAnalyzer(node->tensors);
  node->ops = node->access_analyzer->ops_topo_order;
  node->flop_ct = FlopEstimator().EstimateFlop(node->ops);
  node->init_state = State(node->ops);
  data_ = std::move(node);
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
  // Call each step's ApplyToSchedule method
  for (const auto& step : transform_steps) {
    StepApplyToSchedule(step, stages, stage_to_axes, &schedule, transform_steps);
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
    ss << StepPrintAsPythonAPI(step, &stages, &stage_to_axes, &schedule, transform_steps);
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
    for (const auto& step : pstate->transform_steps) {
      StepApplyToState(step, &ret_state, *this);
    }
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

ComputeDAG ComputeDAG::ReplayAndGetDAG(const Array<Step>& transform_steps) const {
  te::Schedule sch;
  Array<te::Tensor> old_tensors;
  std::tie(sch, old_tensors) = ApplySteps(transform_steps);

  Array<te::Tensor> new_tensors;
  for (auto stage : sch->stages) {
    if (stage->op->IsInstance<te::PlaceholderOpNode>() || stage->is_output) {
      for (auto i = 0; i < stage->op->num_outputs(); ++i) {
        new_tensors.push_back(stage->op.output(i));
      }
    }
  }

  return ComputeDAG(new_tensors);
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
