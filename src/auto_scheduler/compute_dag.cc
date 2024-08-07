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
#include <tvm/auto_scheduler/search_policy.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/topi/transform.h>

#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../arith/pattern_match.h"
#include "../relay/transforms/auto_scheduler_layout_rewrite.h"
#include "search_policy/utils.h"
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
  // The innermost vector stores multi-dimensional indices.
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
      node->is_strictly_inlineable[op] = false;
      node->is_output[op] = false;
    } else if (auto cop = op.as<te::ComputeOpNode>()) {
      // check whether this op is element-wise and strict-inlineable
      bool is_simple_access = true;
      bool is_strictly_inlineable = true;

      bool axis_missing, axis_duplicated, same_order;
      for (const auto& pair : node->read_from[op]) {
        const std::vector<std::vector<PrimExpr>>& access_list = pair.second;
        for (const auto& access : access_list) {
          if (!auto_scheduler::IsSimpleAccess(op, access, &axis_missing, &axis_duplicated,
                                              &same_order)) {
            is_simple_access = false;
            is_strictly_inlineable = false;
            break;
          }
          if (!same_order || axis_duplicated) {
            // do not strictly inline transpose
            is_strictly_inlineable = false;
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
        is_strictly_inlineable = false;
      }

      // constant tensor is strict-inlineable
      if (node->read_from[op].empty()) {
        is_strictly_inlineable = true;
      }

      node->is_simple_access[op] = is_simple_access;
      node->is_strictly_inlineable[op] = is_strictly_inlineable;

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

      // do not perform multi-level tiling on "fake reduction" with const tensors
      if (op->attrs.count(SearchPolicyKey::simplify_const_tensor_indices)) {
        needs_multi_level_tiling = false;
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

bool AccessAnalyzer::IsStrictlyInlineable(const te::Operation& op) const {
  return operator->()->is_strictly_inlineable.at(op);
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
        if (pop->attrs.count("FLOP")) {
          // Use user-provided FLOP
          auto pint = pop->attrs["FLOP"].as<IntImmNode>();
          ICHECK(pint != nullptr);
          ret += pint->value;
        } else {
          // Estimate by parsing the compute body
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
        }
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

// Index calculations (e.g., the "i + j" expression in A[i + j]) are not counted in FLOPS.
#define VisitBinary(Node)                                                                     \
  double VisitExpr_(const Node* op) final {                                                   \
    double base = 1.0;                                                                        \
    if ((op->a->dtype.code() != cur_type_code_) && (op->b->dtype.code() != cur_type_code_)) { \
      base = 0.0;                                                                             \
    }                                                                                         \
    return base + VisitExpr(op->a) + VisitExpr(op->b);                                        \
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

void CheckComputeValidity(const te::Schedule& sch) {
  // Check the validity of a compute definition:
  // The name of each iterator should be unique.
  for (auto stage : sch->stages) {
    if (stage->op->IsInstance<te::ComputeOpNode>()) {
      std::unordered_set<std::string> names;
      for (const auto& x : stage->leaf_iter_vars) {
        ICHECK(!names.count(x->var->name_hint))
            << "Find duplicated iterator names in the compute definition: " << x->var->name_hint
            << ". Please use different names for different iterators.";
        names.insert(x->var->name_hint);
      }
    }
  }
}

ComputeDAG::ComputeDAG(Array<te::Tensor> tensors) {
  auto node = make_object<ComputeDAGNode>();
  node->tensors = std::move(tensors);
  node->access_analyzer = AccessAnalyzer(node->tensors);

  Array<te::Operation> out_ops;
  for (const auto& op : node->access_analyzer->ops_topo_order) {
    if (node->access_analyzer.IsOutput(op)) {
      out_ops.push_back(op);
    }
  }
  te::Schedule sch = te::create_schedule(out_ops);
  for (auto stage : sch->stages) {
    node->ops.push_back(stage->op);
  }

  // Make sure it is a valid compute definition
  CheckComputeValidity(sch);

  node->flop_ct = FlopEstimator().EstimateFlop(node->ops);
  node->init_state = State(node->ops);
  data_ = std::move(node);
}

ComputeDAG::ComputeDAG(const te::Schedule& sch) {
  auto node = make_object<ComputeDAGNode>();

  // Make sure it is a valid compute definition
  CheckComputeValidity(sch);

  // Initialize ops. Here we enforce the order of ops and stages are consistent
  for (auto stage : sch->stages) {
    node->ops.push_back(stage->op);
  }

  // Collect input and output tensors
  Array<te::Tensor> tensors;
  for (auto stage : sch->stages) {
    if (stage->op->IsInstance<te::PlaceholderOpNode>() || stage->is_output) {
      for (auto i = 0; i < stage->op->num_outputs(); ++i) {
        tensors.push_back(stage->op.output(i));
      }
    }
  }
  node->tensors = std::move(tensors);
  node->access_analyzer = AccessAnalyzer(node->tensors);
  node->flop_ct = FlopEstimator().EstimateFlop(node->ops);
  node->init_state = State(node->ops);
  data_ = std::move(node);
}

class IndexRewriter : public StmtExprMutator {
 public:
  IndexRewriter(const te::Operation& placeholder_op, const std::string& new_layout)
      : placeholder_op_(placeholder_op) {
    ParseKernelLayout(new_layout, &new_shape_, &new_names_);
  }

  PrimExpr Rewrite(PrimExpr expr) { return this->VisitExpr(expr); }

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    te::Tensor t = Downcast<te::Tensor>(op->producer);
    if (t->op == placeholder_op_) {
      std::unordered_map<std::string, PrimExpr> name_to_arg;
      for (const auto& arg : op->indices) {
        std::string axis_name;
        if (const auto* int_imm = arg.as<IntImmNode>()) {
          ICHECK_EQ(int_imm->value, 0);
          axis_name = "IntImm";
        } else {
          axis_name = AxisBaseName(CleanName(Downcast<Var>(arg)->name_hint));
          ICHECK_EQ(name_to_arg.count(axis_name), 0);
          name_to_arg[axis_name] = arg;
        }
      }

      std::unordered_map<std::string, PrimExpr> div_factors;
      std::vector<PrimExpr> r_new_args;
      for (int i = new_names_.size() - 1; i >= 0; --i) {
        auto ori_iter_name = new_names_[i];
        auto name_it = name_to_arg.find(ori_iter_name);
        ICHECK(name_it != name_to_arg.end());
        PrimExpr ori_arg = name_it->second;

        PrimExpr mod_factor = new_shape_[i];

        PrimExpr div_factor = 1;
        if (div_factors.count(ori_iter_name)) {
          div_factor = div_factors[ori_iter_name];
        }
        div_factors[ori_iter_name] = div_factor * new_shape_[i];

        PrimExpr new_arg = indexmod(indexdiv(ori_arg, div_factor), mod_factor);

        r_new_args.push_back(new_arg);
      }

      Array<PrimExpr> new_args(std::make_move_iterator(r_new_args.rbegin()),
                               std::make_move_iterator(r_new_args.rend()));
      return ProducerLoad(op->producer, new_args);
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  const te::Operation& placeholder_op_;
  Array<PrimExpr> new_shape_;
  std::vector<std::string> new_names_;
};

std::string GetOrigLayout(std::set<std::string>* placeholder_axis_names, const te::Operation& op,
                          const te::Tensor& placeholder) {
  ReadAccessExtractor extractor;
  for (const auto& exp : op.as<te::ComputeOpNode>()->body) {
    extractor.Extract(exp);
  }

  std::ostringstream os;
  uint32_t i = 0;
  const auto& placeholder_op = placeholder->op;
  ICHECK_GT(extractor.read_access.count(placeholder_op), 0);
  for (const auto& ev : extractor.read_access[placeholder_op]) {
    for (const auto& e : ev) {
      std::string axis_name;
      if (const auto* int_imm = e.as<IntImmNode>()) {
        ICHECK_EQ(int_imm->value, 0);
        axis_name = "IntImm";
      } else {
        axis_name = AxisBaseName(CleanName(Downcast<Var>(e)->name_hint));
      }

      placeholder_axis_names->insert(axis_name);
      os << placeholder->shape[i++] << axis_name;
    }
  }

  ICHECK_EQ(placeholder_axis_names->size(), placeholder->shape.size());
  std::string orig_layout = os.str();
  os.str("");
  ::tvm::relay::AutoSchedulerLayoutRewriter::global_ori_layouts_queue.push_back(orig_layout);
  return orig_layout;
}

std::string GetNewLayout(const State& state, const int stage_id, const Stage& stage,
                         const te::Operation& op, const te::Tensor& placeholder,
                         const std::set<std::string>& placeholder_axis_names) {
  std::ostringstream os;
  Array<Iterator> stage_iters;

  auto attach_it = state->attach_map->stage_to_attach_iter.find(stage_id);
  int attach_pos = -1;
  size_t iters_before_attach = 0;
  if (attach_it != state->attach_map->stage_to_attach_iter.end()) {
    auto attach = attach_it->second;
    const auto& attach_stage = state->stages[attach.first];
    attach_pos = attach.second;
    stage_iters.insert(stage_iters.end(), attach_stage->iters.begin(),
                       attach_stage->iters.begin() + attach_pos + 1);
  }

  stage_iters.insert(stage_iters.end(), stage->iters.begin(), stage->iters.end());

  std::vector<Iterator> iters;
  for (size_t i = 0; i < stage_iters.size(); ++i) {
    const auto& iter = stage_iters[i];
    if (iter->orig_iters.empty()) {
      iters.push_back(iter);
    } else {
      for (const Iterator& ori_iter : iter->orig_iters) {
        iters.push_back(ori_iter);
      }
    }
    if (static_cast<int>(i) == attach_pos) {
      iters_before_attach = iters.size();
    }
  }

  std::vector<std::string> new_names;
  std::vector<std::string> new_axis_names;
  for (const Iterator& iter : iters) {
    std::set<std::string> ori_iter_names;
    ExtractOriginalIterators(iter->name, &ori_iter_names);
    // fused iters have been replaced with iter->orig_iters.
    // So there should be only one ori iter name extracted from iter->name.
    ICHECK_EQ(ori_iter_names.size(), 1);
    auto ori_iter_name = AxisBaseName(*ori_iter_names.begin());
    new_axis_names.push_back(ori_iter_name);
  }
  for (size_t i = 0; i < new_axis_names.size(); ++i) {
    auto iter = iters[i];
    std::string ori_iter_name;
    if (i < iters_before_attach) {
      ori_iter_name = new_axis_names[i + iters_before_attach];
    } else {
      ori_iter_name = new_axis_names[i];
    }
    if (placeholder_axis_names.count(ori_iter_name)) {
      PrimExpr extent;
      if (iter->range.defined()) {
        extent = iter->range->extent;
      } else {
        // This iter is simplified by InferBound, so it must have a length of one.
        extent = 1;
      }
      os << extent << ori_iter_name;
      new_names.push_back(ori_iter_name);
    }
  }
  std::string new_layout = os.str();
  os.str("");
  ::tvm::relay::AutoSchedulerLayoutRewriter::global_new_layouts_queue.push_back(new_layout);
  return new_layout;
}

ComputeDAG ComputeDAG::RewriteLayout(Array<Step>* transform_steps,
                                     LayoutRewriteOption layout_rewrite) const {
  CHECK(layout_rewrite != LayoutRewriteOption::NoRewrite)
      << "Call ComputeDAG::RewriteLayout with NoRewrite.";
  ComputeDAG new_dag = *this;
  ComputeDAGNode* p_dag = new_dag.CopyOnWrite();

  auto node = make_object<StateNode>();
  node->transform_steps = *transform_steps;
  node->concrete = true;
  const State& state = InferBound(State(node));

  OperationSet handled_ops;
  for (size_t stage_id = 0; stage_id < state->stages.size(); stage_id++) {
    const auto& stage = state->stages[stage_id];

    const te::Operation& op = stage->op;
    if (!op->IsInstance<te::ComputeOpNode>()) {
      continue;
    }
    const Map<String, ObjectRef>& attrs = op->attrs;
    if (attrs.count(layout_free_placeholders_key) == 0) {
      continue;
    }
    const ObjectRef& attr_value = attrs[layout_free_placeholders_key];
    for (const auto& placeholder : Downcast<Array<te::Tensor>>(attr_value)) {
      const auto& placeholder_op = placeholder->op;

      // Check whether this placeholder has already been handled
      if (handled_ops.count(placeholder_op)) {
        continue;
      }
      // Skip the op that is not direct consumer of this placeholder.
      // This is usually caused by cache read/write.
      bool direct_consumer = false;
      for (auto& t : op->InputTensors()) {
        if (t->op == placeholder_op) {
          direct_consumer = true;
          break;
        }
      }
      if (!direct_consumer) {
        continue;
      }
      handled_ops.insert(placeholder_op);

      // Process original layout
      std::set<std::string> placeholder_axis_names;
      std::string origin_layout = GetOrigLayout(&placeholder_axis_names, op, placeholder);
      Array<PrimExpr> origin_shape;
      std::vector<std::string> origin_axes;
      ParseKernelLayout(origin_layout, &origin_shape, &origin_axes);

      // Process new layout
      std::string new_layout =
          GetNewLayout(state, stage_id, stage, op, placeholder, placeholder_axis_names);
      Array<PrimExpr> new_shape;
      std::vector<std::string> new_axes;
      ParseKernelLayout(new_layout, &new_shape, &new_axes);

      // Process op updates
      te::Operation new_op_to_update;
      if (layout_rewrite == LayoutRewriteOption::RewriteForPreTransformed) {
        // Create new placeholder
        new_op_to_update = te::PlaceholderOp(placeholder_op->name, new_shape,
                                             placeholder_op.as<te::PlaceholderOpNode>()->dtype);
      } else if (layout_rewrite == LayoutRewriteOption::InsertTransformStage) {
        // Process index strides
        std::unordered_map<std::string, PrimExpr> axes_stride;
        for (const auto& i : origin_axes) {
          axes_stride[i] = Integer(1);
        }
        Array<PrimExpr> new_stride(new_shape.size(), PrimExpr());
        PrimExpr temp = Integer(1);
        for (int i = new_shape.size() - 1; i >= 0; i--) {
          new_stride.Set(i, axes_stride[new_axes[i]]);
          axes_stride[new_axes[i]] *= new_shape[i];
        }

        // Add an extra layout transform stage
        const auto& layout_transform_tensor = te::compute(
            new_shape,
            [&new_stride, &placeholder_op, &origin_shape, &new_shape, &origin_axes,
             &new_axes](const tvm::runtime::Array<tvm::tir::Var>& indices) -> tvm::PrimExpr {
              Array<PrimExpr> access_indices;
              for (size_t indice_index = 0; indice_index < origin_shape.size(); indice_index++) {
                PrimExpr temp = Integer(0);
                for (size_t i = 0; i < new_shape.size(); i++) {
                  if (origin_axes[indice_index].compare(new_axes[i]) == 0) {
                    temp += indices[i] * new_stride[i];
                  }
                }
                access_indices.push_back(temp);
              }
              return placeholder_op.output(0)(access_indices);
            },
            "auto_scheduler_layout_transform");
        new_op_to_update = layout_transform_tensor->op;

        // Update the transform steps
        for (size_t i = 0; i < transform_steps->size(); i++) {
          Step step = (*transform_steps)[i];
          if (step->stage_id >= static_cast<int>(stage_id)) {
            step.CopyOnWrite()->stage_id++;
          }
          if (step->IsInstance<ComputeAtStepNode>()) {
            auto compute_at_step = tvm::Downcast<ComputeAtStep>(step);
            if (compute_at_step->target_stage_id >= static_cast<int>(stage_id)) {
              dynamic_cast<ComputeAtStepNode*>(compute_at_step.CopyOnWrite())->target_stage_id++;
            }
            transform_steps->Set(i, std::move(compute_at_step));
          } else {
            transform_steps->Set(i, std::move(step));
          }
        }

        // Add schedule for the new added transform stage
        Array<Integer> to_fuse;

        if (new_shape.size() >= 5) {
          to_fuse.push_back(0);
          to_fuse.push_back(1);
          to_fuse.push_back(2);
          transform_steps->push_back(FuseStep(stage_id, to_fuse));
        } else if (new_shape.size() >= 3) {
          to_fuse.push_back(0);
          to_fuse.push_back(1);
          transform_steps->push_back(FuseStep(stage_id, to_fuse));
        }
        transform_steps->push_back(AnnotationStep(stage_id, 0, IteratorAnnotation::kParallel));
      }

      te::Operation new_compute_op, original_compute_op;
      Array<PrimExpr> new_body;
      IndexRewriter index_rewriter(placeholder_op, new_layout);
      for (const auto& op : p_dag->ops) {
        if (auto* pop = op.as<te::ComputeOpNode>()) {
          bool need_update = false;
          for (auto& t : op->InputTensors()) {
            if (t->op == placeholder_op) {
              need_update = true;
              break;
            }
          }
          if (need_update) {
            for (const auto& body : pop->body) {
              new_body.push_back(index_rewriter.Rewrite(body));
            }
            original_compute_op = op;
            CHECK(!new_compute_op.defined());
            auto new_attrs = pop->attrs;
            new_attrs.Set("ori_placeholder_layout", tvm::String(origin_layout));
            new_attrs.Set("new_placeholder_layout", tvm::String(new_layout));
            new_compute_op = te::ComputeOp(pop->name, pop->tag, new_attrs, pop->axis, new_body);
          }
        }
      }

      // construct the map from original_op to new_op
      std::unordered_map<te::Operation, te::Operation> updated_ops;

      Array<te::Operation> original_ops = p_dag->ops;
      p_dag->ops.clear();
      for (size_t i = 0; i < original_ops.size(); ++i) {
        const auto& original_op = original_ops[i];
        if (original_op == placeholder_op) {
          if (layout_rewrite == LayoutRewriteOption::InsertTransformStage) {
            p_dag->ops.push_back(placeholder_op);
          }
          p_dag->ops.push_back(new_op_to_update);
          updated_ops[placeholder_op] = new_op_to_update;
        } else if (original_op == original_compute_op) {
          p_dag->ops.push_back(new_compute_op);
          updated_ops[original_compute_op] = new_compute_op;
        } else {
          p_dag->ops.push_back(original_op);
        }
      }

      ArrayNode* pops = p_dag->ops.CopyOnWrite();
      // Because ops is sorted in topo-order, only do one pass linear scan here.
      for (size_t i = 0; i < pops->size(); ++i) {
        const auto& original_op = Downcast<te::Operation>(pops->at(i));
        if (auto* pop = original_op.as<te::ComputeOpNode>()) {
          if (original_op == new_op_to_update) {
            continue;
          }
          auto inputs = pop->InputTensors();
          std::unordered_map<te::Tensor, te::Tensor> rmap;
          for (auto input : inputs) {
            auto it = updated_ops.find(input->op);
            te::Operation new_op;
            while (it != updated_ops.end()) {
              new_op = it->second;
              it = updated_ops.find(new_op);
            }
            if (new_op.defined()) {
              int index = input->value_index;
              rmap[input] = new_op.output(index);
            }
          }
          if (!rmap.empty()) {
            te::Operation new_op = pop->ReplaceInputs(original_op, rmap);
            updated_ops[original_op] = new_op;
            pops->SetItem(i, new_op);
          }
        }
      }

      Array<te::Tensor> old_tensors = p_dag->tensors;
      ArrayNode* p_tensors = p_dag->tensors.CopyOnWrite();
      for (size_t i = 0; i < old_tensors.size(); ++i) {
        const auto& old_tensor = old_tensors[i];
        if (layout_rewrite != LayoutRewriteOption::RewriteForPreTransformed &&
            old_tensor->op->IsInstance<te::PlaceholderOpNode>()) {
          continue;
        }
        auto it = updated_ops.find(old_tensor->op);
        te::Operation new_op;
        while (it != updated_ops.end()) {
          new_op = it->second;
          it = updated_ops.find(new_op);
        }
        if (new_op.defined()) {
          auto index = old_tensor->value_index;
          p_tensors->SetItem(i, new_op.output(index));
        }
      }
    }  // end for placeholder
  }    // end for stage
  p_dag->access_analyzer = AccessAnalyzer(p_dag->tensors);

  Array<te::Operation> out_ops;
  for (const auto& op : p_dag->access_analyzer->ops_topo_order) {
    if (p_dag->access_analyzer.IsOutput(op)) {
      out_ops.push_back(op);
    }
  }

  p_dag->ops.clear();
  te::Schedule sch = te::create_schedule(out_ops);
  for (auto stage : sch->stages) {
    p_dag->ops.push_back(stage->op);
  }
  p_dag->flop_ct = FlopEstimator().EstimateFlop(p_dag->ops);
  p_dag->init_state = State(p_dag->ops);

  return new_dag;
}

// Return whether a DAG has placeholders that are marked as "layout free".
bool HasLayoutFreeTensors(const ComputeDAG& dag) {
  for (const auto& op : dag->ops) {
    if (!op->IsInstance<te::ComputeOpNode>()) {
      continue;
    }
    if (op->attrs.count(ComputeDAG::layout_free_placeholders_key)) {
      return true;
    }
  }

  return false;
}

std::pair<te::Schedule, Array<te::Tensor>> ComputeDAG::ApplySteps(
    const Array<Step>& transform_steps, Array<te::Stage>* stages, StageToAxesMap* stage_to_axes,
    LayoutRewriteOption layout_rewrite) const {
  if (layout_rewrite != LayoutRewriteOption::NoRewrite && HasLayoutFreeTensors(*this) &&
      !transform_steps.empty()) {
    Array<Step> steps = transform_steps;
    const auto& dag = RewriteLayout(&steps, layout_rewrite);
    return dag.ApplySteps(steps);
  }

  // Temporal object to be used if the input pointer is nullptr
  Array<te::Stage> temp_stages;
  StageToAxesMap temp_stage_to_axes;
  if (stages == nullptr) {
    stages = &temp_stages;
  }
  if (stage_to_axes == nullptr) {
    stage_to_axes = &temp_stage_to_axes;
  }
  Array<te::Operation> out_ops;
  for (const auto& op : operator->()->ops) {
    if (operator->()->access_analyzer.IsOutput(op)) {
      out_ops.push_back(op);
    }
  }

  // Create the initial schedule
  te::Schedule schedule = te::create_schedule(out_ops);

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
  Array<te::Operation> out_ops;
  for (const auto& op : operator->()->ops) {
    if (operator->()->access_analyzer.IsOutput(op)) {
      out_ops.push_back(op);
    }
  }
  // Create the initial schedule
  te::Schedule schedule = te::create_schedule(out_ops);

  // init axes
  for (const auto& x : operator->()->ops) {
    const te::Stage& stage = schedule[x];
    stages.push_back(stage);
    UpdateStageToAxesMap(stage, &stage_to_axes);
  }

  std::stringstream ss;
  for (const auto& stage : stages) {
    if (stage->op->IsInstance<te::ComputeOpNode>()) {
      auto op_name = CleanName(stage->op->name);

      for (size_t i = 0; i < stage->leaf_iter_vars.size(); ++i) {
        ss << CleanName(stage->leaf_iter_vars[i]->var->name_hint, op_name);
        if (i != stage->leaf_iter_vars.size() - 1) {
          ss << ", ";
        }
      }
      ss << " = "
         << "tuple(" << op_name << ".op.axis)"
         << " + "
         << "tuple(" << op_name << ".op.reduce_axis)\n";
    }
  }
  // Call each step's PrintAsPythonAPI method
  for (const auto& step : transform_steps) {
    ss << StepPrintAsPythonAPI(step, &stages, &stage_to_axes, &schedule, transform_steps);
  }

  return ss.str();
}

String ComputeDAG::PrintDAG(bool simple_mode) const {
  std::stringstream ss;

  for (const auto& op : operator->()->ops) {
    if (op->IsInstance<te::PlaceholderOpNode>()) {
      ss << op->name << " = PLACEHOLDER ";
      if (!simple_mode) {
        ss << op.output(0)->shape;
      }
      ss << "\n";
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
        if (auto p_reduce = pop->body[k].as<ReduceNode>()) {
          ICHECK_LT(k, p_reduce->combiner->result.size());
          PrimExpr combiner = p_reduce->combiner->result[k];
          if (combiner->IsInstance<AddNode>()) {
            ss << " += " << AsLegacyRepr(p_reduce->source[0]) << "\n";
          } else if (combiner->IsInstance<MaxNode>()) {
            ss << " max= " << AsLegacyRepr(p_reduce->source[0]) << "\n";
          } else if (combiner->IsInstance<MinNode>()) {
            ss << " min= " << AsLegacyRepr(p_reduce->source[0]) << "\n";
          } else if (combiner->IsInstance<SelectNode>()) {
            const auto& select = combiner.as<SelectNode>();
            ss << " select(" << AsLegacyRepr(select->condition)  //
               << ", " << AsLegacyRepr(select->true_value)       //
               << ", " << AsLegacyRepr(select->false_value)      //
               << ")= (" << AsLegacyRepr(p_reduce->source[0])    //
               << ',' << AsLegacyRepr(p_reduce->source[1])       //
               << ")\n";
          } else {
            ss << "reduce" << AsLegacyRepr(combiner) << "\n";
          }
        } else {
          auto call = pop->body[k].as<CallNode>();
          if (simple_mode && call) {
            ss << " = " << AsLegacyRepr(call->op) << "\n";
          } else {
            ss << " = " << AsLegacyRepr(pop->body[k]) << "\n";
          }
        }
      }
    } else {
      LOG(FATAL) << "Invalid op";
    }
  }
  return String(ss.str());
}

State ComputeDAG::InferBound(const State& state) const {
  ICHECK(state->concrete) << "Only concrete state can be processed to get bound info.";

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
  // Replay steps to tvm::Schedule
  auto [sch, tensors] = ApplySteps(pstate->transform_steps, &stages, &stage_to_axes);
  (void)tensors;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
  sch = sch.normalize_for_feature_extraction();
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
        new_iters.push_back(Iterator(iter->name, (*find_res).second, iter->iter_kind,
                                     iter->annotation, &iter->orig_iters));
      } else {
        LOG(FATAL) << "Infer bound fails";
      }
    }

    pstate->stages.Set(
        i, Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));
  }

  return ret_state;
}

Array<State> ComputeDAG::InferBound(const Array<State>& states) const {
  Array<State> out_states(states.size(), State());

  support::parallel_for(0, states.size(), [this, &states, &out_states](int i) {
    try {
      out_states.Set(i, (states[i].defined()) ? this->InferBound(states[i]) : states[i]);
    } catch (Error& e) {
      LOG(WARNING) << "InferBound fails on the state:\n"
                   << states[i] << "\n"
                   << "with: " << e.what() << std::endl;
    }
  });

  return out_states;
}

ComputeDAG ComputeDAG::ReplayAndGetDAG(const Array<Step>& transform_steps) const {
  auto [sch, old_tensors] = ApplySteps(transform_steps);
  (void)old_tensors;  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81767
  return ComputeDAG(sch);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AccessAnalyzerNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const AccessAnalyzerNode*>(ref.get());
      for (const auto& op : node->ops_topo_order) {
        p->stream << op << std::endl;
        p->stream << "is_simple_access:\t" << node->is_simple_access.at(op) << "\t\t";
        p->stream << "needs_multi_level_tiling:\t" << node->needs_multi_level_tiling.at(op)
                  << std::endl;
        p->stream << "is_strictly_inlinable:\t" << node->is_strictly_inlineable.at(op) << "\t";
        p->stream << "is_output:\t" << node->is_output.at(op) << std::endl;
        p->stream << "Read from:\t";
        for (const auto& pair : node->read_from.at(op)) {
          for (const auto& index : pair.second) {
            p->stream << pair.first->name << Array<PrimExpr>(index) << ", ";
          }
        }
        p->stream << std::endl;
        p->stream << "Read by:\t";
        for (const auto& pair : node->read_by.at(op)) {
          for (const auto& index : pair.second) {
            p->stream << pair.first->name << Array<PrimExpr>(index) << ", ";
          }
        }
        p->stream << std::endl;
        p->stream << Chars('=', 50) << std::endl;
      }

      AccessAnalyzer ana = GetRef<AccessAnalyzer>(node);
      p->stream << "ElementwiseMatch: \n";
      for (size_t i = 0; i < node->ops_topo_order.size(); ++i) {
        for (size_t j = 0; j < node->ops_topo_order.size(); ++j) {
          if (i == j) {
            continue;
          }
          if (ana.ElementWiseMatch(node->ops_topo_order[i], node->ops_topo_order[j])) {
            p->stream << node->ops_topo_order[i]->name << " -> " << node->ops_topo_order[j]->name
                      << std::endl;
          }
        }
      }
      p->stream << Chars('=', 50) << std::endl;

      p->stream << "NumCommonOuterIterators: \n";
      for (const auto& src_pair : node->num_common_outer_iterators) {
        for (const auto& dst_pair : src_pair.second) {
          p->stream << src_pair.first->name << " " << dst_pair.first->name << " " << dst_pair.second
                    << std::endl;
        }
      }
      p->stream << Chars('=', 50) << std::endl;
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ComputeDAGNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ComputeDAGNode*>(ref.get());
      auto dag = GetRef<ComputeDAG>(node);
      auto dag_str = dag.PrintDAG();
      p->stream << dag_str;
    });

Array<PrimExpr> GetShapeFromRewrittenLayout(String rewritten_layout, Array<String> axis_names) {
  Array<PrimExpr> shape;
  std::vector<std::string> extracted_names;
  topi::parse_auto_scheduler_layout(rewritten_layout, &shape, &extracted_names);

  Array<PrimExpr> ret(axis_names.size(), 1);

  size_t ct = 0;
  for (size_t i = 0; i < axis_names.size(); ++i) {
    for (size_t j = 0; j < extracted_names.size(); ++j) {
      if (axis_names[i] == extracted_names[j]) {
        ret.Set(i, ret[i] * shape[j]);
        ct++;
      }
    }
  }

  CHECK_EQ(ct, extracted_names.size()) << "The number or names of axes do not match";

  return ret;
}

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAG")
    .set_body_typed([](Optional<Array<te::Tensor>> tensors, Optional<te::Schedule> sch) {
      if (sch) {
        return ComputeDAG(sch.value());
      }
      ICHECK(tensors) << "Both tensors and schedule are null";
      return ComputeDAG(tensors.value());
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGApplyStepsFromState")
    .set_body_typed([](const ComputeDAG& dag, const State& state, int layout_rewrite) {
      auto [sch, return_tensors] = dag.ApplySteps(state->transform_steps, nullptr, nullptr,
                                                  static_cast<LayoutRewriteOption>(layout_rewrite));
      return Array<ObjectRef>{sch, return_tensors};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGPrintPythonCodeFromState")
    .set_body_typed([](const ComputeDAG& dag, const State& state) {
      return dag.PrintStepsAsPython(state->transform_steps);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGPrintDAG")
    .set_body_typed([](const ComputeDAG& dag, bool simple_mode) {
      return dag.PrintDAG(simple_mode);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGInferBoundFromState")
    .set_body_typed([](const ComputeDAG& dag, const State& state) {
      return dag.InferBound(state);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.ComputeDAGRewriteLayoutFromState")
    .set_body_typed([](const ComputeDAG& dag, const State& state) {
      Array<Step>* transform_steps = const_cast<Array<Step>*>(&state->transform_steps);
      return dag.RewriteLayout(transform_steps, LayoutRewriteOption::RewriteForPreTransformed);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.RewriteIndexForNewLayout")
    .set_body_typed([](const te::Operation& placeholder_op, const std::string& new_layout,
                       const PrimExpr& body) {
      IndexRewriter index_rewriter(placeholder_op, new_layout);
      return index_rewriter.Rewrite(body);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.RewriteTensorShape")
    .set_body_typed([](te::Tensor tensor, Array<PrimExpr> new_shape) -> void {
      ICHECK(tensor->op->IsInstance<te::PlaceholderOpNode>());
      te::PlaceholderOpNode* op =
          const_cast<te::PlaceholderOpNode*>(tensor->op.as<te::PlaceholderOpNode>());
      te::TensorNode* t = const_cast<te::TensorNode*>(tensor.get());
      op->shape = new_shape;
      t->shape = new_shape;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GetShapeFromRewrittenLayout")
    .set_body_typed(GetShapeFromRewrittenLayout);

}  // namespace auto_scheduler
}  // namespace tvm
