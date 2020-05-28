/*!
 *  Copyright (c) 2020 by Contributors
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
#include "loop_state.h"
#include "utils.h"
// #include "../relay/pass/kernel_layout_transform.h"

namespace tvm {
namespace ansor {

using namespace tvm::tir;

TVM_REGISTER_NODE_TYPE(ComputeDAGNode);

template<class T>
using OperationMap = AccessAnalyzerNode::OperationMap<T>;

using OperationSet = std::unordered_set<te::Operation, ObjectHash, ObjectEqual>;

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

// Extract all tensor accesses in an expr
class TensorAccessExtractor : public StmtExprVisitor {
 public:
  void Extract(PrimExpr expr) {
    this->VisitExpr(expr);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->call_type == CallNode::CallType::Halide) {
      buf_accesses[Downcast<te::Operation>(op->func)].emplace_back(
          op->args.begin(), op->args.end());
    }
    if (op->name == tir::intrinsic::tvm_if_then_else) {
      has_branch = true;
    }
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

  OperationMap<std::vector<std::vector<PrimExpr> > > buf_accesses;
  bool has_branch{false};
};

// Returns whether the expr equals to the var with a const shift
bool IsConstShiftEqual(const Var& var, const PrimExpr& expr) {
  if (auto pv = expr.as<VarNode>()) {
    return pv == var.get();
  } else if (auto padd = expr.as<AddNode>()) {
    return ((padd->a.get() == var.get() && padd->b->IsInstance<IntImmNode>()) ||
            (padd->b.get() == var.get() && padd->a->IsInstance<IntImmNode>()));
  } else if (auto psub = expr.as<SubNode>()) {
    return ((psub->a.get() == var.get() && psub->b->IsInstance<IntImmNode>()) ||
            (psub->b.get() == var.get() && psub->a->IsInstance<IntImmNode>()));
  } else {
    return false;
  }
}

// Return whether the access is injective
bool IsInjective(const te::Operation& op,  const std::vector<PrimExpr>& index,
                 bool* axis_missing, bool* axis_duplicated, bool* same_order) {
  auto cop = op.as<te::ComputeOpNode>();
  if (cop == nullptr) { return false; }

  std::vector<int> index_to_var_idx;
  std::vector<int> var_idx_ct(cop->axis.size(), 0);

  for (const auto& expr : index) {
    if (!is_const(expr)) {
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

  *axis_missing = false;    // Some axes are missing
  *axis_duplicated = false;  // Some axes appear more than once
  *same_order = true;       // The axis order is the same as op->axis
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
static void GatherVars(const PrimExpr& expr,
                       std::unordered_set<const VarNode*>* vars) {
  PostOrderVisit(expr, [&vars](const ObjectRef &node) {
    if (const VarNode* op = node.as<VarNode>()) {
      vars->insert(op);
    }
  });
}

// Check whether an expr has expensive operations (e.g. exp)
static bool HasExpensiveOp(const PrimExpr& expr) {
  bool found = false;
  PostOrderVisit(expr, [&found](const ObjectRef &node) {
    if (const CallNode* op = node.as<CallNode>()) {
      if (op->call_type == CallNode::CallType::PureIntrinsic &&
          op->name == "exp") {
        found = true;
      }
    }
  });
  return found;
}

AccessAnalyzer AccessAnalyzerNode::make(const Array<te::Tensor>& tensors) {
  auto node = make_object<AccessAnalyzerNode>();
  OperationMap<bool> has_branch;

  // get all ops
  TopoSortOps(tensors, &node->ops_topo_order);

  // build read & write access map
  for (const auto& op : node->ops_topo_order) {
    if (op->IsInstance<te::PlaceholderOpNode>()) {
      node->read_from[op] =
          OperationMap<std::vector<std::vector<PrimExpr> > >();
    } else if (auto cop = op.as<te::ComputeOpNode>()) {
      TensorAccessExtractor extractor;
      for (const auto& exp : cop->body) {
        extractor.Extract(exp);
      }

      for (const auto& iter : extractor.buf_accesses) {
        std::vector<std::vector<PrimExpr> >& accesses =
            node->read_by[iter.first][op];
        accesses.insert(accesses.begin(), iter.second.begin(),
                        iter.second.end());
      }

      node->read_from[op] = std::move(extractor.buf_accesses);
      has_branch[op] = extractor.has_branch;
    } else {
      LOG(FATAL) << "Invalid op: " << op;
    }
  }

  // do some static analysis
  for (const auto& op : node->ops_topo_order) {
    if (op->IsInstance<te::PlaceholderOpNode>()) {
      node->is_injective[op] = true;
      node->needs_multi_level_tiling[op] = false;
      node->is_strict_inlineable[op] = false;
      node->is_output[op] = false;
    } else if (auto pop = op.as<te::ComputeOpNode>()) {
      // check whether is element-wise and strict-inlineable
      // (see definition in compute_dag.h)
      bool is_injective = true;
      bool is_strict_inlineable = true;

      bool axis_missing, axis_duplicated, same_order;
      for (const auto& pair : node->read_from[op]) {
        const std::vector<std::vector<PrimExpr> >& access = pair.second;
        for (const auto& index : access) {
          if (!IsInjective(op, index, &axis_missing, &axis_duplicated,
                           &same_order)) {
            is_injective = false;
            is_strict_inlineable = false;
            break;
          }
          if (!same_order || axis_duplicated) {
            // do not strictly inline transpose
            is_strict_inlineable = false;
          }
        }
        if (!is_injective) { break; }
      }
      if (has_branch[op]) {
        is_strict_inlineable = false;
      }

      // don't strictly inline expensive op (e.g. exp)
      bool has_expensive_op = false;
      for (const auto& expr : pop->body) {
        has_expensive_op |= HasExpensiveOp(expr);
      }

      node->is_injective[op] = is_injective;
      node->is_strict_inlineable[op] = is_strict_inlineable &&
                                       !has_expensive_op;

      // check whether the op needs multi-level tiling
      // (see definition in compute_dag.h)
      bool needs_multi_level_tiling = false;
      int n_missing = 0;

      for (const auto& pair : node->read_from[op]) {
        const std::vector<std::vector<PrimExpr> > &access = pair.second;
        std::unordered_set<const VarNode*> vars;
        for (const std::vector<PrimExpr> &indices : access) {
          for (const PrimExpr& expr : indices) {
            GatherVars(expr, &vars);
          }
        }
        bool missing = false;
        for (const auto& axis : pop->axis) {
          if (GetIntImm(axis->dom->extent) > 1 &&
              vars.count(axis->var.get()) == 0) {
            missing = true;
          }
        }
        if (missing) {
          n_missing++;
        }

        if (n_missing >= 2 || (n_missing >= 1 && !pop->reduce_axis.empty())) {
          needs_multi_level_tiling = true;
          break;
        }
      }

      node->needs_multi_level_tiling[op] = needs_multi_level_tiling;

      // check whether is output
      node->is_output[op] = node->read_by[op].empty();
    } else {
      LOG(FATAL) << "Invalid op" << op;
    }
  }

  return AccessAnalyzer(node);
}

bool AccessAnalyzer::NeedsMultiLevelTiling(const te::Operation &op) const {
  return operator->()->needs_multi_level_tiling.at(op);
}

bool AccessAnalyzer::IsOutput(const te::Operation& op) const {
  return operator->()->is_output.at(op);
}

bool AccessAnalyzer::IsInjective(const te::Operation& op) const {
  return operator->()->is_injective.at(op);
}

bool AccessAnalyzer::IsStrictInlineable(const te::Operation &op) const {
  return operator->()->is_strict_inlineable.at(op);
}

void AccessAnalyzer::GetProducers(const State& state, const te::Operation& op,
                                  OperationSet* producers) const {
  producers->clear();
  for (const auto& iter : operator->()->read_from.at(op)) {
    producers->insert(iter.first);
  }
}

void AccessAnalyzer::GetConsumers(const State& state, const te::Operation& op,
                                  OperationSet* consumers) const {
  OperationSet inlined_ops;

  for (const auto& stage : state->stages) {
    if (stage->compute_at == kInlined) {
      inlined_ops.insert(stage->op);
    }
  }
  std::function<void(const te::Operation& op)> collect;

  collect = [this, &collect, &inlined_ops, &consumers](const te::Operation& op) {
    for (const auto& iter : operator->()->read_by.at(op)) {
      if (inlined_ops.count(iter.first)) {
        collect(iter.first);
      } else {
        consumers->insert(iter.first);
      }
    }
  };

  consumers->clear();
  collect(op);
}

bool IntArrayEqual(const Array<PrimExpr>& arr1, const Array<PrimExpr>& arr2) {
  if (arr1.size() != arr2.size()) {
    return false;
  }

  for (size_t i = 0; i < arr1.size(); ++i) {
    auto int1 = arr1[i].as<IntImmNode>();
    auto int2 = arr2[i].as<IntImmNode>();
    CHECK(int1 != nullptr);
    CHECK(int2 != nullptr);
    if (int1->value != int2->value) {
      return false;
    }
  }
  return true;
}

bool AccessAnalyzer::ElementWiseMatch(const te::Operation& op,
                                      const te::Operation& target_op) const {
  te::Operation cur_op = op;
  while (cur_op != target_op) {
    const AccessAnalyzerNode::OperationMap<std::vector<std::vector<PrimExpr> > >& map =
        operator->()->read_by.at(cur_op);

    if (map.size() != 1) {
      return false;
    }
    te::Operation next_op = map.begin()->first;

    // Check condition 1: has the same output size
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

    // Check condition 2: read is elementwise
    const std::vector<std::vector<PrimExpr> > reads = map.begin()->second;
    bool is_injective, axis_missing, axis_duplicated, same_order;
    for (const auto& read : reads) {
      is_injective = ::tvm::ansor::IsInjective(
          next_op, read, &axis_missing, &axis_duplicated, &same_order);
      if (!is_injective || axis_missing || axis_duplicated || !same_order) {
        return false;
      }
    }

    cur_op = std::move(next_op);
  }
  return true;
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
//   double VisitExpr_(const UIntImm* op) final { return 0.0; }

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
    if (op->call_type == CallNode::CallType::Halide) {
      // ignore flops in index expressions
      return 0.0;
    }

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

State ComputeDAG::GetInitState() const {
  return Downcast<State>(operator->()->init_state);
}

ComputeDAG ComputeDAGNode::make(Array<te::Tensor> tensors) {
  auto node = make_object<ComputeDAGNode>();
  FlopEstimator estimator;

  node->tensors = std::move(tensors);
  node->access_analyzer = AccessAnalyzerNode::make(node->tensors);
  node->ops = Array<te::Operation>(node->access_analyzer->ops_topo_order);
  node->flop_ct = estimator.EstimateFlop(node->ops);
  node->init_state = StateNode::make(node->ops);

  return ComputeDAG(node);
}

ComputeDAG ComputeDAGNode::make_by_workload_key(const std::string& workload_key) {
  Array<te::Tensor> tens;
  // Call python function to decode the workload_key and get the I/O tensors
  if (const auto* f = runtime::Registry::Get("ansor.workload_key_to_tensors")) {
    tens = (*f)(workload_key);
  } else {
    LOG(FATAL) << "ansor.workload_key_to_tensors is not registered";
  }
  return ComputeDAGNode::make(std::move(tens));
}

void ComputeDAGNode::VisitAttrs(tvm::AttrVisitor* v) {
  v->Visit("tensors", &tensors);
  v->Visit("ops", &ops);
  v->Visit("flop_ct", &flop_ct);
  v->Visit("access_analyzer", &access_analyzer);
  State s = Downcast<State>(init_state);
  v->Visit("init_state", &s);
}

// Implemented in multi_stage_policy.cc
// Extract primitive iterators from a nested fused or splitted iterator's name
extern void ExtractOriginalIterators(const std::string& name, std::set<std::string>* rets);

// Implemented in loop_state.cc
extern std::string CleanName(const std::string& str);

std::string BaseName(const std::string& str) {
  return str.substr(0, str.rfind("_"));
}

// class IndexRewriter : public ExprMutator {
//  public:
//   IndexRewriter(const OperationMap<std::vector<std::string> >& placeholder_new_names,
//                 const OperationMap<Array<PrimExpr> >& placeholder_new_shapes):
//                 placeholder_new_names_(placeholder_new_names),
//                 placeholder_new_shapes_(placeholder_new_shapes) {}

//   Expr Mutate_(const Call* op, const Expr& e) {
//     Expr op_ = IRMutator::Mutate_(op, e);

//     const Call* call = op_.as<Call>();

//     if (call->call_type == Call::CallType::Halide) {
//       Tensor t = Downcast<Operation>(call->func).output(call->value_index);
//       auto it = placeholder_new_names_.find(t->op);
//       if (it != placeholder_new_names_.end()) {
//         const std::vector<std::string>& new_names = it->second;
//         const Array<Expr>& new_shape = placeholder_new_shapes_.at(t->op);
//         std::unordered_map<std::string, Expr> name_to_arg;
//         for (const auto& arg : call->args) {
//           std::string axis_name;
//           if (const auto* pimm = arg.as<IntImm>()) {
//               CHECK_EQ(pimm->value, 0);
//            axis_name = "IntImm";
//           } else {
//             axis_name = BaseName(CleanName(Downcast<Var>(arg)->name_hint));
//             CHECK_EQ(name_to_arg.count(axis_name), 0);
//             name_to_arg[axis_name] = arg;
//           }
//         }

//         std::unordered_map<std::string, Expr> div_factors;
//         std::vector<Expr> r_new_args;
//         for (int i = new_names.size() - 1; i >= 0; --i) {
//           auto ori_iter_name = new_names[i];
//           auto name_it = name_to_arg.find(ori_iter_name);
//           CHECK(name_it != name_to_arg.end());
//           Expr ori_arg = name_it->second;

//           Expr mod_factor = new_shape[i];

//           Expr div_factor = 1;
//           if (div_factors.count(ori_iter_name)) {
//             div_factor = div_factors[ori_iter_name];
//           }
//           div_factors[ori_iter_name] = div_factor * new_shape[i];

//           Expr new_arg = indexmod(indexdiv(ori_arg, div_factor), mod_factor);

//           r_new_args.push_back(new_arg);
//         }

//         Array<Expr> new_args(std::make_move_iterator(r_new_args.rbegin()),
//                              std::make_move_iterator(r_new_args.rend()));

//         return Call::make(call->type, call->name, new_args, call->call_type,
//                 call->func, call->value_index);
//       }
//     }
//     return op_;
//   }

//  private:
//   const OperationMap<std::vector<std::string> >& placeholder_new_names_;
//   const OperationMap<Array<PrimExpr> >& placeholder_new_shapes_;
// };

// // TODO(minminsun): spill out new functions
// void ComputeDAG::RewriteLayout(
//     const std::vector<Step> &transform_steps, LayoutRewriteLevel layout_rewrite_level) const {
//   ComputeDAGNode* pdag = const_cast<ComputeDAG*>(this)->CopyOnWrite();
//   const State& state = ReplayAndInferBound(transform_steps);

//   OperationMap<std::vector<std::string> > placeholder_new_names;
//   OperationMap<Array<PrimExpr> > placeholder_new_shapes;
//   int stage_id = -1;
//   for (const auto& stage : state->stages) {
//     stage_id += 1;
//     const Operation& op = stage->op;
//     if (op->IsInstance<ComputeOpNode>()) {
//       const Map<std::string, ObjectRef>& attrs = op->attrs;
//       if (attrs.count(_layout_free_placeholders_key)) {
//         const ObjectRef& attr_value = attrs[_layout_free_placeholders_key];
//         Array<Tensor> placeholders = Downcast<Array<Tensor>>(attr_value);
//         for (auto& placeholder : placeholders) {
//           const auto placeholder_op = placeholder->op;

//           // Check whether this placeholder has already been handled
//           if (placeholder_new_names.count(placeholder_op)) {
//             continue;
//           }

//           // skip the op that is not direct consumer of this placeholder,
//           // mostly due to cache read/write.
//           bool direct_consumer = false;
//           for (auto& t : op->InputTensors()) {
//             if (t->op == placeholder_op) {
//               direct_consumer = true;
//               break;
//             }
//           }
//           if (!direct_consumer) {
//             continue;
//           }

//           std::set<std::string> placeholder_axis_names;
//           TensorAccessExtractor extractor;
//           for (const auto& exp : op.as<ComputeOpNode>()->body) {
//             extractor.Extract(exp);
//           }
//           bool rewrite_placeholder = (layout_rewrite_level == kPlaceholderRewrite ||
//                                       layout_rewrite_level == kBothRewrite);
//           bool rewrite_body = (layout_rewrite_level == kComputeRewrite ||
//                                layout_rewrite_level == kBothRewrite);
//           std::ostringstream os;

//           uint i = 0;
//           if (extractor.buf_accesses.count(placeholder_op)) {
//             for (const auto& ev : extractor.buf_accesses[placeholder_op]) {
//               for (const auto& e : ev) {
//                 // TODO(minminsun): check whether the extents match the shape of placeholder
//                 std::string axis_name;
//                 if (const auto* pimm = e.as<IntImm>()) {
//                   CHECK_EQ(pimm->value, 0);
//                   // CHECK_EQ(placeholder->shape[i].as<IntImm>()->value, 1);
//                   axis_name = "IntImm";
//                 } else {
//                   axis_name = BaseName(CleanName(Downcast<Var>(e)->name_hint));
//                 }

//                 placeholder_axis_names.insert(axis_name);
//                 if (rewrite_placeholder) {
//                   os << placeholder->shape[i++] << axis_name;
//                 }
//               }
//             }

//             if (rewrite_placeholder) {
//               CHECK_EQ(placeholder_axis_names.size(), placeholder->shape.size());
//               std::string ori_layout = os.str();
//               os.str("");
//               ::tvm::relay::KernelLayoutVisitor::global_ori_layouts_queue.push_back(ori_layout);
//             }
//           }

//           std::vector<Iterator> stage_iters;

//           auto attach_it = state->attach_map->stage_to_attach_iter.find(stage_id);
//           int attach_pos = -1;
//           size_t iters_before_attach = 0;
//           if (attach_it != state->attach_map->stage_to_attach_iter.end()) {
//             auto attach = attach_it->second;
//             const auto& attach_stage = state->stages[attach.first];
//             attach_pos = attach.second;
//             stage_iters.insert(stage_iters.end(),
//                                attach_stage->iters.begin(),
//                                attach_stage->iters.begin() + attach_pos + 1);
//           }

//           stage_iters.insert(stage_iters.end(), stage->iters.begin(), stage->iters.end());

//           std::vector<Iterator> iters;
//           for (size_t i = 0; i < stage_iters.size(); ++i) {
//             const auto& iter = stage_iters[i];
//             if (iter->ori_iters.empty()) {
//               iters.push_back(iter);
//             } else {
//               for (const Iterator& ori_iter : iter->ori_iters) {
//                 iters.push_back(ori_iter);
//               }
//             }
//             if (static_cast<int>(i) == attach_pos) {
//               iters_before_attach = iters.size();
//             }
//           }

//           std::vector<std::string> new_names;
//           Array<Expr> new_shape;
//           std::vector<std::string> new_axis_names;
//           for (const Iterator& iter : iters) {
//             std::set<std::string> ori_iter_names;
//             ExtractOriginalIterators(iter->name, &ori_iter_names);
//             // fused iters have been replaced with iter->ori_iters.
//             // So there should be only one ori iter name extracted from iter->name.
//             CHECK_EQ(ori_iter_names.size(), 1);
//             auto ori_iter_name = BaseName(*ori_iter_names.begin());
//             new_axis_names.push_back(ori_iter_name);
//           }
//           for (size_t i = 0; i < new_axis_names.size(); ++i) {
//             auto iter = iters[i];
//             std::string ori_iter_name;
//             if (i < iters_before_attach) {
//               ori_iter_name = new_axis_names[i + iters_before_attach];
//             } else {
//               ori_iter_name = new_axis_names[i];
//             }
//             if (placeholder_axis_names.count(ori_iter_name)) {
//               os << iter->range->extent << ori_iter_name;
//               new_names.push_back(ori_iter_name);
//               new_shape.push_back(iter->range->extent);
//             }
//           }
//           std::string new_layout = os.str();
//           os.str("");
//           ::tvm::relay::KernelLayoutVisitor::global_new_layouts_queue.push_back(new_layout);
//           placeholder_new_names[placeholder_op] = new_names;
//           placeholder_new_shapes[placeholder_op] = new_shape;

//           Array<Operation> old_ops = pdag->ops;
//           ArrayNode* pops = pdag->ops.CopyOnWrite();

//           // Create new placeholder
//           Operation new_placeholder_op;
//           if (rewrite_placeholder) {
//             new_placeholder_op =
//               te::PlaceholderOpNode::make(placeholder_op->name,
//                                       new_shape,
//                                       placeholder_op.as<te::PlaceholderOpNode>()->dtype);
//           } else {
//             new_placeholder_op = placeholder_op;
//           }

//           Operation new_compute_op, old_compute_op;
//           if (rewrite_body) {
//             Array<Expr> new_body;
//             IndexRewriter index_rewriter(placeholder_new_names,
//                                          placeholder_new_shapes);
//             for (auto& op : old_ops) {
//               if (auto* pop = op.as<ComputeOpNode>()) {
//                 bool need_update = false;
//                 for (auto& t : op->InputTensors()) {
//                   if (t->op == placeholder_op) {
//                     need_update = true;
//                     break;
//                   }
//                 }
//                 if (need_update) {
//                   for (auto& body : pop->body) {
//                     new_body.push_back(index_rewriter.Mutate(body));
//                   }
//                   old_compute_op = op;
//                   CHECK(!new_compute_op.defined());
//                   new_compute_op = ComputeOpNode::make(
//                     pop->name, pop->tag, pop->attrs, pop->axis, new_body);
//                 }
//               }
//             }
//           }

//           // construct the map from old_op to new_op
//           std::unordered_map<Operation, Operation> updated_ops;
//           for (size_t i = 0; i < old_ops.size(); ++i) {
//             auto old_op = old_ops[i];
//             if (rewrite_placeholder && old_op == placeholder_op) {
//               pops->data[i] = new_placeholder_op;
//               updated_ops[placeholder_op] = new_placeholder_op;
//             } else if (rewrite_body && old_op == old_compute_op) {
//               pops->data[i] = new_compute_op;
//               updated_ops[old_compute_op] = new_compute_op;
//             } else {
//               pops->data[i] = old_op;
//             }
//           }

//           // Because ops is sorted in topo-order, only do one pass linear scan here.
//           for (size_t i = 0; i < pops->data.size(); ++i) {
//             auto old_op = Downcast<Operation>(pops->data[i]);
//             if (auto* pop = old_op.as<ComputeOpNode>()) {
//               auto inputs = pop->InputTensors();
//               std::unordered_map<Tensor, Tensor> rmap;
//               for (auto input : inputs) {
//                 auto it = updated_ops.find(input->op);
//                 Operation new_op;
//                 while (it != updated_ops.end()) {
//                   new_op = it->second;
//                   it = updated_ops.find(new_op);
//                 }
//                 if (new_op.defined()) {
//                   int index = input->value_index;
//                   rmap[input] = new_op.output(index);
//                 }
//               }
//               if (!rmap.empty()) {
//                 Operation new_op = pop->ReplaceInputs(old_op, rmap);
//                 updated_ops[old_op] = new_op;
//                 pops->data[i] = new_op;
//               }
//             }
//           }

//           pdag->init_state = StateNode::make(pdag->ops);

//           Array<Tensor> old_tensors = pdag->tensors;
//           ArrayNode* ptensors = pdag->tensors.CopyOnWrite();

//           for (size_t i = 0; i < old_tensors.size(); ++i) {
//             const auto& old_tensor = old_tensors[i];
//             auto it =  updated_ops.find(old_tensor->op);
//             Operation new_op;
//             while (it != updated_ops.end()) {
//               new_op = it->second;
//               it = updated_ops.find(new_op);
//             }
//             if (new_op.defined()) {
//               if (layout_rewrite_level == kBothRewrite) {
//                 auto index = old_tensor->value_index;
//                 ptensors->data[i] = new_op.output(index);
//               } else if (layout_rewrite_level == kComputeRewrite) {
//                 TensorNode* old_tensor_node = const_cast<TensorNode*>(old_tensor.as<TensorNode>());
//                 old_tensor_node->op = new_op;
//               }
//             }
//           }
//         }  // end for placeholder
//       }
//     }
//   }  // end for stage
// }

std::pair<te::Schedule, Array<te::Tensor> > ComputeDAG::ApplySteps(
    const std::vector<Step>& transform_steps,
    LayoutRewriteLevel layout_rewrite_level) const {
  std::vector<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  if (layout_rewrite_level != kNoRewrite && !transform_steps.empty()) {
    ComputeDAG new_dag = *this;
    new_dag.RewriteLayout(transform_steps, layout_rewrite_level);
    return new_dag.ReplaySteps(transform_steps, &stages, &stage_to_axes);
  } else {
    return ReplaySteps(transform_steps, &stages, &stage_to_axes);
  }
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
      ss << " = " << "tuple(" << stage->op->func_name() << ".op.axis)"
         << " + " << "tuple(" << stage->op->func_name() << ".op.reduce_axis)\n";
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

void ComputeDAG::ReplayAndGetDAG(const std::vector<Step> &transform_steps,
                                 ComputeDAG *task_dag) const {
  std::vector<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  te::Schedule sch;
  Array<te::Tensor> old_tensors;

  std::tie(sch, old_tensors) = ReplaySteps(transform_steps, &stages,
                                           &stage_to_axes);

  Array<te::Tensor> new_tensors;
  for (auto stage : sch->stages) {
    if (stage->op->IsInstance<te::PlaceholderOpNode>() ||
        stage->is_output) {
      for (auto i = 0; i < stage->op->num_outputs(); ++i) {
        new_tensors.push_back(stage->op.output(i));
      }
    }
  }

  *task_dag = ComputeDAGNode::make(new_tensors);
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
        new_iters.push_back(IteratorNode::make(iter->name, (*find_res).second,
                                               iter->iter_type,
                                               iter->annotation,
                                               &iter->ori_iters));
      } else {
        LOG(FATAL) << "Infer bound fails";
      }
    }

    pstate->stages[i] = StageNode::make(stage->op, stage->op_type,
            std::move(new_iters), stage->compute_at,
            stage->auto_unroll_max_step, stage->storage_offset);
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

  // todo(lmzheng): should we maintain the attach_map and keep the validity of
  // compute_at an splitted axis?

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
    } else if (auto ps = step.as<FollowSplitStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes, transform_steps);
    } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes, transform_steps);
    } else if (auto ps = step.as<FuseStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<AnnotationStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<ComputeAtStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<ComputeRootStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<ComputeInlineStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<CacheReadStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes, &schedule);
    } else if (auto ps = step.as<CacheWriteStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes, &schedule);
    } else if (auto ps = step.as<PragmaStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes);
    } else if (auto ps = step.as<RfactorStepNode>()) {
      ps->ApplyToSchedule(stages, stage_to_axes, &schedule);
    } else if (auto ps = step.as<StorageAlignStepNode>()) {
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
      ss << op->func_name() << " = PLACEHOLDER " << op.output(0)->shape << "\n";
    } else if (auto pop = op.as<te::ComputeOpNode>()) {
      for (size_t k = 0; k < pop->body.size(); ++k) {
        ss << op->func_name() << "(";
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

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<AccessAnalyzerNode>([](const ObjectRef& ref, ReprPrinter *p) {
  auto* node = static_cast<const AccessAnalyzerNode*>(ref.get());
  for (const auto& op : node->ops_topo_order) {
    p->stream << op << std::endl;
    p->stream << "is_injective:\t" << node->is_injective.at(op) << "\t\t";
    p->stream << "needs_multi_level_tiling:\t"
              << node->needs_multi_level_tiling.at(op) << std::endl;
    p->stream << "is_strict_inlinable:\t" << node->is_strict_inlineable.at(op)
              << "\t";
    p->stream << "is_output:\t" << node->is_output.at(op) << std::endl;
    p->stream << "Read from:\t";
    for (const auto& pair : node->read_from.at(op)) {
      for (const auto& index : pair.second) {
        p->stream << pair.first->func_name() << Array<PrimExpr>(index) << ", ";
      }
    }
    p->stream << "\n";
    p->stream << "Read by:\t";
    for (const auto& pair : node->read_by.at(op)) {
      for (const auto& index : pair.second) {
        p->stream << pair.first->func_name() << Array<PrimExpr>(index) << ", ";
      }
    }
    p->stream << "\n";
    p->stream << "==================================================\n";
  }

  AccessAnalyzer ana = GetRef<AccessAnalyzer>(node);

  p->stream << "ElementwiseMatch: \n";
  for (size_t i = 0; i < node->ops_topo_order.size(); ++i) {
    for (size_t j = 0; j < node->ops_topo_order.size(); ++j) {
      if (i == j) { continue; }
      if (ana.ElementWiseMatch(node->ops_topo_order[i],
                               node->ops_topo_order[j])) {
        p->stream << node->ops_topo_order[i]->func_name() << " -> "
                  << node->ops_topo_order[j]->func_name() << "\n";
      }
    }
  }
});

}  // namespace ansor
}  // namespace tvm
