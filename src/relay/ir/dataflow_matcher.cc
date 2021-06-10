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
 * \file src/tvm/relay/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relay.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <stack>

#include "indexed_graph.h"

namespace tvm {
namespace relay {

// Pattern Matcher

class DominatorMatcher;

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  explicit DFPatternMatcher(const Expr& root_expr) : expr_graph_(CreateIndexedGraph(root_expr)) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Array<Expr>> GetMemo() { return Map<DFPattern, Array<Expr>>(memo_); }
  const IndexedGraph<Expr> expr_graph_;

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const AltPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const IfPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const LetPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  bool MatchesPath(const DominatorPatternNode* op, const Expr& expr);
  bool DominatesParent(const DominatorPatternNode* op, const Expr& expr);

  std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
  bool memoize_ = true;
};

bool DFPatternMatcher::Match(const DFPattern& pattern, const Expr& expr) {
  memo_.clear();
  matched_nodes_.clear();
  return VisitDFPattern(pattern, expr);
}

void DFPatternMatcher::ClearMap(size_t watermark) {
  for (size_t i = watermark; i < matched_nodes_.size(); ++i) {
    memo_.erase(matched_nodes_[i]);
  }
  matched_nodes_.erase(matched_nodes_.begin() + watermark, matched_nodes_.end());
}

bool DFPatternMatcher::VisitDFPattern(const DFPattern& pattern, const Expr& expr) {
  if (memoize_ && memo_.count(pattern)) {
    ICHECK_EQ(memo_[pattern].size(), 1);
    return expr.same_as(memo_[pattern][0]);
  } else {
    auto watermark = matched_nodes_.size();
    auto out = DFPatternFunctor::VisitDFPattern(pattern, expr);
    if (out) {
      memo_[pattern].push_back(expr);
      matched_nodes_.push_back(pattern);
    } else {
      ClearMap(watermark);
    }
    return out;
  }
}

bool DFPatternMatcher::VisitDFPattern_(const AltPatternNode* op, const Expr& expr) {
  return VisitDFPattern(op->left, expr) || VisitDFPattern(op->right, expr);
}

bool MatchRetValue(const ObjectRef& lhs, const TVMRetValue& rhs) {
  switch (rhs.type_code()) {
    case kDLInt:
      if (auto* val = lhs.as<IntImmNode>()) {
        return val->value == rhs.operator int64_t();
      }
      break;
    case kDLFloat:
      if (auto* val = lhs.as<FloatImmNode>()) {
        return val->value == rhs.operator double();
      }
      break;
    case kTVMStr:
      if (auto* val = lhs.as<tir::StringImmNode>()) {
        return val->value == rhs.operator std::string();
      } else if (auto* val = lhs.as<StringObj>()) {
        return val->data == rhs.operator std::string();
      }
      break;
    case kTVMDataType:
      if (auto* val = lhs.as<tir::StringImmNode>()) {
        return rhs.operator std::string() == val->value;
      } else if (auto* val = lhs.as<StringObj>()) {
        return rhs.operator std::string() == val->data;
      } else {
        ICHECK(false) << "PatternMatcher: Unsupported TVMDataType " << lhs;
      }
      break;
    case kTVMObjectHandle:
      if (rhs.IsObjectRef<String>()) {
        if (auto* val = lhs.as<tir::StringImmNode>()) {
          return rhs.operator String() == val->value;
        } else if (auto* val = lhs.as<StringObj>()) {
          return rhs.operator String() == val->data;
        }
      } else {
        // Compare the objects for structural equality
        static auto* structural_equal = runtime::Registry::Get("node.StructuralEqual");
        ICHECK(structural_equal) << "node.StructuralEqual is not registered.";
        if ((*structural_equal)(lhs, GetRef<ObjectRef>(rhs.ptr<Object>()), false, true)) {
          return true;
        }
      }
      break;
    default:
      ICHECK(false) << "Unsupported type code in Pattern Node " << rhs.type_code();
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const AttrPatternNode* attr_pattern, const Expr& expr) {
  bool matches = VisitDFPattern(attr_pattern->pattern, expr);
  if (!matches) {
    return matches;
  }
  auto attributes = attr_pattern->attrs.as<DictAttrsNode>()->dict;
  if (const auto* op_node = expr.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    for (auto kv : attributes) {
      auto attr_name = kv.first;
      auto attr_value = kv.second;
      if (Op::HasAttrMap(attr_name)) {
        auto op_map = Op::GetAttrMap<TVMRetValue>(attr_name);
        if (op_map.count(op)) {
          matches &= MatchRetValue(attr_value, op_map[op]);
        } else {
          matches = false;
        }
      } else {
        matches = false;
      }
    }
  } else if (auto* op = expr.as<CallNode>()) {
    matches = true;
    // TODO(mbrookhart): When OpNode Attrs move from TVMRetValue to the Object system, remove this
    // and replace the whole thing with a Visitor-based approach
    ReflectionVTable* reflection = ReflectionVTable::Global();
    auto attrs_node = const_cast<BaseAttrsNode*>(op->attrs.get());
    // attrs may be undefined on non-op calls so we check first
    std::vector<std::string> attr_names;
    if (attrs_node) {
      attr_names = reflection->ListAttrNames(attrs_node);
    }
    for (auto kv : attributes) {
      std::string attr = kv.first;
      if (matches && std::find(attr_names.begin(), attr_names.end(), attr) != attr_names.end()) {
        matches &= MatchRetValue(kv.second, reflection->GetAttr(attrs_node, attr));
      } else {
        matches = false;
        break;
      }
    }
  } else if (auto* op = expr.as<FunctionNode>()) {
    matches = true;
    for (auto kv : attributes) {
      if (matches && op->attrs.defined() && op->attrs->dict.count(kv.first)) {
        matches &= StructuralEqual()(kv.second, op->attrs->dict[kv.first]);
      } else {
        matches = false;
        break;
      }
    }
  } else {
    matches = false;
  }
  return matches;
}

Array<DFPattern> reverse(const Array<DFPattern>& args) {
  Array<DFPattern> new_args;
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    new_args.push_back(*it);
  }
  return new_args;
}

bool DFPatternMatcher::VisitDFPattern_(const CallPatternNode* op, const Expr& expr) {
  // utilities
  auto get_op_node = [](const CallPatternNode* op) -> const tvm::OpNode* {
    if (op) {
      if (auto* expr_pattern = op->op.as<ExprPatternNode>()) {
        return expr_pattern->expr.as<OpNode>();
      }
    }
    return nullptr;
  };
  auto is_pattern_op = [&get_op_node](const CallPatternNode* op, std::string op_type) {
    if (const auto* op_node = get_op_node(op)) {
      if (op_node->name == op_type) {
        return true;
      }
    }
    return false;
  };
  auto is_expr_op = [](const Expr& expr, std::string op_type) {
    if (const auto* call_node = expr.as<CallNode>()) {
      if (const auto* op_node = call_node->op.as<OpNode>()) {
        if (op_node->name == op_type) {
          return true;
        }
      }
    }
    return false;
  };

  // logic
  auto watermark = matched_nodes_.size();
  if (const auto* call_node = expr.as<CallNode>()) {
    auto matches_op = VisitDFPattern(op->op, call_node->op);
    if (matches_op) {
      auto watermark2 = matched_nodes_.size();

      auto match_args = [this, &watermark2](const Array<DFPattern> pattern_args,
                                            const Array<Expr> expr_args) {
        bool matches = true;
        size_t i = 0;
        if (pattern_args.defined()) {
          if (pattern_args.size() == expr_args.size()) {
            while (matches && i < pattern_args.size()) {
              matches &= VisitDFPattern(pattern_args[i], expr_args[i]);
              ++i;
            }
          } else {
            matches = false;
          }
        }
        if (!matches) {
          ClearMap(watermark2);
        }
        return matches;
      };

      // Standard case
      if (match_args(op->args, call_node->args)) {
        return true;
      }
      // Commutative Matching
      if (const OpNode* op_node = get_op_node(op)) {
        if ((op_node->name == "add") || (op_node->name == "multiply")) {
          if (match_args(reverse(op->args), call_node->args)) {
            return true;
          }
        }
      }
    } else {
      ClearMap(watermark);
      // associate divide/multiply
      if (is_pattern_op(op, "divide")) {
        if (const auto* arg_node = op->args[0].as<CallPatternNode>()) {
          if (is_pattern_op(arg_node, "multiply") && is_expr_op(expr, "multiply") &&
              (is_expr_op(call_node->args[0], "divide") ||
               is_expr_op(call_node->args[1], "divide"))) {
            bool out = false;
            for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
              auto div = CallPattern(op->op, {arg_node->args[arg_id], op->args[1]});
              auto mul = CallPattern(arg_node->op, {arg_node->args[(arg_id + 1) % 2], div});
              out = VisitDFPattern(mul, expr);
              if (out) {
                return true;
              } else {
                ClearMap(watermark);
              }
            }
            return out;
          }
        }
      }
      if (is_pattern_op(op, "multiply")) {
        // associate multiply/divide
        for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
          if (auto* arg_node = op->args[arg_id].as<CallPatternNode>()) {
            if (is_pattern_op(arg_node, "divide") && is_expr_op(expr, "divide") &&
                (is_expr_op(call_node->args[0], "multiply") ||
                 is_expr_op(call_node->args[1], "multiply"))) {
              auto mul = CallPattern(op->op, {arg_node->args[0], op->args[(arg_id + 1) % 2]});
              auto div = CallPattern(arg_node->op, {mul, arg_node->args[1]});
              return VisitDFPattern(div, expr);
            }
          }
        }
      }
    }
  }
  return false;
}

// Recursively find the Dominator parent along all inputs paths.
bool DFPatternMatcher::MatchesPath(const DominatorPatternNode* op, const Expr& expr) {
  auto call_node = expr.as<CallNode>();
  for (auto node : expr_graph_.node_map_.at(expr)->inputs_) {
    if (!(call_node && node->ref_ == call_node->op)) {
      memoize_ = true;
      if (VisitDFPattern(op->parent, node->ref_)) {
        return true;
      } else {
        memoize_ = false;
        if (!VisitDFPattern(op->path, node->ref_) || !MatchesPath(op, node->ref_)) {
          return false;
        }
      }
    }
  }
  return true;
}

// Iteratively ensure that the parent is dominated somewhere by the child or the path
bool DFPatternMatcher::DominatesParent(const DominatorPatternNode* op, const Expr& expr) {
  std::stack<Expr> stack;
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> visited;
  stack.push(expr);
  while (!stack.empty()) {
    Expr current = stack.top();
    stack.pop();
    for (auto node : expr_graph_.node_map_.at(current)->dominator_children_) {
      if (visited.count(node->ref_) == 0) {
        if (VisitDFPattern(op->parent, node->ref_)) {
          return true;
        } else {
          stack.push(node->ref_);
        }
        visited.insert(node->ref_);
      }
    }
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) {
  if (VisitDFPattern(op->child, expr)) {
    bool matches_path = MatchesPath(op, expr);
    memoize_ = true;
    if (matches_path) {
      return DominatesParent(op, expr);
    }
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) {
  return StructuralEqual()(op->expr, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* func = expr.as<FunctionNode>()) {
    matches = true;
    if (op->params.defined()) {
      size_t i = 0;
      if (op->params.size() == func->params.size()) {
        while (matches && i < op->params.size()) {
          matches &= VisitDFPattern(op->params[i], func->params[i]);
          ++i;
        }
      } else {
        matches = false;
      }
    }
    if (matches) {
      matches &= VisitDFPattern(op->body, func->body);
    }
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_get_item_node = expr.as<TupleGetItemNode>()) {
    matches = (op->index == -1 || op->index == tuple_get_item_node->index) &&
              VisitDFPattern(op->tuple, tuple_get_item_node->tuple);
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_node = expr.as<TupleNode>()) {
    matches = true;
    if (op->fields.defined()) {
      if (op->fields.size() == tuple_node->fields.size()) {
        size_t i = 0;
        while (matches && i < op->fields.size()) {
          matches &= VisitDFPattern(op->fields[i], tuple_node->fields[i]);
          ++i;
        }
      } else {
        matches = false;
      }
    }
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const IfPatternNode* op, const Expr& expr) {
  if (const auto* if_node = expr.as<IfNode>()) {
    auto cond = if_node->cond;
    auto true_branch = if_node->true_branch;
    auto false_branch = if_node->false_branch;
    return VisitDFPattern(op->cond, cond) && VisitDFPattern(op->true_branch, true_branch) &&
           VisitDFPattern(op->false_branch, false_branch);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const LetPatternNode* op, const Expr& expr) {
  if (const auto* let_node = expr.as<LetNode>()) {
    return VisitDFPattern(op->var, let_node->var) && VisitDFPattern(op->value, let_node->value) &&
           VisitDFPattern(op->body, let_node->body);
  }
  return false;
}

Expr InferType(const Expr& expr) {
  auto mod = IRModule::FromExpr(expr);
  mod = transform::InferType()(mod);
  if (expr.as<FunctionNode>()) {
    return mod->Lookup("main");
  } else {
    return mod->Lookup("main").as<FunctionNode>()->body;
  }
}

Expr InferTypeWithModule(const Expr& expr, const IRModule& m) {
  IRModule mod(m->functions, m->type_definitions, m->Imports());
  int idx = 0;
  std::string gv_name;
  do {
    std::ostringstream oss;
    oss << "_tmp" << idx;
    gv_name = oss.str();
    ++idx;
  } while (mod->ContainGlobalVar(gv_name));
  GlobalVar gvar(gv_name);
  BaseFunc func;
  if (expr.as<FunctionNode>()) {
    func = Downcast<Function>(expr);
  } else {
    func = relay::Function(relay::FreeVars(expr), expr, Type(), relay::FreeTypeVars(expr, mod), {});
  }
  mod->Add(gvar, func);
  mod = transform::InferType()(mod);
  Expr ret;
  if (expr.as<FunctionNode>()) {
    ret = mod->Lookup(gvar);
  } else {
    ret = mod->Lookup(gvar).as<FunctionNode>()->body;
  }
  return ret;
}

bool DFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  return (StructuralEqual()(op->type, expr_type)) && VisitDFPattern(op->pattern, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  if (const TensorTypeNode* tensor_type = expr_type.as<TensorTypeNode>()) {
    return (StructuralEqual()(op->shape, tensor_type->shape)) && VisitDFPattern(op->pattern, expr);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  if (const TensorTypeNode* tensor_type = expr_type.as<TensorTypeNode>()) {
    return (StructuralEqual()(op->dtype, tensor_type->dtype)) && VisitDFPattern(op->pattern, expr);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const VarPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* var_node = expr.as<VarNode>()) {
    matches = true;
    if (op->name_hint() != "") {
      matches &= op->name_hint() == var_node->name_hint();
    }
  }
  return matches;
}

bool DFPatternMatcher::VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr) {
  return expr.as<ConstantNode>() != nullptr;
}

bool DFPatternMatcher::VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) {
  return true;
}

bool MatchPattern(DFPattern pattern, Expr expr) {
  return DFPatternMatcher(expr).Match(pattern, expr);
}

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.match").set_body_typed(MatchPattern);

/*!
 * \brief PatternGrouper does pre-rewriting pattern matching and analysis
 *
 * This class creates a number of groups of matched expressions, ensures they don't overlap, and
 * returns them to the caller for post-analysis rewriting.
 *
 * This is primarily needed to support the post-dominator analysis required for dominator pattern
 * matching.
 */
class PatternGrouper {
 public:
  /*! \brief Internal Group class for storing analysis */
  struct Group {
    Expr root_node;
    int gid;
    Map<DFPattern, Array<Expr>> matched_nodes;
    std::string name;
    Function function;
    Array<Expr> args;
  };

  /*! \brief Return the group assignments of expressions */
  const std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual>& GetGIDAssignments() {
    return gid_assignments_;
  }
  /*! \brief Group expressions that match the pattern */
  const std::unordered_map<int, Group>& GroupMatches(const DFPattern& pattern, const Expr& pre) {
    groups_.clear();
    gid_assignments_.clear();

    pattern_ = pattern;
    pattern_graph_ = CreateIndexedGraph(pattern_);
    auto matcher = DFPatternMatcher(pre);
    matcher_ = &matcher;
    this->VisitExprs();
    return this->groups_;
  }

 protected:
  /*! \brief Iteratively traverse the Expression in pre-order to find subgraphs
   *
   * If we traverse the graph in post-order, we can run into situtations where a small subgraph will
   * match the pattern. Due to options like AltPattern, a larger subgraph with more nodes later in
   * the graph may also match the pattern. With post-order traversal, we mark the smaller subgraph
   * as matched and fail to catch the larger subgraph. This problem is fixed by using pre-order
   * traversal.
   */
  void VisitExprs() {
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> pre_partitioned;
    for (size_t i = matcher_->expr_graph_.topological_order_.size(); i != 0; --i) {
      size_t index = i - 1;
      Expr current = matcher_->expr_graph_.topological_order_.at(index)->ref_;
      if (gid_assignments_.count(current) == 0) {  // Don't visit nodes we've already grouped
        if (auto op = current.as<FunctionNode>()) {
          if (op->attrs.defined() && op->attrs->dict.count(attr::kPartitionedFromPattern) != 0) {
            pre_partitioned.insert(current);
            PostOrderVisit(op->body,
                           [&pre_partitioned](const Expr& expr) { pre_partitioned.insert(expr); });
          }
        }
        if (pre_partitioned.count(current) == 0 && matcher_->Match(pattern_, current)) {
          CreateGroup(current);
        }
      }
    }
  }
  /*! \brief Creates a new set of nodes based on Group inputs, used to create functions and perform
   * group overlap analysis */
  class MatchExtractor : public ExprMutator {
   public:
    explicit MatchExtractor(
        const std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual>& inputs)
        : inputs_(inputs) {}
    const std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual>& GetMemo() {
      return this->memo_;
    }
    const std::string& GetName() { return name_; }

   protected:
    Expr VisitExpr(const Expr& pre) override {
      if (inputs_.count(pre)) {
        return inputs_.at(pre);
      }
      return ExprMutator::VisitExpr(pre);
    }
    Expr VisitExpr_(const TupleNode* op) override {
      auto out = ExprMutator::VisitExpr_(op);
      name_ += "Tuple_";
      return out;
    };
    Expr VisitExpr_(const FunctionNode* op) override {
      auto out = ExprMutator::VisitExpr_(op);
      name_ += "Function";
      return out;
    };
    Expr VisitExpr_(const CallNode* call_node) override {
      auto out = ExprMutator::VisitExpr_(call_node);
      if (auto operation = call_node->op.as<OpNode>()) {
        name_ += operation->name + "_";
      } else {
        name_ += "Call_";
      }
      return out;
    };
    Expr VisitExpr_(const LetNode* op) override {
      auto out = ExprMutator::VisitExpr_(op);
      name_ += "Let_";
      return out;
    };
    Expr VisitExpr_(const IfNode* op) override {
      auto out = ExprMutator::VisitExpr_(op);
      name_ += "If_";
      return out;
    };
    Expr VisitExpr_(const TupleGetItemNode* op) override {
      auto out = ExprMutator::VisitExpr_(op);
      name_ += "TupleGetItem" + std::to_string(op->index) + "_";
      return out;
    };
    Expr VisitExpr_(const MatchNode* op) override {
      auto out = ExprMutator::VisitExpr_(op);
      name_ += "Match_";
      return out;
    };
    std::string name_;
    const std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> inputs_;
  };

  /*! \brief Create a group based on a matched expression */
  void CreateGroup(const Expr& expr) {
    int var_number = 0;

    auto node_map = matcher_->GetMemo();
    // Get fuzzy patterns
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> fuzzy_matches;
    for (auto node : pattern_graph_.topological_order_) {
      // Don't treat fuzzy Dominator patterns input variables for partition
      if (auto op = node->ref_.as<DominatorPatternNode>()) {
        for (auto fuzzy_op : {op->parent, op->path}) {
          for (auto match : node_map[fuzzy_op]) {
            fuzzy_matches.insert(match);
          }
        }
      }
      // Don't treat Function params or body as input variables for partition
      if (node->ref_.as<FunctionPatternNode>()) {
        auto matches = node_map[node->ref_];
        for (auto match : matches) {
          auto graph = CreateIndexedGraph(match.as<FunctionNode>()->body);
          for (auto node : graph.topological_order_) {
            fuzzy_matches.insert(node->ref_);
          }
        }
      }
    }

    // Create input variables
    Group group;
    group.root_node = expr;
    group.matched_nodes = node_map;

    std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> inputs;
    Array<Var> params;

    for (auto node : pattern_graph_.topological_order_) {
      auto make_input = [&](const Expr& input) {
        if (fuzzy_matches.count(input) == 0 && input.as<OpNode>() == nullptr &&
            input.as<FunctionNode>() == nullptr && !EmbedConst(input, node->ref_)) {
          inputs[input] =
              Var("FunctionVar_" + std::to_string(graph_number_) + "_" + std::to_string(var_number),
                  NullValue<Type>());
          group.args.push_back(input);
          params.push_back(inputs[input]);
          var_number++;
        }
      };
      auto tuple = node->ref_.as<TuplePatternNode>();
      auto call = node->ref_.as<CallPatternNode>();
      if (tuple && !tuple->fields.defined()) {
        if (node_map.count(node->ref_)) {
          auto matches = node_map[node->ref_];
          for (auto match : matches) {
            for (auto input : match.as<TupleNode>()->fields) {
              make_input(input);
            }
          }
        }
      } else if (call && !call->args.defined()) {
        if (node_map.count(node->ref_)) {
          auto matches = node_map[node->ref_];
          for (auto match : matches) {
            for (auto input : match.as<CallNode>()->args) {
              make_input(input);
            }
          }
        }
      } else if (node->inputs_.size() == 0) {
        if (node_map.count(node->ref_)) {
          auto matches = node_map[node->ref_];
          for (auto match : matches) {
            make_input(match);
          }
        }
      }
    }

    graph_number_++;

    // Extract a Function. Used in Partition directly,
    // used to determine Group overlap in other passes
    auto extractor = MatchExtractor(inputs);
    auto body = extractor.Mutate(expr);

    group.function = Function(params, body, NullValue<Type>(), Array<TypeVar>());
    group.name = extractor.GetName();
    // Check to make sure we aren't overlapping with another group or creating an invalid fusion
    // The MatchExtractor will create a new graph by replacing nodes that match the inputs of the
    // pattern with the input FunctionVar* Variables. The resulting memoization map will only
    // contain nodes in the expression that matched the pattern. If a non-input node of the pattern
    // (i.e., some piece of computation) overlaps with the nodes in a previous group, we'll have a
    // situation where we try to rewrite the same node twice in the second rewriting or parition
    // pass. This isn't valid, so we check for it here. We ignore Ops, functions, and constants
    // because they exist more globally outside of the fusion.
    // Similiarly, if interior nodes in a group are used outside of the group fusing to a single
    // output would create an invalid graph tranformation, so we block the creation of such groups.
    auto memo = extractor.GetMemo();
    for (auto kv : memo) {
      // Check to ensure that this node isn't an input or a global
      if (inputs.count(kv.first) == 0 && kv.first.as<OpNode>() == nullptr &&
          kv.first.as<FunctionNode>() == nullptr && kv.first.as<ConstantNode>() == nullptr) {
        if (gid_assignments_.count(kv.first) != 0) {
          // check to see if the node is use in other groups
          // Exit due to overlapping partitions
          return;
        } else if (kv.second != body) {
          // if the node isn't the output of the group
          auto node = matcher_->expr_graph_.node_map_.at(kv.first);
          for (auto* output : node->outputs_) {
            // and the node is used by nodes outside of the group
            if (memo.count(output->ref_) == 0 &&
                !matcher_->expr_graph_.node_map_.at(expr)->Dominates(output)) {
              // Exit because nodes in this pattern's body are used outside the pattern
              // fusing it would be invalid
              return;
            }
          }
        }
      }
    }
    // Assign Group Ids
    group.gid = ++gid_;
    for (auto kv : extractor.GetMemo()) {
      gid_assignments_[kv.first] = gid_;
    }

    // Save Group
    groups_[group.gid] = std::move(group);
  }

  /*! \brief EmbedConst implements rules for embedding constants into partitioned functions or
   * lifting them into the function arguments.
   *
   * The rules depend on what pattern the ConstantNode matched.
   *
   * The basic rules are:
   *  If the constant matches ExprPattern(relay.const(*)) or a ConstantPattern(), embed the constant
   * in the partitioned function. If the constant matched an AltPattern, recursively check the
   * matched side of the pattern. For any other matching pattern (i.e, wildcard, VarPattern, etc),
   * lift the constant into the arguments of the partitioned function.
   */
  bool EmbedConst(const Expr& expr, const DFPattern pattern) {
    bool embed = false;
    if (expr.as<ConstantNode>()) {
      if (pattern.as<ConstantPatternNode>() != nullptr) {
        embed = true;
      } else if (auto expr_pat = pattern.as<ExprPatternNode>()) {
        if (expr_pat->expr.as<ConstantNode>()) {
          embed = true;
        }
      } else if (auto alt_pat = pattern.as<AltPatternNode>()) {
        if (matcher_->Match(alt_pat->left, expr)) {
          embed = EmbedConst(expr, alt_pat->left);
        } else {
          embed = EmbedConst(expr, alt_pat->right);
        }
      }
    }
    return embed;
  }
  // Internal State
  DFPattern pattern_;
  std::unordered_map<int, Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
  DFPatternMatcher* matcher_ = nullptr;
  IndexedGraph<DFPattern> pattern_graph_;
  int gid_ = 0;
  int graph_number_ = 0;
};

// Rewrite

DFPatternCallback::DFPatternCallback(DFPattern pattern, PackedFunc function, bool require_type) {
  ObjectPtr<DFPatternCallbackNode> n = make_object<DFPatternCallbackNode>();
  n->pattern = std::move(pattern);
  n->function = std::move(function);
  n->require_type = require_type;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(DFPatternCallbackNode);

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.DFPatternCallback")
    .set_body_typed([](DFPattern pattern, PackedFunc function, bool require_type) {
      return DFPatternCallback(pattern, function, require_type);
    });

/*!
 * \brief PatternRewriter rewrites the expression by finding matches and allowing user callback
 * function to rewrite those matches
 *
 * The class uses PatternGrouper to support the dominator pattern.
 */
class PatternRewriter : protected MixedModeMutator {
 public:
  PatternRewriter(IRModule mod) : mod_(mod) {}
  /*! \brief Rewrite can take a number of callbacks and will repeatedly rewrite the graph with the
   * callbacks until it stops changing */
  Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) {
    auto post = pre;
    auto last = post;
    // rewrite the graph until it stops changing to make sure all rewrites are complete
    int count = 0;
    bool equal = true;
    static auto* structural_equal = runtime::Registry::Get("node.StructuralEqual");
    ICHECK(structural_equal) << "node.StructuralEqual is not registered.";
    do {
      last = post;
      for (auto callback : callbacks) {
        callback_ = callback;
        if (callback_->require_type) {
          post = InferTypeWithModule(post, mod_);
        }
        auto grouper = PatternGrouper();
        groups_ = grouper.GroupMatches(callback_->pattern, post);
        gid_assignments_ = grouper.GetGIDAssignments();
        memo_.clear();
        post = this->VisitExpr(post);
        count++;
      }
      equal = (*structural_equal)(last, post, false, true);
    } while (!equal && count < 100);
    if (count >= 100) {
      LOG(FATAL) << "Observed 100 rewrite passes, possible conflicting passes?";
    }
    return post;
  }

 protected:
  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    if (gid_assignments_.count(pre) && pre == groups_[gid_assignments_[pre]].root_node) {
      // Convert the pre-rewrite node map to a post-rewrite node map
      auto group = groups_[gid_assignments_[pre]];
      std::unordered_map<DFPattern, Array<Expr>, ObjectPtrHash, ObjectPtrEqual> node_map;
      for (auto kv : group.matched_nodes) {
        Array<Expr> tmp;
        for (size_t i = 0; i < kv.second.size(); ++i) {
          tmp.push_back(this->memo_[kv.second[i]]);
        }
        node_map.insert({kv.first, tmp});
      }
      // run the user callback function
      return callback_->function(pre, post, Map<DFPattern, Array<Expr>>(node_map));
    }
    return post;
  }

  IRModule mod_;
  DFPatternCallback callback_;
  std::unordered_map<int, PatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
};

Expr RewritePatterns(Array<DFPatternCallback> callbacks, Expr expr, IRModule mod) {
  return PatternRewriter(mod).Rewrite(callbacks, expr);
}

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.rewrite").set_body_typed(RewritePatterns);

/*!
 * \brief PatternPartitioner replaces expressions that match a pattern with function call that
 * perform the same computation but allow for further analysis and lowering.
 *
 * The class uses PatternGrouper to support the dominator pattern.
 */
class PatternPartitioner : protected MixedModeMutator {
 public:
  Expr Partition(const DFPattern& pattern, const Expr& pre, const Map<String, ObjectRef>& attrs,
                 PackedFunc check) {
    if (pattern.as<FunctionPatternNode>()) {
      LOG(WARNING) << "Partioning a Function that isn't called doesn't make sense, skipping"
                   << pattern;
      return pre;
    }
    auto grouper = PatternGrouper();
    groups_ = grouper.GroupMatches(pattern, pre);
    gid_assignments_ = grouper.GetGIDAssignments();
    attrs_ = attrs;
    check_ = check;
    return this->VisitExpr(pre);
  }

 protected:
  Expr RewritePartition(const PatternGrouper::Group& group) {
    Array<Expr> args;
    for (size_t i = 0; i < group.args.size(); ++i) {
      args.push_back(memo_[group.args[i]]);
    }
    Function func = WithAttr(group.function, attr::kPartitionedFromPattern, String(group.name));
    if (!attrs_.empty()) {
      for (auto kv : attrs_) {
        func = WithAttr(std::move(func), kv.first, kv.second);
      }
    }
    return Call(func, args);
  }

  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    if (gid_assignments_.count(pre) && pre == groups_[gid_assignments_[pre]].root_node &&
        static_cast<bool>(check_(pre))) {
      post = RewritePartition(groups_[gid_assignments_[pre]]);
    }
    return post;
  }

  Map<String, ObjectRef> attrs_;
  std::unordered_map<int, PatternGrouper::Group> groups_;
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> gid_assignments_;
  PackedFunc check_;
};

Expr PartitionPattern(DFPattern pattern, Expr expr, Map<String, ObjectRef> attrs,
                      PackedFunc check) {
  return PatternPartitioner().Partition(pattern, expr, attrs, check);
}

TVM_REGISTER_GLOBAL("relay.dataflow_pattern.partition")
    .set_body_typed([](DFPattern pattern, Expr expr, Map<String, ObjectRef> attrs,
                       PackedFunc check) { return PartitionPattern(pattern, expr, attrs, check); });

}  // namespace relay
}  // namespace tvm
