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

namespace tvm {
namespace relay {

// Pattern Matcher

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  bool Match(const DFPattern& pattern, const Expr& expr);

 protected:
  bool VisitDFPattern(const DFPattern& pattern, const Expr& expr) override;
  bool VisitDFPattern_(const AltPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const AttrPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const CallPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const TypePatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const VarPatternNode* op, const Expr& expr) override;
  bool VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) override;

  void ClearMap(size_t watermark);
  std::unordered_map<DFPattern, Expr, ObjectHash, ObjectEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
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
  if (memo_.count(pattern)) {
    return expr.same_as(memo_[pattern]);
  } else {
    auto watermark = matched_nodes_.size();
    auto out = DFPatternFunctor::VisitDFPattern(pattern, expr);
    if (out) {
      memo_[pattern] = expr;
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
bool DFPatternMatcher::VisitDFPattern_(const AttrPatternNode* attr_pattern, const Expr& expr) {
  bool matches = false;
  if (const auto* op_node = expr.as<OpNode>()) {
    Op op = GetRef<Op>(op_node);
    auto attributes = attr_pattern->attrs.as<DictAttrsNode>()->dict;
    for (auto kv : attributes) {
      auto attr_name = kv.first;
      auto attr_value = kv.second;
      auto op_map = Op::GetAttr<TVMRetValue>(attr_name);
      if (op_map.count(op)) {
        switch (op_map[op].type_code()) {
          case kDLInt:
            if (auto* val = kv.second.as<IntImmNode>()) {
              std::cout << op << " " << op_map[op].operator int64_t() <<  std::endl;
              matches = val->value == op_map[op].operator int64_t();
            }
            break;
          case kDLFloat:
            if (auto* val = kv.second.as<FloatImmNode>()) {
              matches = val->value == op_map[op].operator double();
            }
            break;
          case kTVMStr:
            if (auto* val = kv.second.as<tir::StringImmNode>()) {
              matches = val->value == op_map[op].operator std::string();
            }
            break;
          default:
            throw "Unsupported type";
        }
      }
    }
  }
  return matches;
}

Array<DFPattern> reverse(const Array<DFPattern> args) {
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
        if (pattern_args.size() == expr_args.size()) {
          while (matches && i < pattern_args.size()) {
            matches &= VisitDFPattern(pattern_args[i], expr_args[i]);
            ++i;
          }
        } else {
          matches = false;
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
          if (is_pattern_op(arg_node, "multiply")) {
            if (is_expr_op(expr, "multiply")) {
              if (is_expr_op(call_node->args[0], "divide") ||
                  is_expr_op(call_node->args[1], "divide")) {
                bool out = false;
                for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
                  auto div = CallPatternNode::make(op->op, {arg_node->args[arg_id], op->args[1]},
                                                   op->attrs, op->type_args);
                  auto mul =
                      CallPatternNode::make(arg_node->op, {arg_node->args[(arg_id + 1) % 2], div},
                                            arg_node->attrs, arg_node->type_args);
                  out = VisitDFPattern(mul, expr);
                  if (out) {
                    return out;
                  } else {
                    ClearMap(watermark);
                  }
                }
                return out;
              }
            }
          }
        }
      }
      if (is_pattern_op(op, "multiply")) {
        // associate multiply/divide
        for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
          if (auto* arg_node = op->args[arg_id].as<CallPatternNode>()) {
            if (is_pattern_op(arg_node, "divide")) {
              if (is_expr_op(expr, "divide")) {
                if (is_expr_op(call_node->args[0], "multiply") ||
                    is_expr_op(call_node->args[1], "multiply")) {
                  auto mul =
                      CallPatternNode::make(op->op, {arg_node->args[0], op->args[(arg_id + 1) % 2]},
                                            op->attrs, op->type_args);
                  auto div = CallPatternNode::make(arg_node->op, {mul, arg_node->args[1]},
                                                   arg_node->attrs, arg_node->type_args);
                  return VisitDFPattern(div, expr);
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}
bool DFPatternMatcher::VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) {
  auto watermark = matched_nodes_.size();
  auto backup_memo = memo_;
  auto backup_matched_nodes = matched_nodes_;

  if (VisitDFPattern(op->child, expr)) {
    auto child_graph = CreateIndexedGraph(GetRef<DFPattern>(op));
    auto expr_graph = CreateIndexedGraph(expr);
    auto find_dominated = [&child_graph, this](const DFPattern& node) {
      std::unordered_set<Expr, ObjectHash, ObjectEqual> dominated_exprs;
      auto indexed_node = child_graph.node_map_[node];
      for (auto dominated : indexed_node->dominator_children_) {
        if (dominated->ref_.as<WildcardPatternNode>() || dominated->ref_.as<OpNode>()) {
          continue;
        }
        dominated_exprs.insert(memo_[dominated->ref_]);
      }
      return dominated_exprs;
    };
    std::function<bool(const Expr&, const std::unordered_set<Expr, ObjectHash, ObjectEqual>&)>
        find_parent;
    find_parent = [this, &op, &watermark, &backup_memo, &backup_matched_nodes, &find_dominated,
                   &expr_graph, &find_parent](
                      const Expr& expr,
                      const std::unordered_set<Expr, ObjectHash, ObjectEqual>& dominated_exprs) {
      bool out = true;
      for (auto node : expr_graph.node_map_[expr]->dominator_children_) {
        if (out && dominated_exprs.count(node->ref_) == 0) {
          if (VisitDFPattern(op->parent, node->ref_)) {
            backup_memo[op->parent] = memo_.at(op->parent);
            backup_matched_nodes.push_back(op->parent);
            memo_ = backup_memo;
            matched_nodes_ = backup_matched_nodes;
            watermark += 1;
            return true;
          } else {
            if (VisitDFPattern(op->path, node->ref_)) {
              auto new_dominated_exprs = find_dominated(op->path);
              std::cout << watermark << std::endl;
              ClearMap(watermark);
              out &= find_parent(node->ref_, new_dominated_exprs);
            } else {
              out = false;
            }
          }
        }
      }
      return out;
    };

    auto dominated_exprs = find_dominated(op->child);
    ClearMap(watermark);
    bool matches = find_parent(expr, dominated_exprs);
    if (matches) {
      backup_memo[op->parent] = memo_.at(op->parent);
      backup_memo[op->child] = expr;
      memo_ = backup_memo;
    }
    return matches;
  }
  return false;
}
bool DFPatternMatcher::VisitDFPattern_(const ExprPatternNode* op, const Expr& expr) {
  return StructuralEqual()(op->expr, expr);
}
bool DFPatternMatcher::VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_get_item_node = expr.as<TupleGetItemNode>()) {
    matches = (op->index == tuple_get_item_node->index) &&
              VisitDFPattern(op->tuple, tuple_get_item_node->tuple);
  }
  return matches;
}
bool DFPatternMatcher::VisitDFPattern_(const TuplePatternNode* op, const Expr& expr) {
  bool matches = false;
  if (const auto* tuple_node = expr.as<TupleNode>()) {
    if (op->fields.size() == tuple_node->fields.size()) {
      matches = true;
      size_t i = 0;
      while (matches && i < op->fields.size()) {
        matches &= VisitDFPattern(op->fields[i], tuple_node->fields[i]);
        ++i;
      }
    }
  }
  return matches;
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
bool DFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr) {
  auto expr_type = InferType(expr).as<ExprNode>()->checked_type();
  return (StructuralEqual()(op->type, expr_type)) && VisitDFPattern(op->pattern, expr);
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
bool DFPatternMatcher::VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) {
  return true;
}

TVM_REGISTER_GLOBAL("relay.df_pattern.match").set_body_typed([](DFPattern pattern, Expr expr) {
  return DFPatternMatcher().Match(pattern, expr);
});

// Rewrite

DFPatternCallback DFPatternCallbackNode::make(DFPattern pattern, PackedFunc function) {
  ObjectPtr<DFPatternCallbackNode> n = make_object<DFPatternCallbackNode>();
  n->pattern_ = std::move(pattern);
  n->function_ = std::move(function);
  return DFPatternCallback(n);
}

TVM_REGISTER_NODE_TYPE(DFPatternCallbackNode);

TVM_REGISTER_GLOBAL("relay.df_pattern.DFPatternCallback")
    .set_body_typed(DFPatternCallbackNode::make);

class PatternRewriter : protected MixedModeMutator {
 public:
  explicit PatternRewriter(const Array<DFPatternCallback>& callbacks) : callbacks_(callbacks) {}
  Expr Rewrite(const Expr& pre) { return this->VisitExpr(pre); }

 protected:
  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    Expr out = post;
    for (auto& callback : callbacks_) {
      if (auto* callback_node = callback.as<DFPatternCallbackNode>()) {
        if (matcher_.Match(callback_node->pattern_, out)) {
          out = callback_node->function_(pre, out);
        }
      }
    }
    return out;
  }
  DFPatternMatcher matcher_;
  Array<DFPatternCallback> callbacks_;
};

Expr RewritePatterns(Array<DFPatternCallback> callbacks, Expr expr) {
  return PatternRewriter(callbacks).Rewrite(expr);
}

TVM_REGISTER_GLOBAL("relay.df_pattern.rewrite").set_body_typed(RewritePatterns);

}  // namespace relay
}  // namespace tvm
