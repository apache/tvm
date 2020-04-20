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


class DominatorMatcher;

class DFPatternMatcher : public DFPatternFunctor<bool(const DFPattern&, const Expr&)> {
 public:
  explicit DFPatternMatcher(const Expr& root_expr) : expr_graph_(CreateIndexedGraph(root_expr)) {}
  bool Match(const DFPattern& pattern, const Expr& expr);
  Map<DFPattern, Array<Expr>> GetMemo() { return Map<DFPattern, Array<Expr>>(memo_); }

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
  std::unordered_set<Expr, ObjectHash, ObjectEqual> FindDominated(const DFPattern& node);
  bool FindParent(const Expr& expr,
                  const std::unordered_set<Expr, ObjectHash, ObjectEqual>& dominated_exprs,
                  const DominatorPatternNode* op);

  std::unordered_map<DFPattern, Array<Expr>, ObjectHash, ObjectEqual> memo_;
  std::vector<DFPattern> matched_nodes_;
  IndexedGraph<Expr> expr_graph_;
  IndexedGraph<DFPattern> pattern_graph_;
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
    CHECK_EQ(memo_[pattern].size(), 1);
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
          if (is_pattern_op(arg_node, "multiply") && is_expr_op(expr, "multiply") &&
                 (is_expr_op(call_node->args[0], "divide") ||
                  is_expr_op(call_node->args[1], "divide"))) {
            bool out = false;
            for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
              auto div = CallPatternNode::make(op->op, {arg_node->args[arg_id], op->args[1]},
                                               op->attrs, op->type_args);
              auto mul =
                  CallPatternNode::make(arg_node->op, {arg_node->args[(arg_id + 1) % 2], div},
                                        arg_node->attrs, arg_node->type_args);
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
  return false;
}

std::unordered_set<Expr, ObjectHash, ObjectEqual> DFPatternMatcher::FindDominated(
    const DFPattern& node) {
  std::unordered_set<Expr, ObjectHash, ObjectEqual> dominated_exprs;
  auto indexed_node = pattern_graph_.node_map_[node];
  for (auto dominated : indexed_node->dominator_children_) {
    if (dominated->ref_.as<WildcardPatternNode>()) {
      continue;
    }
    if (memo_.count(dominated->ref_)) {
      Array<Expr> matched = memo_[dominated->ref_];
      dominated_exprs.insert(matched[matched.size() - 1]);
    }
  }
  return dominated_exprs;
}

bool DFPatternMatcher::FindParent(
    const Expr& expr, const std::unordered_set<Expr, ObjectHash, ObjectEqual>& dominated_exprs,
    const DominatorPatternNode* op) {
  bool out = true;
  for (auto node : expr_graph_.node_map_[expr]->dominator_children_) {
    if (out && dominated_exprs.count(node->ref_) == 0 && node->ref_.as<OpNode>() == nullptr) {
      memoize_ = true;
      if (VisitDFPattern(op->parent, node->ref_)) {
        return true;
      } else {
        memoize_ = false;
        if (VisitDFPattern(op->path, node->ref_)) {
          auto new_dominated_exprs = FindDominated(op->path);
          out &= FindParent(node->ref_, new_dominated_exprs, op);
        } else {
          return false;
        }
      }
    }
  }
  return out;
}

bool DFPatternMatcher::VisitDFPattern_(const DominatorPatternNode* op, const Expr& expr) {
  pattern_graph_ = CreateIndexedGraph(GetRef<DFPattern>(op));
  if (VisitDFPattern(op->child, expr)) {
    memoize_ = false;
    auto dominated_exprs = FindDominated(op->child);
    bool matches = FindParent(expr, dominated_exprs, op);
    memoize_ = true;
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
  return DFPatternMatcher(expr).Match(pattern, expr);
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
  PatternRewriter() {}
  Expr Rewrite(const Array<DFPatternCallback>& callbacks, const Expr& pre) {
    auto post = pre;
    auto last = post;
    // rewrite the graph until it stops changing to make sure all rewrites are complete
    do {
      last = post;
      for (auto callback : callbacks) {
        callback_ = &callback;
        auto matcher = DFPatternMatcher(post);
        matcher_ = &matcher;
        memo_.clear();
        post = this->VisitExpr(post);
      }
    } while (last != post);
    return post;
  }

 protected:
  Expr DispatchVisitExpr(const Expr& pre) override {
    auto post = MixedModeMutator::DispatchVisitExpr(pre);
    if (auto* callback_node = callback_->as<DFPatternCallbackNode>()) {
      if (matcher_->Match(callback_node->pattern_, post)) {
        return callback_node->function_(pre, post, matcher_->GetMemo());
      }
    }
    return post;
  }

  DFPatternMatcher* matcher_ = nullptr;
  DFPatternCallback* callback_ = nullptr;
};

Expr RewritePatterns(Array<DFPatternCallback> callbacks, Expr expr) {
  return PatternRewriter().Rewrite(callbacks, expr);
}

TVM_REGISTER_GLOBAL("relay.df_pattern.rewrite").set_body_typed(RewritePatterns);

}  // namespace relay
}  // namespace tvm
