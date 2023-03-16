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
 * \file src/relax/ir/dataflow_matcher.cc
 * \brief The dataflow pattern matcher for Relax.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/tir/op.h>

#include <array>
#include <cstddef>
#include <limits>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "dataflow_matcher_impl.h"

namespace tvm {
namespace relax {

using tvm::arith::Analyzer;

// Pattern Matcher
bool DFPatternMatcher::Match(const DFPattern& pattern, const Expr& expr) {
  memo_.clear();
  matched_nodes_.clear();
  return VisitDFPattern(pattern, expr);
}

static Expr TryGetValOfVar(const Expr& expr, const Map<Var, Expr>& var2val) {
  if (var2val.empty()) return expr;

  // if not match, try to match value of var if expr is a var.
  if (const VarNode* var = expr.as<VarNode>()) {
    auto may = var2val.Get(GetRef<Var>(var));
    if (may.defined()) return may.value();
  }

  return expr;
}

void DFPatternMatcher::ClearMap(size_t watermark) {
  for (size_t i = watermark; i < matched_nodes_.size(); ++i) {
    memo_.erase(matched_nodes_[i]);
  }
  matched_nodes_.erase(matched_nodes_.begin() + watermark, matched_nodes_.end());
}

bool DFPatternMatcher::VisitDFPattern(const DFPattern& pattern, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  if (memoize_ && memo_.count(pattern)) {
    ICHECK_EQ(memo_[pattern].size(), 1);
    return expr.same_as(memo_[pattern][0]);
  } else {
    size_t watermark = matched_nodes_.size();
    bool out = DFPatternFunctor::VisitDFPattern(pattern, expr);
    if (out) {
      memo_[pattern].push_back(expr);
      matched_nodes_.push_back(pattern);
    } else {
      ClearMap(watermark);
    }
    return out;
  }
}

bool DFPatternMatcher::VisitDFPattern_(const OrPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  return VisitDFPattern(op->left, expr) || VisitDFPattern(op->right, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const AndPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  return VisitDFPattern(op->left, expr) && VisitDFPattern(op->right, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const NotPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  return !VisitDFPattern(op->reject, expr);
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

bool DFPatternMatcher::VisitDFPattern_(const AttrPatternNode* attr_pattern, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  bool matches = VisitDFPattern(attr_pattern->pattern, expr);
  if (!matches) return matches;
  VLOG(1) << "considering AttrPatternNode at:\n" << expr;
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

bool DFPatternMatcher::VisitDFPattern_(const CallPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
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

      auto match_args = [this, &watermark2](const Array<DFPattern>& pattern_args, auto expr_begin,
                                            auto expr_end) {
        bool matches = true;
        auto pattern_it = pattern_args.begin();
        auto expr_it = expr_begin;
        if (pattern_args.defined()) {
          while (matches && pattern_it != pattern_args.end())
            matches &= VisitDFPattern(*(pattern_it++), *(expr_it++));
        }
        if (!matches) ClearMap(watermark2);
        return matches;
      };

      const size_t n_arg_pattern = op->args.size();
      const size_t n_arg_expr = call_node->args.size();
      // if allow variable args, #pattern must >= #expr.
      if (op->varg_default_wildcard && n_arg_expr < n_arg_pattern) return false;
      // if variable args are not allowed, #pattern must == #expr.
      if (!op->varg_default_wildcard && n_arg_expr != n_arg_pattern) return false;

      // Standard case
      if (match_args(op->args, call_node->args.begin(), call_node->args.end())) return true;

      // Commutative Matching
      if (const OpNode* op_node = get_op_node(op)) {
        if ((op_node->name == "relax.add") || (op_node->name == "relax.multiply")) {
          if (match_args(op->args, call_node->args.rbegin(), call_node->args.rend())) {
            return true;
          }
        }
      }
    } else {
      ClearMap(watermark);
      // associate divide/multiply
      if (is_pattern_op(op, "relax.divide")) {
        if (const auto* arg_node = op->args[0].as<CallPatternNode>()) {
          if (is_pattern_op(arg_node, "relax.multiply") && is_expr_op(expr, "relax.multiply") &&
              (is_expr_op(call_node->args[0], "relax.divide") ||
               is_expr_op(call_node->args[1], "relax.divide"))) {
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
      if (is_pattern_op(op, "relax.multiply")) {
        // associate multiply/divide
        for (size_t arg_id = 0; arg_id < 2; ++arg_id) {
          if (auto* arg_node = op->args[arg_id].as<CallPatternNode>()) {
            if (is_pattern_op(arg_node, "relax.divide") && is_expr_op(expr, "relax.divide") &&
                (is_expr_op(call_node->args[0], "relax.multiply") ||
                 is_expr_op(call_node->args[1], "relax.multiply"))) {
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

bool DFPatternMatcher::VisitDFPattern_(const ExprPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  return StructuralEqual()(op->expr, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
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

bool DFPatternMatcher::VisitDFPattern_(const TupleGetItemPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  if (const auto* tuple_get_item_node = expr.as<TupleGetItemNode>()) {
    return (op->index == -1 || op->index == tuple_get_item_node->index) &&
           VisitDFPattern(op->tuple, tuple_get_item_node->tuple);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const TuplePatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  bool matches = false;
  if (const auto* tuple_node = expr.as<TupleNode>()) {
    matches = true;
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
  return matches;
}

bool DFPatternMatcher::TryUnorderedMatch(size_t idx, const tvm::Array<DFPattern> patterns,
                                         const tvm::Array<Expr> fields,
                                         std::vector<int8_t>& match_cache,
                                         std::vector<bool>& matched) {
  if (idx >= patterns.size()) return true;
  constexpr int8_t kUnknown = -1;
  auto this_pattern = patterns[idx];
  for (size_t i = 0; i < fields.size(); ++i) {
    if (matched[i]) continue;
    const size_t table_idx = idx * fields.size() + i;
    match_cache[table_idx] =
        kUnknown ? VisitDFPattern(this_pattern, fields[i]) : match_cache[table_idx];
    if (match_cache[table_idx]) {
      // continue to match the rest;
      matched[i] = true;
      if (TryUnorderedMatch(idx + 1, patterns, fields, match_cache, matched)) return true;
      matched[i] = false;
    }
  }

  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const UnorderedTuplePatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);

  if (const auto* tuple_node = expr.as<TupleNode>()) {
    if (op->fields.size() == tuple_node->fields.size()) {
      constexpr int8_t kUnknown = -1;
      ICHECK_LE(op->fields.size(), std::numeric_limits<uint8_t>::max()) << "Too many fields!";
      // dynamic programming.
      std::vector<int8_t> match_cache(op->fields.size() * op->fields.size(), kUnknown);
      std::vector<bool> field_match_bitmap(op->fields.size(), false);
      return TryUnorderedMatch(0, op->fields, tuple_node->fields, match_cache, field_match_bitmap);
    }
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  auto expr_type = expr.as<ExprNode>()->checked_type();
  return (StructuralEqual()(op->type, expr_type)) && VisitDFPattern(op->pattern, expr);
}

static bool ShapeEqual(Analyzer* analyzer, const Array<PrimExpr>& lhs, const Array<PrimExpr>& rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (size_t i = 0; i < lhs.size(); ++i)
    if (!tir::is_one(analyzer->Simplify(lhs[i] == rhs[i]))) return false;
  return true;
}

bool DFPatternMatcher::VisitDFPattern_(const ShapePatternNode* op, const Expr& expr) {
  // no need to jump, as var.shape == value.shape
  if (const auto* tinfo = GetStructInfoAs<TensorStructInfoNode>(expr)) {
    if (const ShapeExprNode* shape_expr = tinfo->shape.as<ShapeExprNode>()) {
      return ShapeEqual(&analyzer_, op->shape, shape_expr->values) &&
             VisitDFPattern(op->pattern, expr);
    }
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const PrimArrPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  if (const ShapeExprNode* shape_expr = expr.as<ShapeExprNode>())
    return ShapeEqual(&analyzer_, op->fields, shape_expr->values);
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const DataTypePatternNode* op, const Expr& expr) {
  // no need to jump, as var.dtype == value.dtype
  auto expr_type = expr.as<ExprNode>()->checked_type();
  if (const DynTensorTypeNode* tensor_type = expr_type.as<DynTensorTypeNode>()) {
    return (StructuralEqual()(op->dtype, tensor_type->dtype)) && VisitDFPattern(op->pattern, expr);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const VarPatternNode* op, const Expr& expr) {
  // We don't jump for var pattern, as there's no need to access its value to judge it.
  if (const auto* var_node = expr.as<VarNode>()) {
    // "" means any name.
    return "" == op->name_hint() || op->name_hint() == var_node->name_hint();
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ExternFuncPatternNode* op, const Expr& expr0) {
  auto expr = TryGetValOfVar(expr0, var2val_);
  if (const auto* extern_fn = expr.as<ExternFuncNode>()) {
    return "" == op->global_symbol() || op->global_symbol() == extern_fn->global_symbol;
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr0) {
  // constants can be binded to relax.Var as well.
  auto expr = TryGetValOfVar(expr0, var2val_);
  return expr.as<ConstantNode>() != nullptr;
}

bool DFPatternMatcher::VisitDFPattern_(const DataflowVarPatternNode* op, const Expr& expr) {
  // DataflowVar is inherented from Var, so dispatch it to VarPattern.
  return expr->IsInstance<DataflowVarNode>() &&
         VisitDFPattern_(static_cast<const VarPatternNode*>(op), expr);
}

bool DFPatternMatcher::VisitDFPattern_(const GlobalVarPatternNode* op, const Expr& expr) {
  // GlobalVarPattern is not inherited from Var, so we need to handle it separately.
  if (const auto* var_node = expr.as<GlobalVarNode>())
    return "" == op->name_hint() || op->name_hint() == var_node->name_hint;
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const WildcardPatternNode* op, const Expr& expr) {
  return true;
}

Optional<Map<DFPattern, Expr>> ExtractMatchedExpr(DFPattern pattern, Expr expr,
                                                  Optional<Map<Var, Expr>> bindings_opt) {
  auto bindings = bindings_opt.value_or({});
  DFPatternMatcher matcher(bindings);

  if (!matcher.Match(pattern, expr)) {
    return NullOpt;
  }

  Map<DFPattern, Expr> matching;
  for (const auto& [pat, matches] : matcher.GetMemo()) {
    ICHECK_EQ(matches.size(), 1) << "More than one match for the pattern " << pat;
    matching.Set(pat, matches[0]);
  }
  return matching;
}

TVM_REGISTER_GLOBAL("relax.dpl.extract_matched_expr").set_body_typed(ExtractMatchedExpr);

bool MatchExpr(DFPattern pattern, Expr expr, Optional<Map<Var, Expr>> bindings_opt) {
  return static_cast<bool>(ExtractMatchedExpr(pattern, expr, bindings_opt));
}

TVM_REGISTER_GLOBAL("relax.dpl.match_expr").set_body_typed(MatchExpr);

struct PNode {
  const DFPatternNode* ptr;
  const VarNode* matched = nullptr;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> children;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> parents;
};

struct RNode {
  const VarNode* ptr;
  const DFPatternNode* matched = nullptr;
  std::vector<RNode*> children;
  std::vector<RNode*> parents;
};

/**
 * \brief This method try to match a real node and a pattern node along with its neighbors.
 */
static bool try_match(PNode* p, RNode* r, DFPatternMatcher* m,
                      const std::map<const VarNode*, std::set<const VarNode*>>& def2use,
                      const std::map<const VarNode*, std::vector<const VarNode*>>& use2def) {
  if (nullptr != p->matched && p->matched == r->ptr) return true;  // matched before.
  if (!m->Match(GetRef<DFPattern>(p->ptr), GetRef<Var>(r->ptr))) return false;

  std::stack<std::pair<PNode*, RNode*>> undo_stack{};

  const auto commit = [&undo_stack](PNode* p, RNode* r) {
    // match with each other.
    p->matched = r->ptr;
    r->matched = p->ptr;
    undo_stack.emplace(p, r);
  };

  const auto quit = [&undo_stack] {
    while (!undo_stack.empty()) {
      auto& top = undo_stack.top();
      top.first->matched = nullptr;
      top.second->matched = nullptr;
      undo_stack.pop();
    }
    return false;
  };

  commit(p, r);

  // match parent patterns.
  for (auto& pparent_pairs : p->parents) {
    PNode* pparent = pparent_pairs.first;
    const std::vector<PairCons>& constraints = pparent_pairs.second;

    bool any_cons_sat = false;
    for (auto& rparent : r->parents) {
      // skip if mismatch.
      if (rparent->matched && rparent->matched != pparent->ptr) continue;

      const auto& uses = def2use.at(rparent->ptr);
      // skip if `rparent` is not used by `r`.
      if (uses.cend() == uses.find(r->ptr)) continue;

      // check edge constraints.
      bool cons_sat = true;
      for (const auto& cons : constraints) {
        if (PairCons::kOnlyUsedBy == cons.type && uses.size() != 1) {
          cons_sat = false;
          break;
        }

        if (-1 != cons.index) {
          const auto& callees = use2def.at(r->ptr);
          if (static_cast<size_t>(cons.index) >= callees.size() ||
              rparent->ptr != callees[cons.index]) {
            cons_sat = false;
            break;
          }
        }
      }
      if (!cons_sat) continue;
      any_cons_sat = true;

      // try all parent R nodes that are not matched yet.
      // as long as ppattern can match one node.
      if (!pparent->matched && try_match(pparent, rparent, m, def2use, use2def)) {
        commit(pparent, rparent);
        break;
      }
    }
    if (!pparent->matched || !any_cons_sat) return quit();
  }

  // forward matching;
  for (auto& pchild_pairs : p->children) {
    PNode* pchild = pchild_pairs.first;
    const std::vector<PairCons>& constraints = pchild_pairs.second;
    bool any_cons_sat = false;
    for (auto& rchild : r->children) {
      if (rchild->matched && rchild->matched != pchild->ptr) continue;

      const auto& uses = def2use.at(r->ptr);
      if (uses.cend() == uses.find(rchild->ptr)) continue;

      // check edge constraints.
      bool all_cons_pass = true;
      for (const auto& cons : constraints) {
        if (PairCons::kOnlyUsedBy == cons.type && uses.size() != 1) {
          all_cons_pass = false;
          break;
        }

        if (-1 != cons.index) {
          const auto& callees = use2def.at(rchild->ptr);
          if (static_cast<size_t>(cons.index) >= callees.size() || r->ptr != callees[cons.index]) {
            all_cons_pass = false;
            break;
          }
        }
      }
      if (!all_cons_pass) continue;
      any_cons_sat = true;

      if (!pchild->matched && try_match(pchild, rchild, m, def2use, use2def)) {
        commit(pchild, rchild);
        break;
      }
    }
    if (!pchild->matched || !any_cons_sat) return quit();
  }

  return true;
}

class MatcherUseDefAnalysis : public relax::ExprVisitor {
 public:
  std::map<const VarNode*, std::set<const VarNode*>> def2use;
  // caller -> callee table.
  std::map<const VarNode*, std::vector<const VarNode*>> caller2callees;

  const VarNode* cur_user_;

  void VisitBinding_(const VarBindingNode* binding) override {
    // init
    cur_user_ = binding->var.get();
    this->VisitVarDef(binding->var);
    this->VisitExpr(binding->value);
    cur_user_ = nullptr;
  }

  void VisitExpr_(const VarNode* op) override {
    if (nullptr == cur_user_) return;

    def2use[op].insert(cur_user_);
    caller2callees[cur_user_].push_back(op);
  }

  void VisitExpr_(const DataflowVarNode* op) override {
    VisitExpr_(static_cast<const VarNode*>(op));
  }
};

Map<DFPattern, Var> MatchGraph(const PatternContext& ctx, const DataflowBlock& dfb,
                               Optional<Var> start_hint, bool must_include_hint) {
  Map<DFPattern, Var> ret;
  // TODO(@ganler): Handle non-may external use.
  ICHECK(ctx->allow_extern_use == PatternContextNode::kMay) << "Only kMay is supported yet.";
  ICHECK(!must_include_hint || start_hint.defined())
      << "must_include_hint is only supported with start_hint.";

  const auto var2val = AnalyzeVar2Value(dfb);
  DFPatternMatcher matcher(var2val);

  // std::map<const VarNode*, std::set<const VarNode*>>
  MatcherUseDefAnalysis ud_analysis;
  ud_analysis.VisitBindingBlock_(dfb.get());
  const auto& def2use = ud_analysis.def2use;
  const auto& caller2callees = ud_analysis.caller2callees;

  // First construct a graph of PNode and RNode.
  std::unordered_map<const VarNode*, RNode> var2node;
  var2node.reserve(dfb->bindings.size());

  for (const auto& du : def2use) {
    const VarNode* cur_var = du.first;
    const std::set<const VarNode*>& uses = du.second;
    RNode& cur_node = var2node[cur_var];
    cur_node.ptr = cur_var;
    for (const VarNode* use : uses) {
      auto& use_node = var2node[use];
      use_node.ptr = use;
      cur_node.children.push_back(&use_node);
      use_node.parents.push_back(&cur_node);
    }
  }

  std::unordered_map<const DFPatternNode*, PNode> pattern2node;
  pattern2node.reserve(ctx->constraints.size());

  for (const auto& def2use_pattern : ctx->constraints) {
    const DFPatternNode* def_pattern = def2use_pattern.first.get();
    const std::map<DFPattern, std::vector<PairCons>>& uses = def2use_pattern.second;
    PNode& def_node = pattern2node[def_pattern];
    def_node.ptr = def_pattern;
    def_node.children.reserve(uses.size());
    for (const auto& use : uses) {
      const auto& cons = use.second;
      const DFPatternNode* use_pattern = use.first.get();
      PNode& use_node = pattern2node[use_pattern];
      use_node.ptr = use_pattern;
      use_node.parents.emplace_back(&def_node, std::ref(cons));
      def_node.children.emplace_back(&use_node, std::ref(cons));
    }
  }

  if (start_hint.defined()) {
    Var v = start_hint.value();
    auto rnode_ptr = var2node.find(v.get());
    for (auto& ppair : pattern2node) {
      if (try_match(&ppair.second, &rnode_ptr->second, &matcher, def2use, caller2callees)) {
        for (auto ppair : pattern2node)
          ret.Set(GetRef<DFPattern>(ppair.first), GetRef<Var>(ppair.second.matched));
        return ret;
      }
    }

    if (must_include_hint) return ret;
  }

  PNode* pnode_start = &pattern2node.begin()->second;

  if (!pnode_start->matched) {
    for (auto& rpair : var2node) {
      if (start_hint.defined() && start_hint.value().get() == rpair.first) continue;
      if (try_match(pnode_start, &rpair.second, &matcher, def2use, caller2callees)) {
        for (auto ppair : pattern2node)
          ret.Set(GetRef<DFPattern>(ppair.first), GetRef<Var>(ppair.second.matched));

        return ret;
      }
    }
  }

  return ret;
}

TVM_REGISTER_GLOBAL("relax.dpl.match_dfb").set_body_typed(MatchGraph);

/*!
 * \brief Apply pattern matching to each call node and replace matching ones with the output of
 * a user-provided rewriter function.
 */
class PatternRewriter : ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  PatternRewriter(DFPattern pat, PackedFunc rewriter_func)
      : pattern_(pat), rewriter_func_(rewriter_func) {}

  static Expr Run(DFPattern pat, PackedFunc rewriter_func, Function f) {
    PatternRewriter rewriter(pat, rewriter_func);
    return RemoveAllUnused(Downcast<Function>(rewriter.VisitExpr(f)));
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    bindings_.Set(binding->var, binding->value);
    ExprMutator::VisitBinding_(binding);
    if (auto it = memo_.find(binding->value.get()); it != memo_.end()) {
      // We need to update the binding to pass to ExtractMatchedExpr, so that the rewritten
      // expression can be subject to further pattern matchings.
      bindings_.Set(binding->var, it->second);
    }
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    auto call = ExprMutator::VisitExpr_(call_node);
    if (auto matches_opt = ExtractMatchedExpr(pattern_, call, bindings_)) {
      auto rewriten_expr = rewriter_func_(call, matches_opt.value());
      memo_[call_node] = rewriten_expr;
      return rewriten_expr;
    }
    return call;
  }

 private:
  DFPattern pattern_;
  PackedFunc rewriter_func_;
  Map<Var, Expr> bindings_;
  std::unordered_map<const Object*, Expr> memo_;
};

TVM_REGISTER_GLOBAL("relax.dpl.rewrite")
    .set_body_typed([](DFPattern pat, PackedFunc rewriter, Function f) {
      return PatternRewriter::Run(pat, rewriter, f);
    });

}  // namespace relax
}  // namespace tvm
