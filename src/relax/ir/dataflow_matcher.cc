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

#include <tvm/arith/analyzer.h>
#include <tvm/node/structural_equal.h>
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
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../transform/utils.h"
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
  CHECK(pattern.defined()) << "Null pattern found when matching against " << expr0;

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

      // Commutative Matching.
      if (const OpNode* op_node = call_node->op.as<OpNode>()) {
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

std::tuple<PrimExpr, bool> SameShapeConstraintNode::AsPrimExpr(
    std::function<Optional<Var>(const DFPatternNode*)> match_state) const {
  Optional<Array<PrimExpr>> expected_shape;
  bool all_shapes_defined = true;

  // The expression that must be true in order
  PrimExpr all_dimensions_equal = Bool(true);

  for (const auto& arg : args) {
    if (auto opt_var = match_state(arg.get())) {
      auto var = opt_var.value();
      auto opt_var_shape = [&]() -> Optional<Array<PrimExpr>> {
        auto sinfo = GetStructInfo(var);
        if (auto tensor = sinfo.as<TensorStructInfoNode>()) {
          return tensor->GetShape();
        } else if (auto shape_expr = sinfo.as<ShapeStructInfoNode>()) {
          return shape_expr->values;
        } else {
          return NullOpt;
        }
      }();

      if (!opt_var_shape.defined()) {
        // The pattern has matched to something without a shape.
        // Therefore, it cannot have the same shape as something else.
        return {PrimExpr(Bool(false)), true};
      }
      auto var_shape = opt_var_shape.value();

      if (expected_shape.defined()) {
        auto prev_shape = expected_shape.value();
        if (prev_shape.size() == var_shape.size()) {
          // The dimensionalities match, so build up the expression
          // that must be true for the shapes to be equivalent.
          for (size_t i = 0; i < prev_shape.size(); i++) {
            all_dimensions_equal = all_dimensions_equal && (var_shape[i] == prev_shape[i]);
          }

        } else {
          // The shapes have different dimensionality.  No need to
          // perform potentially-expensive simplifications, because
          // the dimensions do not match.
          return {PrimExpr(Bool(false)), true};
        }

      } else {
        // This is the first pattern with a known match.  Store the
        // shape so it can be compared against later shapes.
        expected_shape = var_shape;
      }

    } else {
      // Missing an argument, so the constraint will either return
      // NullOpt or false at this point.  However, delay the return of
      // NullOpt until the end of the function, because we'd rather
      // return "false" if it possible to do so.
      all_shapes_defined = false;
    }
  }

  return {all_dimensions_equal, all_shapes_defined};
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
  if (const auto* var_node = expr.as<GlobalVarNode>()) {
    std::string pat = std::string(op->name_hint());
    std::string var_name = std::string(var_node->name_hint);
    return pat.empty() || var_name.find(pat) != std::string::npos;
  }
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

class MatcherUseDefAnalysis : public relax::ExprVisitor {
 public:
  std::vector<const VarNode*> vars;
  std::map<const VarNode*, std::vector<const VarNode*>> def2use;
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

    auto check_and_push = [](std::vector<const VarNode*>& vec, const VarNode* var) {
      if (std::find(vec.begin(), vec.end(), var) == vec.end()) {
        vec.push_back(var);
      }
    };

    check_and_push(def2use[op], cur_user_);
    check_and_push(vars, op);

    caller2callees[cur_user_].push_back(op);
  }
};

struct PNode {
  const DFPatternNode* ptr;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> children;
  std::vector<std::pair<PNode*, const std::vector<PairCons>&>> parents;
};

struct RNode {
  const VarNode* ptr;
  std::vector<RNode*> children;
  std::vector<RNode*> parents;
};

struct MatchState {
  void add(const PNode* p, const RNode* r) {
    match_p_r[p] = r;
    match_r_p[r] = p;
  }

  void add(const DFConstraintNode* constraint) { validated_constraints_.insert(constraint); }

  void add(MatchState&& other) {
    match_p_r.merge(std::move(other.match_p_r));
    match_r_p.merge(std::move(other.match_r_p));
    validated_constraints_.merge(other.validated_constraints_);
  }

  const VarNode* matched(const PNode* p) const {
    if (auto it = match_p_r.find(p); it != match_p_r.end()) {
      return it->second->ptr;
    }
    return nullptr;
  }

  const DFPatternNode* matched(const RNode* r) const {
    if (auto it = match_r_p.find(r); it != match_r_p.end()) {
      return it->second->ptr;
    }
    return nullptr;
  }

  const VarNode* matched(const PNode& p) const { return matched(&p); }
  const DFPatternNode* matched(const RNode& r) const { return matched(&r); }

  bool is_validated(const DFConstraintNode* constraint) const {
    return validated_constraints_.count(constraint);
  }

 private:
  std::unordered_map<const PNode*, const RNode*> match_p_r;
  std::unordered_map<const RNode*, const PNode*> match_r_p;
  std::unordered_set<const DFConstraintNode*> validated_constraints_;
};

/**
 * \brief This method try to match a real node and a pattern node along with its neighbors.
 */
static std::optional<MatchState> TryMatch(const PNode& p, const RNode& r,
                                          const MatchState& current_match, DFPatternMatcher* m,
                                          const MatcherUseDefAnalysis& ud_analysis) {
  if (!m->Match(GetRef<DFPattern>(p.ptr), GetRef<Var>(r.ptr))) return std::nullopt;

  MatchState new_match;

  new_match.add(&p, &r);

  // forward matching;
  for (const auto& [pchild, constraints] : p.children) {
    bool any_cons_sat = false;
    for (const auto& rchild : r.children) {
      if (new_match.matched(rchild)) {
        // The child variable is already matched to other child pattern in a previous iteration.
        continue;
      }
      if (auto v = current_match.matched(pchild); v && v != rchild->ptr) {
        // The child pattern is already matched to other variable in a earlier call to TryMatch.
        continue;
      }

      const auto& uses = ud_analysis.def2use.at(r.ptr);

      // check edge constraints.
      bool all_cons_pass = true;
      for (const auto& cons : constraints) {
        if (cons.type == PairCons::kOnlyUsedBy && uses.size() != 1) {
          all_cons_pass = false;
          break;
        }

        if (cons.index != -1) {
          const auto& callees = ud_analysis.caller2callees.at(rchild->ptr);
          if (callees.size() <= static_cast<size_t>(cons.index) || callees[cons.index] != r.ptr) {
            all_cons_pass = false;
            break;
          }
        }
      }
      if (!all_cons_pass || new_match.matched(pchild)) continue;
      any_cons_sat = true;

      if (auto match_rec = TryMatch(*pchild, *rchild, current_match, m, ud_analysis)) {
        new_match.add(pchild, rchild);
        new_match.add(std::move(*match_rec));
      }
    }
    if (!new_match.matched(pchild) || !any_cons_sat) return std::nullopt;
  }

  return new_match;
}

static std::optional<MatchState> TryValidate(
    const MatchState& current_match,
    const std::unordered_map<const DFPatternNode*, PNode>& pattern2node,
    const std::vector<DFConstraint>& validation_constraints, arith::Analyzer* analyzer) {
  MatchState new_match;

  std::function<Optional<Var>(const DFPatternNode*)> query_match_state =
      [&pattern2node, &current_match](const DFPatternNode* pattern) -> Optional<Var> {
    auto it = pattern2node.find(pattern);
    ICHECK(it != pattern2node.end())
        << "DFConstraint attempted to access DFPattern " << GetRef<DFPattern>(pattern)
        << ", which does not appear in the PatternContext";
    const auto& p_node = it->second;
    if (auto ptr = current_match.matched(p_node)) {
      return GetRef<Var>(ptr);
    } else {
      return NullOpt;
    }
  };

  for (const auto& constraint : validation_constraints) {
    if (!current_match.is_validated(constraint.get())) {
      auto [necessary_condition, is_sufficient] = constraint->AsPrimExpr(query_match_state);

      necessary_condition = analyzer->Simplify(necessary_condition);
      const auto* known = tir::as_const_int(necessary_condition);

      if (known && *known && is_sufficient) {
        // The condition passes, and the expression provided is both
        // necessary and sufficient for the constraint to pass.  Mark
        // the constraint as passing, to avoid re-checking it unless
        // we backtrack.
        new_match.add(constraint.get());
      } else if (known && !*known) {
        // The condition fails.  Even if additional information would
        // be required to pass a constraint, it may bail out early as
        // a failure (e.g. shape mismatch in the first two items out
        // of N shapes that must all match).
        return std::nullopt;
      } else if (is_sufficient) {
        // The condition depends on dynamic parameters.  In the
        // future, this may be exposed to the user as a condition for
        // optimization, or can be combined with the conditions
        // provided from other constraints.
        return std::nullopt;
      }
    }
  }

  return new_match;
}

static std::optional<MatchState> MatchTree(
    const MatchState& current_match, size_t current_root_idx,
    const std::unordered_map<const DFPatternNode*, PNode>& pattern2node,
    const std::unordered_map<const VarNode*, RNode>& var2node, DFPatternMatcher* matcher,
    const std::vector<DFPattern>& roots, const std::vector<DFConstraint>& validation_constraints,
    const MatcherUseDefAnalysis& ud_analysis, arith::Analyzer* analyzer) {
  auto get_next_root = [&](size_t root_idx) -> const PNode* {
    // Look for the next unmatched root node.
    for (; root_idx < roots.size(); ++root_idx) {
      const auto& root = pattern2node.at(roots[root_idx].get());
      if (!current_match.matched(root)) {
        return &root;
      }
    }
    return nullptr;
  };

  const auto root = get_next_root(current_root_idx);

  if (!root) {
    // All root nodes have been matched
    return current_match;
  }

  MatchState new_match = current_match;

  for (const auto& var : ud_analysis.vars) {
    const RNode& r_node = var2node.at(var);
    if (new_match.matched(r_node)) continue;
    if (auto match = TryMatch(*root, r_node, new_match, matcher, ud_analysis)) {
      // Recursively try to match the next subtree.
      new_match.add(std::move(*match));
      if (auto validation =
              TryValidate(new_match, pattern2node, validation_constraints, analyzer)) {
        new_match.add(std::move(*validation));
        if (auto match_rec =
                MatchTree(new_match, current_root_idx + 1, pattern2node, var2node, matcher, roots,
                          validation_constraints, ud_analysis, analyzer)) {
          new_match.add(std::move(*match_rec));
          return new_match;
        }
      }
      // Recursive matching has failed, backtrack.
      new_match = current_match;
      continue;
    }
  }

  return std::nullopt;
}

Optional<Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx, const DataflowBlock& dfb,
                                         const Map<Var, Expr>& bindings) {
  // TODO(@ganler): Handle non-may external use.
  ICHECK(ctx->allow_extern_use == PatternContextNode::kMay) << "Only kMay is supported yet.";
  DFPatternMatcher matcher(bindings);

  MatcherUseDefAnalysis ud_analysis;
  ud_analysis.VisitBindingBlock_(dfb.get());

  // First construct a graph of PNode and RNode.
  std::unordered_map<const VarNode*, RNode> var2node;
  var2node.reserve(dfb->bindings.size());

  for (const VarNode* cur_var : ud_analysis.vars) {
    const auto& uses = ud_analysis.def2use.at(cur_var);
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
  pattern2node.reserve(ctx->edge_constraints.size());

  for (const auto& def_pattern : ctx->src_ordered) {
    PNode& def_node = pattern2node[def_pattern.get()];
    const auto& uses = ctx->edge_constraints.at(def_pattern);
    def_node.ptr = def_pattern.get();
    def_node.children.reserve(uses.size());
    for (const auto& [use_pattern, cons] : uses) {
      PNode& use_node = pattern2node[use_pattern.get()];
      use_node.ptr = use_pattern.get();
      use_node.parents.emplace_back(&def_node, std::ref(cons));
      def_node.children.emplace_back(&use_node, std::ref(cons));
    }
  }

  std::vector<DFPattern> roots;
  for (const auto& pat : ctx->src_ordered) {
    if (pattern2node[pat.get()].parents.empty()) {
      roots.push_back(pat);
    }
  }

  if (roots.empty()) {
    return NullOpt;
  }

  arith::Analyzer analyzer;
  auto match = MatchTree({}, 0, pattern2node, var2node, &matcher, roots,
                         ctx->validation_constraints, ud_analysis, &analyzer);
  if (!match) {
    return NullOpt;
  }

  Map<DFPattern, Var> ret;
  for (const auto& [pat, p_node] : pattern2node) {
    ICHECK(match->matched(p_node));
    ret.Set(GetRef<DFPattern>(pat), GetRef<Var>(match->matched(p_node)));
  }
  return ret;
}

Optional<Map<DFPattern, Var>> MatchGraph(const PatternContext& ctx, const DataflowBlock& dfb) {
  return MatchGraph(ctx, dfb, AnalyzeVar2Value(dfb));
}

TVM_REGISTER_GLOBAL("relax.dpl.match_dfb")
    .set_body_typed([](const PatternContext& ctx, const DataflowBlock& dfb) {
      return MatchGraph(ctx, dfb);
    });

/*!
 * \brief Apply pattern matching to each call node and dataflow block, and replace matching ones
 * with the output of a user-provided rewriter function.
 */
class PatternRewriter : ExprMutator {
 public:
  using ExprMutator::VisitBindingBlock_;
  using ExprMutator::VisitExpr_;

  PatternRewriter(DFPattern pat, PackedFunc rewriter_func,
                  const std::unordered_set<const VarNode*>& params)
      : pattern_(pat), rewriter_func_(rewriter_func), params_(params) {}

  PatternRewriter(const PatternContext& ctx, PackedFunc rewriter_func,
                  const std::unordered_set<const VarNode*>& params)
      : ctx_(ctx), rewriter_func_(rewriter_func), params_(params) {}

  template <typename PatternType>
  static Function Run(PatternType pat, PackedFunc rewriter_func, Function f) {
    std::unordered_set<const VarNode*> params;
    for (const auto& p : f->params) {
      params.insert(p.get());
    }
    PatternRewriter rewriter(pat, rewriter_func, params);
    return Downcast<Function>(RemoveAllUnused(rewriter.VisitExpr(f)));
  }

  Expr VisitExpr_(const SeqExprNode* seq) override {
    if (ctx_) {
      return ExprMutator::VisitExpr_(seq);
    }

    auto cache = bindings_;
    SeqExpr prev = GetRef<SeqExpr>(seq);

    StructuralEqual struct_equal;

    while (true) {
      SeqExpr next = Downcast<SeqExpr>(builder_->Normalize(ExprMutator::VisitExpr_(prev.get())));
      if (struct_equal(prev, next)) {
        return std::move(next);
      }

      // Canonicalization may result in two previously-different
      // expressions being recognized as identical.  Elimination of
      // common subexpressions may result in trival var-to-var
      // bindings that can be canonicalized.  Therefore, iterate the
      // simplification steps until converged.
      while (true) {
        auto start_of_loop = next;
        next = Downcast<SeqExpr>(CanonicalizeBindings(next));
        next = Downcast<SeqExpr>(EliminateCommonSubexpr(next));
        next = Downcast<SeqExpr>(RemoveAllUnused(next));
        if (struct_equal(start_of_loop, next)) {
          break;
        }
      }

      if (struct_equal(prev, next)) {
        return std::move(next);
      }

      // Reset all knowledge of bindings that were collected from
      // this DataflowBlock.  The collected bindings are only after
      // the point where they were collected, and we are repeating
      // the mutation of this DataflowBlock.
      bindings_ = cache;
      prev = next;
    }
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block_node) override {
    if (ctx_) {
      return RewriteDataflowBlockFixedPoint(GetRef<DataflowBlock>(block_node));
    } else {
      return ExprMutator::VisitBindingBlock_(block_node);
    }
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    auto expr = VisitExpr(binding->value);
    bindings_.Set(binding->var, expr);
    ReEmitBinding(binding, expr);
  }

  Expr VisitExpr(const Expr& expr) override {
    auto node = ExprMutator::VisitExpr(expr);

    if (pattern_) {
      if (auto matches_opt = ExtractMatchedExpr(pattern_.value(), node, bindings_)) {
        Expr rewritten_expr = rewriter_func_(node, matches_opt.value());
        if (!rewritten_expr.same_as(node)) {
          return builder_->Normalize(rewritten_expr);
        }
      }
    }
    return node;
  }

 private:
  void EmitUsedVars(Expr val, const Array<Binding>& pending_bindings,
                    std::unordered_set<const VarNode*>* emitted_vars) {
    std::unordered_set<const VarNode*> unemitted_vars;
    PostOrderVisit(val, [=, &unemitted_vars](Expr e) {
      if (auto v = e.as<VarNode>(); v && !emitted_vars->count(v)) {
        unemitted_vars.insert(v);
      }
    });

    if (unemitted_vars.empty()) {
      return;
    }

    size_t num_unemitted = unemitted_vars.size();
    for (size_t i = 0; i < pending_bindings.size(); ++i) {
      const auto& binding = pending_bindings[i];
      if (auto var_bind = binding.as<VarBindingNode>();
          var_bind && unemitted_vars.count(var_bind->var.get())) {
        // var_bind->value may also depend on other unemitted vars in this range
        Array<Binding> prev_bindings(pending_bindings.begin(), pending_bindings.begin() + i);
        EmitUsedVars(var_bind->value, prev_bindings, emitted_vars);
        this->VisitBinding(binding);
        emitted_vars->insert(var_bind->var.get());
        if (--num_unemitted == 0) {
          return;
        }
      }
    }
  }

  // Repeat until all matchable subsets of bindings are rewritten.
  BindingBlock RewriteDataflowBlockFixedPoint(BindingBlock block) {
    auto df_block = Downcast<DataflowBlock>(block);
    Map<Var, Expr> bindings = AnalyzeVar2Value(df_block);
    if (auto matches = MatchGraph(ctx_.value(), df_block, bindings)) {
      builder_->BeginDataflowBlock();
      Map<Var, Expr> replacements = rewriter_func_(matches.value(), bindings);

      std::unordered_set<const VarNode*> emitted_vars;

      bool changed = false;
      for (size_t i = 0; i < block->bindings.size(); ++i) {
        const auto& binding = block->bindings[i];
        if (auto var_bind = binding.as<VarBindingNode>()) {
          if (auto new_val = replacements.Get(var_bind->var).value_or(var_bind->value);
              !StructuralEqual()(var_bind->value, new_val)) {
            Array<Binding> pending_bindings(block->bindings.begin() + i + 1, block->bindings.end());
            // Make sure there is no unbound variable used in the new value before it is emitted
            EmitUsedVars(new_val, pending_bindings, &emitted_vars);
            this->ReEmitBinding(var_bind, builder_->Normalize(new_val));
            changed = true;
          } else if (!emitted_vars.count(var_bind->var.get())) {
            this->VisitBinding(binding);
            emitted_vars.insert(var_bind->var.get());
          }
        } else {
          this->VisitBinding(binding);
        }
      }

      auto new_block = builder_->EndBlock();

      if (!changed) return new_block;
      return RewriteDataflowBlockFixedPoint(new_block);
    }
    return block;
  }

  /*! \brief The pattern for rewriting call nodes */
  Optional<DFPattern> pattern_;
  /*! \brief The pattern constraint contexts for rewriting dataflow blocks */
  Optional<PatternContext> ctx_;
  /*!
   * \brief The user-provided rewriter function. Its signature and semantics are:
   * - (Call, Map<DFPattern, Expr>) -> Call for call node rewriting. Given the matched
   *    call node and the map of patterns and matched expressions, it should return a new call node
   *    to replace the original one or the original matched call node as is.
   * - (Map<DFPattern, Var>, Map<Var, Expr>) -> Map<Var, Expr> for dataflow block rewriting.
   *    Given the map of patterns and corresponding variables (bound variables or parameters),
   *    it should return a map that specifies new values for matched bound variables. It can refer
   *    to the passed bindings to create the replacement expressions.
   */
  PackedFunc rewriter_func_;
  std::unordered_set<const VarNode*> params_;
  Map<Var, Expr> bindings_;
};

Function RewriteBindings(const PatternContext& ctx, PackedFunc rewriter, Function f) {
  return PatternRewriter::Run(ctx, rewriter, f);
}

TVM_REGISTER_GLOBAL("relax.dpl.rewrite_bindings").set_body_typed(RewriteBindings);

Function RewriteCall(const DFPattern& pat,
                     TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)> rewriter, Function f) {
  return PatternRewriter::Run(pat, rewriter, f);
}

TVM_REGISTER_GLOBAL("relax.dpl.rewrite_call").set_body_typed(RewriteCall);

}  // namespace relax
}  // namespace tvm
