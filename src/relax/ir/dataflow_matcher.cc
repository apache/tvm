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

#include "dataflow_matcher.h"

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
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../arith/constraint_extract.h"
#include "../transform/utils.h"

namespace tvm {
namespace relax {

using tvm::arith::Analyzer;

// Pattern Matcher
bool DFPatternMatcher::Match(const DFPattern& pattern, const Expr& expr) {
  memo_.clear();
  matched_nodes_.clear();
  return VisitDFPattern(pattern, expr);
}

Expr DFPatternMatcher::UnwrapBindings(Expr expr, const Map<Var, Expr>& var2val) {
  auto unwrap = [&](Expr expr) -> Optional<Expr> {
    // Unwrap variables into the value to which they are bound.
    if (var2val.size()) {
      if (const VarNode* var = expr.as<VarNode>()) {
        if (auto may = var2val.Get(GetRef<Var>(var))) {
          return may.value();
        }
      }
    }

    // Unwrap SeqExpr with no bindings.  These can occur due to Relax
    // IR constraints for the bodies of Function and If nodes.
    if (auto seq = expr.as<SeqExprNode>()) {
      if (seq->blocks.empty()) {
        return seq->body;
      }
    }

    return NullOpt;
  };

  while (auto unwrapped = unwrap(expr)) {
    expr = unwrapped.value();
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

  auto expr = UnwrapBindings(expr0, var2val_);
  if (memoize_ && memo_.count(pattern)) {
    return expr.same_as(memo_[pattern]);
  } else {
    PrimExpr cached_condition = symbolic_expr_condition_;
    size_t watermark = matched_nodes_.size();
    bool out = DFPatternFunctor::VisitDFPattern(pattern, expr);
    if (out) {
      memo_[pattern] = expr;
      matched_nodes_.push_back(pattern);
    } else {
      ClearMap(watermark);
      symbolic_expr_condition_ = cached_condition;
    }
    return out;
  }
}

bool DFPatternMatcher::VisitDFPattern_(const OrPatternNode* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  return VisitDFPattern(op->left, expr) || VisitDFPattern(op->right, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const AndPatternNode* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
  return VisitDFPattern(op->left, expr) && VisitDFPattern(op->right, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const NotPatternNode* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
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
  auto expr = UnwrapBindings(expr0, var2val_);
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
  auto expr = UnwrapBindings(expr0, var2val_);
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
  auto expr = UnwrapBindings(expr0, var2val_);
  return StructuralEqual()(op->expr, expr);
}

bool DFPatternMatcher::VisitDFPattern_(const FunctionPatternNode* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
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
  auto expr = UnwrapBindings(expr0, var2val_);
  if (const auto* tuple_get_item_node = expr.as<TupleGetItemNode>()) {
    return (op->index == -1 || op->index == tuple_get_item_node->index) &&
           VisitDFPattern(op->tuple, tuple_get_item_node->tuple);
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const TuplePatternNode* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
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
  auto expr = UnwrapBindings(expr0, var2val_);

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

bool DFPatternMatcher::VisitDFPattern_(const StructInfoPatternNode* op, const Expr& expr0) {
  if (!VisitDFPattern(op->pattern, expr0)) {
    return false;
  }

  auto expr = UnwrapBindings(expr0, var2val_);
  auto expr_struct_info = GetStructInfo(expr);

  PrimExpr new_constraint = StructInfoBaseCheckPrecondition(op->struct_info, expr_struct_info);
  if (auto* as_int = new_constraint.as<IntImmNode>()) {
    return as_int->value;
  }

  symbolic_expr_condition_ = SimplifyCondition(symbolic_expr_condition_ && new_constraint);

  if (auto* as_int = symbolic_expr_condition_.as<IntImmNode>()) {
    return as_int->value;
  } else {
    return true;
  }
}

PrimExpr DFPatternMatcher::SimplifyCondition(PrimExpr condition) {
  if (condition->IsInstance<IntImmNode>()) {
    return condition;
  }

  std::vector<PrimExpr> constraints = arith::ExtractConstraints(condition, false);
  if (constraints.size() == 1) {
    return condition;
  }

  auto sort_key = [](PrimExpr expr) -> String {
    if (const auto* equal = expr.as<tir::EQNode>()) {
      if (const auto* var = equal->a.as<tir::VarNode>()) {
        return var->name_hint;
      }
    }
    return "";
  };
  std::stable_sort(
      constraints.begin(), constraints.end(),
      [&sort_key](const PrimExpr& a, const PrimExpr& b) { return sort_key(a) < sort_key(b); });

  PrimExpr sorted_condition = Bool(true);
  for (const PrimExpr& constraint : constraints) {
    sorted_condition = sorted_condition && constraint;
  }

  return analyzer_.Simplify(sorted_condition);
}

bool DFPatternMatcher::VisitDFPattern_(const TypePatternNode* op, const Expr& expr0) {
  auto expr = UnwrapBindings(expr0, var2val_);
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
  auto expr = UnwrapBindings(expr0, var2val_);
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
  auto expr = UnwrapBindings(expr0, var2val_);
  if (const auto* extern_fn = expr.as<ExternFuncNode>()) {
    return "" == op->global_symbol() || op->global_symbol() == extern_fn->global_symbol;
  }
  return false;
}

bool DFPatternMatcher::VisitDFPattern_(const ConstantPatternNode* op, const Expr& expr0) {
  // constants can be binded to relax.Var as well.
  auto expr = UnwrapBindings(expr0, var2val_);
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

}  // namespace relax
}  // namespace tvm
