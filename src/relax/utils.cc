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

#include "transform/utils.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/index.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/utils.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace relax {

/*! \brief Helper to implement bind params.*/
class ExprBinder : public ExprMutator {
 public:
  explicit ExprBinder(const tvm::ffi::Map<Var, Expr>& bindings) : bindings_(bindings) {}

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const ShapeExprNode* op) final {
    ffi::Array<PrimExpr> values =
        op->values.Map([this](const PrimExpr& value) { return BindShapeValue(value); });
    return values.same_as(op->values) ? ffi::GetRef<Expr>(op) : ShapeExpr(values, op->span);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    tvm::ffi::Array<Var> params;
    bool all_params_unchanged = true;
    for (const Var& param : op->params) {
      if (bindings_.count(param)) {
        all_params_unchanged = false;
      } else {
        Var new_param = this->VisitVarDef(param);
        params.push_back(new_param);
        if (!param.same_as(new_param)) {
          this->var_remap_[param] = new_param;
          all_params_unchanged = false;
        }
      }
    }

    Expr body = this->VisitWithNewScope(op->body, params);

    // FuncType does not depend on Expr
    if (all_params_unchanged && body.same_as(op->body)) {
      return ffi::GetRef<Expr>(op);
    } else {
      // purity won't be affected, no need to update annotation
      return Function(params, body, VisitExprDepTypeField(op->ret_ty), op->is_pure, op->attrs);
    }
  }

  Expr VisitExpr_(const VarNode* op) final {
    auto id = ffi::GetRef<Var>(op);
    auto it = bindings_.find(id);
    if (it != bindings_.end()) {
      return (*it).second;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr VisitTypePrimExprField(const PrimExpr& expr) final { return BindShapeValue(expr); }

  PrimExpr BindShapeValue(const PrimExpr& expr) {
    PrimExpr output = tirx::Substitute(expr, [this](const Var& var) -> ffi::Optional<Expr> {
      auto it = bindings_.find(var);
      if (it == bindings_.end()) return std::nullopt;
      if (auto value = (*it).second.as<PrimExpr>()) {
        return ffi::Optional<Expr>(*value);
      }
      return std::nullopt;
    });
    return output.same_as(expr) ? expr : analyzer_->Simplify(output);
  }

  const tvm::ffi::Map<Var, Expr>& bindings_;
  arith::Analyzer analyzer_;
};

/*!
 * \brief Bind params on expr
 * \param expr The expr where to bind params
 * \param binds The map from param var to the expr it binds to
 * \return The result expr after bind params
 */
Expr Bind(const Expr& expr, const tvm::ffi::Map<Var, Expr>& binds) {
  return ExprBinder(binds).VisitExpr(expr);
}

Type Bind(const Type& ty, const tvm::ffi::Map<Var, Expr>& binds) {
  return ExprBinder(binds).VisitExprDepTypeField(ty);
}

tvm::ffi::Map<Var, Expr> InferSymbolicVarMap(
    const tvm::ffi::Map<tvm::Var, relax::Expr>& relax_var_remap, const arith::Analyzer& analyzer) {
  (void)analyzer;
  tvm::ffi::Map<Var, Expr> var_remap = relax_var_remap;

  auto bind_from_prim_expr = [&var_remap](const PrimExpr& var_shape, const PrimExpr& expr_shape) {
    if (auto var = var_shape.as<tirx::PrimVar>()) {
      var_remap.Set(var.value(), expr_shape);
    }
  };

  auto bind_from_shape = [&bind_from_prim_expr](const Type& var, const Type& expr) {
    auto var_shape = var.as<ShapeTypeNode>();
    if (!var_shape) return;
    if (!var_shape->values.has_value()) return;

    auto expr_shape = expr.as<ShapeTypeNode>();
    if (!expr_shape) return;
    if (!expr_shape->values.has_value()) return;

    auto var_shape_arr = var_shape->values.value();
    auto expr_shape_arr = expr_shape->values.value();
    if (var_shape_arr.size() != expr_shape_arr.size()) return;
    for (size_t i = 0; i < var_shape_arr.size(); i++) {
      bind_from_prim_expr(var_shape_arr[i], expr_shape_arr[i]);
    }
  };

  auto bind_from_tensor = [&bind_from_shape](const Type& var, const Type& expr) {
    auto var_tensor = var.as<TensorTypeNode>();
    if (!var_tensor) return;
    if (!var_tensor->shape.has_value()) return;

    auto expr_tensor = expr.as<TensorTypeNode>();
    if (!expr_tensor) return;
    if (!expr_tensor->shape.has_value()) return;

    bind_from_shape(GetType(var_tensor->shape.value()), GetType(expr_tensor->shape.value()));
  };

  std::function<void(const Type&, const Type&)> bind_from_ty = nullptr;
  auto bind_from_tuple = [&bind_from_ty](const Type& var, const Type& expr) {
    auto var_tuple = var.as<TupleTypeNode>();
    if (!var_tuple) return;

    auto expr_tuple = expr.as<TupleTypeNode>();
    if (!expr_tuple) return;

    if (var_tuple->fields.size() != expr_tuple->fields.size()) return;

    for (size_t i = 0; i < var_tuple->fields.size(); i++) {
      bind_from_ty(var_tuple->fields[i], expr_tuple->fields[i]);
    }
  };

  bind_from_ty = [&](const Type& var, const Type& expr) {
    bind_from_tensor(var, expr);
    bind_from_shape(var, expr);
    bind_from_tuple(var, expr);
  };

  for (const auto& [relax_var, relax_expr] : relax_var_remap) {
    auto var_ty = GetType(relax_var);
    auto expr_ty = GetType(relax_expr);
    bind_from_ty(var_ty, expr_ty);
  }

  return var_remap;
}

bool IsBoolType(const Type& ty, bool permit_unknown_rank, bool permit_unknown_dtype) {
  DLDataType dtype;
  int ndim;

  if (const auto* tensor = ty.as<TensorTypeNode>()) {
    ndim = tensor->ndim;
    if (tensor->IsUnknownDtype()) {
      bool correct_rank = ndim == 0 || (permit_unknown_rank && ndim == -1);
      return permit_unknown_dtype && correct_rank;
    }
    dtype = tensor->dtype.value()->dtype;
  } else if (const auto* prim = ty.as<PrimTypeNode>()) {
    dtype = prim->dtype;
    ndim = 0;
  } else {
    return false;
  }

  // Bool-type matching uses element-code-only behavior; rank is checked separately.
  // Unknown dtype is already handled above via IsUnknownDtype().
  bool correct_dtype = dtype.code == DLDataTypeCode::kDLBool;
  bool correct_rank = ndim == 0 || (permit_unknown_rank && ndim == -1);
  return correct_dtype && correct_rank;
}

bool IsLeafOrTuple(const Expr& expr) {
  return !expr.as<CallNode>() && !expr.as<TupleGetItemNode>() && !expr.as<SeqExprNode>() &&
         !expr.as<IfNode>() && !expr.as<FunctionNode>();
}

bool IsImpureCall(const Call& call) {
  if (auto op_ptr = call->op.as<OpNode>()) {
    auto op = ffi::GetRef<Op>(op_ptr);
    static auto purity_map = Op::GetAttrMap<bool>("FPurity");
    if (purity_map.count(op)) {
      return !(purity_map[op]);
    }
    static auto effect_map = Op::GetAttrMap<tirx::TCallEffectKind>("TCallEffectKind");
    TVM_FFI_ICHECK(effect_map.count(op))
        << "Cannot find the registered purity or call effect of this op: " << op->name;
    auto effect = static_cast<tirx::CallEffectKind>(effect_map[op]);
    return effect > tirx::CallEffectKind::kPure;
  }
  // the Type must be FuncType
  auto func_ty = GetTypeAs<FuncTypeNode>(call->op);
  return !func_ty->purity;
}

Expr GetBoundValue(const Binding& b) {
  if (auto* var_binding = b.as<VarBindingNode>()) {
    return var_binding->value;
  } else if (auto* match_binding = b.as<MatchCastNode>()) {
    return match_binding->value;
  } else {
    TVM_FFI_ICHECK(false) << "Invalid binding (should never happen)";
  }
}

/*!
 * \brief Copy a new Relax function with new remapped vars and symbolic vars.
 * To get the var mapping from old vars to new vars, see FuncCopier in src/relax/transform/utils.h.
 * \param func The Relax function we want to copy.
 * \return The copied function.
 */
Function CopyWithNewVars(Function func) { return FunctionCopier().Copy(func); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.CopyWithNewVars", CopyWithNewVars);
}

}  // namespace relax
}  // namespace tvm
