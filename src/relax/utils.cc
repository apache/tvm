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

#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/index.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

/*! \brief Helper to implement bind params.*/
class ExprBinder : public ExprMutator {
 public:
  explicit ExprBinder(const tvm::Map<Var, Expr>& args_map,
                      const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map)
      : args_map_(args_map), symbolic_var_map_(symbolic_var_map) {}

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionNode* op) final {
    tvm::Array<Var> params;
    bool all_params_unchanged = true;
    for (const Var& param : op->params) {
      if (args_map_.count(param)) {
        all_params_unchanged = false;
      } else {
        Var new_param = this->VisitVarDef(param);
        params.push_back(new_param);
        if (!param.same_as(new_param)) {
          this->var_remap_[param->vid] = new_param;
          all_params_unchanged = false;
        }
      }
    }

    Expr body = this->VisitWithNewScope(op->body, params);

    // FuncStructInfo does not depend on Expr
    if (all_params_unchanged && body.same_as(op->body)) {
      return GetRef<Expr>(op);
    } else {
      // purity won't be affected, no need to update annotation
      return Function(params, body, VisitExprDepStructInfoField(op->ret_struct_info), op->is_pure,
                      op->attrs);
    }
  }

  Expr VisitExpr_(const VarNode* op) final {
    auto id = GetRef<Var>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  PrimExpr VisitPrimExpr(const PrimExpr& expr) final {
    auto new_expr = tir::Substitute(expr, symbolic_var_map_);
    if (!expr.same_as(new_expr)) {
      arith::Analyzer analyzer;
      new_expr = analyzer.Simplify(new_expr);
    }
    return new_expr;
  }

 private:
  const tvm::Map<Var, Expr>& args_map_;
  const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map_;
};

/*!
 * \brief Bind params on expr
 * \param expr The expr where to bind params
 * \param binds The map from param var to the expr it binds to
 * \param symbolic_var_map The map from symbolic var to the expr it binds to
 * \return The result expr after bind params
 */
Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& binds,
          const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map) {
  return ExprBinder(binds, symbolic_var_map).VisitExpr(expr);
}

StructInfo Bind(const StructInfo& sinfo, const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map) {
  return ExprBinder({}, symbolic_var_map).VisitExprDepStructInfoField(sinfo);
}

tvm::Map<tir::Var, PrimExpr> InferSymbolicVarMap(
    const tvm::Map<relax::Var, relax::Expr>& relax_var_remap, arith::Analyzer* analyzer) {
  tvm::Map<tir::Var, PrimExpr> tir_var_remap;

  auto bind_from_prim_expr = [&tir_var_remap](const PrimExpr& var_shape,
                                              const PrimExpr& expr_shape) {
    if (auto var = var_shape.as<tir::Var>()) {
      tir_var_remap.Set(var.value(), expr_shape);
    }
  };

  auto bind_from_prim_value = [&bind_from_prim_expr](const StructInfo& var,
                                                     const StructInfo& expr) {
    auto var_sinfo = var.as<PrimStructInfoNode>();
    if (!var_sinfo) return;

    auto expr_sinfo = expr.as<PrimStructInfoNode>();
    if (!expr_sinfo) return;

    if (!var_sinfo->value.defined() || !expr_sinfo->value.defined()) return;

    bind_from_prim_expr(var_sinfo->value.value(), expr_sinfo->value.value());
  };

  auto bind_from_shape = [&bind_from_prim_expr](const StructInfo& var, const StructInfo& expr) {
    auto var_shape = var.as<ShapeStructInfoNode>();
    if (!var_shape) return;
    if (!var_shape->values.defined()) return;

    auto expr_shape = expr.as<ShapeStructInfoNode>();
    if (!expr_shape) return;
    if (!expr_shape->values.defined()) return;

    auto var_shape_arr = var_shape->values.value();
    auto expr_shape_arr = expr_shape->values.value();
    if (var_shape_arr.size() != expr_shape_arr.size()) return;
    for (size_t i = 0; i < var_shape_arr.size(); i++) {
      bind_from_prim_expr(var_shape_arr[i], expr_shape_arr[i]);
    }
  };

  auto bind_from_tensor = [&bind_from_shape](const StructInfo& var, const StructInfo& expr) {
    auto var_tensor = var.as<TensorStructInfoNode>();
    if (!var_tensor) return;
    if (!var_tensor->shape.defined()) return;

    auto expr_tensor = expr.as<TensorStructInfoNode>();
    if (!expr_tensor) return;
    if (!expr_tensor->shape.defined()) return;

    bind_from_shape(GetStructInfo(var_tensor->shape.value()),
                    GetStructInfo(expr_tensor->shape.value()));
  };

  std::function<void(const StructInfo&, const StructInfo&)> bind_from_struct_info = nullptr;
  auto bind_from_tuple = [&bind_from_struct_info](const StructInfo& var, const StructInfo& expr) {
    auto var_tuple = var.as<TupleStructInfoNode>();
    if (!var_tuple) return;

    auto expr_tuple = expr.as<TupleStructInfoNode>();
    if (!expr_tuple) return;

    if (var_tuple->fields.size() != expr_tuple->fields.size()) return;

    for (size_t i = 0; i < var_tuple->fields.size(); i++) {
      bind_from_struct_info(var_tuple->fields[i], expr_tuple->fields[i]);
    }
  };

  bind_from_struct_info = [&](const StructInfo& var, const StructInfo& expr) {
    bind_from_tensor(var, expr);
    bind_from_shape(var, expr);
    bind_from_prim_value(var, expr);
    bind_from_tuple(var, expr);
  };

  for (const auto& [relax_var, relax_expr] : relax_var_remap) {
    auto var_sinfo = GetStructInfo(relax_var);
    auto expr_sinfo = GetStructInfo(relax_expr);
    bind_from_struct_info(var_sinfo, expr_sinfo);
  }

  return tir_var_remap;
}

bool IsBoolStructInfo(const StructInfo& sinfo, bool permit_unknown_rank,
                      bool permit_unknown_dtype) {
  DataType dtype;
  int ndim;

  if (const auto* tensor = sinfo.as<TensorStructInfoNode>()) {
    dtype = tensor->dtype;
    ndim = tensor->ndim;
  } else if (const auto* prim = sinfo.as<PrimStructInfoNode>()) {
    dtype = prim->dtype;
    ndim = 0;
  } else {
    return false;
  }

  bool correct_dtype = dtype.is_bool() || (permit_unknown_dtype && dtype.is_void());
  bool correct_rank = ndim == 0 || (permit_unknown_rank && ndim == -1);
  return correct_dtype && correct_rank;
}

bool IsLeafOrTuple(const Expr& expr) {
  return expr.as<LeafExprNode>() || expr.as<GlobalVarNode>() || expr.as<ExternFuncNode>() ||
         expr.as<OpNode>() || expr.as<TupleNode>();
}

bool IsImpureCall(const Call& call) {
  if (auto op_ptr = call->op.as<OpNode>()) {
    auto op = GetRef<Op>(op_ptr);
    static auto purity_map = Op::GetAttrMap<Bool>("FPurity");
    ICHECK(purity_map.count(op)) << "Cannot find the registered purity of this op: " << op->name;
    return !(purity_map[op]->value);
  }
  // the StructInfo must be FuncStructInfo
  auto func_struct_info = GetStructInfoAs<FuncStructInfoNode>(call->op);
  return !func_struct_info->purity;
}

Expr GetBoundValue(const Binding& b) {
  if (auto* var_binding = b.as<VarBindingNode>()) {
    return var_binding->value;
  } else if (auto* match_binding = b.as<MatchCastNode>()) {
    return match_binding->value;
  } else {
    CHECK(false) << "Invalid binding (should never happen)";
  }
}

/*!
 * \brief Copy a new Relax function with new remapped vars and symbolic vars.
 * To get the var mapping from old vars to new vars, see FuncCopier in src/relax/transform/utils.h.
 * \param func The Relax function we want to copy.
 * \return The copied function.
 */
Function CopyWithNewVars(Function func) { return FunctionCopier().Copy(func); }

TVM_REGISTER_GLOBAL("relax.CopyWithNewVars").set_body_typed(CopyWithNewVars);

}  // namespace relax
}  // namespace tvm
