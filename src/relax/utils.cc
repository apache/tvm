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

#include <tvm/relax/expr_functor.h>

namespace tvm {
namespace relax {

/*! \brief Helper to implement bind params.*/
class ExprBinder : public ExprMutator {
 public:
  explicit ExprBinder(const tvm::Map<Var, Expr>& args_map,
                      const tvm::Map<tir::Var, PrimExpr>& symbolic_var_map)
      : args_map_(args_map), symbolic_var_map_(symbolic_var_map) {}

 private:
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
      return Function(params, body, VisitExprDepStructInfoField(op->ret_struct_info), op->attrs);
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
    if (const tir::VarNode* var = expr.as<tir::VarNode>()) {
      auto it = symbolic_var_map_.find(GetRef<tir::Var>(var));
      if (it != symbolic_var_map_.end()) {
        return (*it).second;
      }
    }
    return ExprMutator::VisitPrimExpr(expr);
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

bool IsBoolStructInfo(const StructInfo& sinfo, bool permit_unknown_rank,
                      bool permit_unknown_dtype) {
  const TensorStructInfoNode* tt = sinfo.as<TensorStructInfoNode>();
  if (!tt) {
    return false;
  }
  bool correct_dtype = tt->dtype.is_bool() || (permit_unknown_dtype && tt->dtype.is_void());
  bool correct_rank = tt->ndim == 0 || (permit_unknown_rank && tt->ndim == -1);
  return correct_dtype && correct_rank;
}

bool IsLeafOrTuple(const Expr& expr) {
  return expr.as<LeafExprNode>() || expr.as<GlobalVarNode>() || expr.as<ExternFuncNode>() ||
         expr.as<OpNode>() || expr.as<TupleNode>();
}

class FunctionCopier : public ExprMutator {
 public:
  static Function Transform(Function func) {
    FunctionCopier copier;
    // All variables that are bound inside the original function would be copied
    // to satisfy the restriction in the well-formed check: Variables in Relax
    // must be bound exactly once.
    return Downcast<Function>(copier.VisitExpr(func));
  }

  Var VisitVarDef_(const DataflowVarNode* var) override {
    Var new_var = ExprMutator::VisitVarDef_(var);
    Var copied_var = DataflowVar(new_var->name_hint(), GetStructInfo(new_var), new_var->span);
    var_remap_[var->vid] = copied_var;
    return copied_var;
  }

  Var VisitVarDef_(const VarNode* var) override {
    Var new_var = ExprMutator::VisitVarDef_(var);
    Var copied_var = Var(new_var->name_hint(), GetStructInfo(new_var), new_var->span);
    var_remap_[var->vid] = copied_var;
    return copied_var;
  }
};

Function CopyWithNewVars(Function func) { return FunctionCopier::Transform(func); }

TVM_REGISTER_GLOBAL("relax.CopyWithNewVars").set_body_typed(CopyWithNewVars);

}  // namespace relax
}  // namespace tvm
