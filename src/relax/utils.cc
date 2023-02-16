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
  explicit ExprBinder(const tvm::Map<Var, Expr>& args_map) : args_map_(args_map) {}

  Expr VisitExpr_(const VarNode* op) final {
    auto id = GetRef<Var>(op);
    auto it = args_map_.find(id);
    if (it != args_map_.end()) {
      return (*it).second;
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

 private:
  const tvm::Map<Var, Expr>& args_map_;
};

/*!
 * \brief Bind params on expr
 * \param expr The expr where to bind params
 * \param args_map The map from param var to the expr it binds to
 * \return The result expr after bind params
 */
Expr Bind(const Expr& expr, const tvm::Map<Var, Expr>& args_map) {
  if (const FunctionNode* func = expr.as<FunctionNode>()) {
    Expr new_body = ExprBinder(args_map).VisitExpr(func->body);
    Array<Var> new_params;
    for (size_t i = 0; i < func->params.size(); ++i) {
      if (!args_map.count(func->params[i])) {
        new_params.push_back(func->params[i]);
      }
    }
    if (new_body.same_as(func->body) && new_params.size() == func->params.size()) {
      return expr;
    }
    // The checked_type_ of the new function is deduced from the function body
    // TODO(@relax-team): Should infer the shape from the body as well
    return Function(new_params, new_body, NullOpt, func->attrs);
  } else {
    return ExprBinder(args_map).VisitExpr(expr);
  }
}

bool IsBoolScalarType(const Type& ty, bool permit_unknown_rank, bool permit_unknown_dtype) {
  const DynTensorTypeNode* tt = ty.as<DynTensorTypeNode>();
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

}  // namespace relax
}  // namespace tvm
