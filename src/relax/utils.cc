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

  Expr VisitExpr_(const CallNode* op) final {
    auto call_node = Downcast<Call>(ExprMutator::VisitExpr_(op));

    // Special case for strided_slice
    //
    // The strided_slice operator currently stores the begins/ends in
    // the CallNode::attrs.  Because the CallNode::attrs is only
    // intended to store static information, any PrimExpr members in
    // the attributes are not visited by `ExprMutator::VisitPrimExpr`.
    // Therefore, these must be explicitly visited.
    //
    // When the strided_slice operator is updated to store begins/ends
    // as a tuple of `relax::PrimValue` in the arguments, this special
    // case can be removed.
    static auto strided_slice_op = Op::Get("relax.strided_slice");
    if (call_node->op.same_as(strided_slice_op)) {
      auto attrs = call_node->attrs.as<StridedSliceAttrs>();

      auto visit_prim_expr = [this](const auto& expr) { return VisitPrimExpr(expr); };

      Array<PrimExpr> begin = attrs->begin.Map(visit_prim_expr);
      Array<PrimExpr> end = attrs->end.Map(visit_prim_expr);
      auto strides = attrs->strides;
      if (strides.defined()) {
        strides = strides.value().Map(visit_prim_expr);
      }

      bool all_same = begin.same_as(attrs->begin) && end.same_as(attrs->end) &&
                      (!strides.defined() || strides.same_as(attrs->strides));
      if (!all_same) {
        ObjectPtr<StridedSliceAttrs> new_attrs = make_object<StridedSliceAttrs>();
        new_attrs->axes = attrs->axes;
        new_attrs->begin = std::move(begin);
        new_attrs->end = std::move(end);
        new_attrs->strides = std::move(strides);
        new_attrs->assume_inbound = attrs->assume_inbound;
        call_node.CopyOnWrite()->attrs = Attrs(new_attrs);
      }
    }

    return std::move(call_node);
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
