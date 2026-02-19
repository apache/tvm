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
 * \file src/relax/transform/canonicalize_shape_expr.cc
 * \brief Canonicalize ShapeExpr by replacing compound PrimExpr with fresh symbolic variables.
 *
 * VMShapeLower can only handle expressions where each PrimExpr dimension is either:
 * - IntImm (concrete integer constant)
 * - tir::Var (symbolic variable from function parameters or match_cast)
 *
 * This pass transforms compound PrimExpr (e.g., n+1, 4*n*m) in ShapeExpr and struct_info by:
 * 1. Creating a fresh tir::Var for each compound expression
 * 2. Emitting a MatchCast that binds the fresh var to a PrimValue computing the expression
 * 3. Replacing the compound expression with the fresh var everywhere (ShapeExpr and struct_info)
 *
 * Example transformation:
 *   Before: y = R.Tensor((n + 1,)) = R.zeros(R.shape([n + 1]), dtype="float32")
 *   After:  _s0_pv: R.Prim(value=_s0) = R.match_cast(R.prim_value(n + 1), R.Prim(value=_s0))
 *           y = R.Tensor((_s0,)) = R.zeros(R.shape([_s0]), dtype="float32")
 *
 * This ensures VMShapeLower only sees simple tir::Var references, which it can resolve
 * through the MatchCast bindings.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/relax/transform.h>

#include <unordered_map>

namespace tvm {
namespace relax {

namespace {

/*!
 * \brief Check if a PrimExpr is trivial (already canonical for VMShapeLower)
 *
 * Trivial expressions are:
 * - IntImm: concrete integer constants
 * - tir::Var: symbolic variables
 *
 * Any other expression (arithmetic, casts, etc.) is compound and needs canonicalization.
 */
bool IsTrivialPrimExpr(const PrimExpr& expr) {
  return expr->IsInstance<IntImmNode>() || expr->IsInstance<tir::VarNode>();
}

/*!
 * \brief Collector for compound PrimExpr in an expression tree.
 *
 * Scans ShapeExpr nodes and collects all compound (non-trivial) PrimExpr.
 */
class CompoundExprCollector : public ExprVisitor {
 public:
  void VisitExpr_(const ShapeExprNode* op) override {
    for (const PrimExpr& dim : op->values) {
      if (!IsTrivialPrimExpr(dim)) {
        compound_exprs_.insert(dim);
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> compound_exprs_;
};

/*!
 * \brief StructInfo mutator that substitutes PrimExpr according to a mapping.
 */
class StructInfoPrimExprMutator : public StructInfoMutator {
 public:
  explicit StructInfoPrimExprMutator(
      const std::unordered_map<PrimExpr, tir::Var, StructuralHash, StructuralEqual>& expr_map)
      : expr_map_(expr_map) {}

  StructInfo VisitStructInfo_(const TensorStructInfoNode* op) override {
    // Substitute PrimExpr in shape
    ffi::Shape new_shape = op->shape;
    if (new_shape.defined()) {
      ffi::Array<PrimExpr> new_shape_values;
      bool shape_changed = false;

      for (const PrimExpr& dim : new_shape->values) {
        auto it = expr_map_.find(dim);
        if (it != expr_map_.end()) {
          new_shape_values.push_back(it->second);
          shape_changed = true;
        } else {
          new_shape_values.push_back(dim);
        }
      }

      if (shape_changed) {
        new_shape = Shape(new_shape_values);
      }
    }

    DataType new_dtype = op->dtype;

    if (new_shape.same_as(op->shape) && new_dtype == op->dtype) {
      return StructInfoMutator::VisitStructInfo_(op);
    }

    return TensorStructInfo(new_shape, new_dtype, new_ndim_sinfo);
  }

  StructInfo VisitStructInfo_(const ShapeStructInfoNode* op) override {
    // Substitute PrimExpr in shape
    if (op->values.defined()) {
      ffi::Array<PrimExpr> new_values;
      bool changed = false;

      for (size_t i = 0; i < op->values.size(); ++i) {
        const PrimExpr& dim = op->values[i];
        auto it = expr_map_.find(dim);
        if (it != expr_map_.end()) {
          new_values.push_back(it->second);
          changed = true;
        } else {
          new_values.push_back(dim);
        }
      }

      if (changed) {
        return ShapeStructInfo(new_values);
      }
    }
    return StructInfoMutator::VisitStructInfo_(op);
  }

 private:
  const std::unordered_map<PrimExpr, tir::Var, StructuralHash, StructuralEqual>& expr_map_;
};

/*!
 * \brief Mutator to canonicalize ShapeExpr and struct_info by replacing compound PrimExpr
 *        with fresh symbolic variables bound via MatchCast.
 */
class ShapeExprCanonicalizer : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionNode* func) override {
    // Save old state
    BlockBuilder saved_builder = builder_;

    // Create new scope with builder
    builder_ = BlockBuilder();

    // Reset state for each function
    sym_var_counter_ = 0;
    expr_to_var_.clear();

    // First pass: collect all compound expressions in the function body
    // so we can emit MatchCast bindings at the beginning
    CollectCompoundExprsInFunction(func);

    // Visit params
    ffi::Array<Var> params;
    bool all_params_unchanged = true;
    for (Var param : func->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      if (!param.same_as(new_param)) {
        var_remap_[param->vid] = new_param;
        all_params_unchanged = false;
      }
    }

    // Process the function body
    Expr new_body = this->VisitWithNewScope(func->body, params);

    // Also substitute in the return struct_info
    StructInfo new_ret_sinfo = SubstituteStructInfo(func->ret_struct_info);

    bool ret_sinfo_changed = !StructuralEqual()(new_ret_sinfo, func->ret_struct_info);
    bool body_changed = !new_body.same_as(func->body);

    builder_ = saved_builder;

    if (all_params_unchanged && !ret_sinfo_changed && !body_changed) {
      return ffi::GetRef<Function>(func);
    }

    return Function(params, new_body, new_ret_sinfo, func->is_pure, func->attrs, func->span);
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    // First, emit MatchCast bindings for any compound PrimExpr in ShapeExpr
    // This populates expr_to_var_ with mappings from compound expr to fresh vars
    EmitMatchCastForCompoundExprs(binding->value);

    // Now visit the binding with substitution
    Expr new_value = this->VisitExpr(binding->value);

    // Get the struct_info from the new value and substitute compound exprs
    StructInfo new_sinfo = SubstituteStructInfo(GetStructInfo(new_value));

    // Create a new relax::Var with the substituted struct_info
    Var new_var(binding->var->name_hint(), new_sinfo, binding->var->span);

    // Remap the old var to the new var
    var_remap_[binding->var->vid] = new_var;

    // Emit the new binding
    builder_->EmitNormalized(VarBinding(new_var, new_value));
  }

  void VisitBinding_(const MatchCastNode* binding) override {
    // Emit MatchCast bindings for compound PrimExpr in ShapeExpr first
    EmitMatchCastForCompoundExprs(binding->value);

    // Visit the value
    Expr new_value = this->VisitExpr(binding->value);

    // Substitute in the struct_info
    StructInfo new_sinfo = SubstituteStructInfo(GetStructInfo(binding->value));

    // Create a new relax::Var with the substituted struct_info
    Var new_var(binding->var->name_hint(), new_sinfo, binding->var->span);

    var_remap_[binding->var->vid] = new_var;

    builder_->EmitNormalized(MatchCast(new_var, new_value, new_sinfo));
  }

  Expr VisitExpr_(const ShapeExprNode* op) override {
    // Rewrite ShapeExpr to replace compound PrimExpr with fresh symbolic variables
    ffi::Array<PrimExpr> new_values;
    bool changed = false;

    for (const PrimExpr& dim : op->values) {
      if (IsTrivialPrimExpr(dim)) {
        new_values.push_back(dim);
      } else {
        auto it = expr_to_var_.find(dim);
        if (it != expr_to_var_.end()) {
          new_values.push_back(it->second);
          changed = true;
        } else {
          new_values.push_back(dim);
        }
      }
    }

    if (changed) {
      return ShapeExpr(new_values, op->span);
    }
    return ffi::GetRef<ShapeExpr>(op);
  }

 private:
  /*!
   * \brief Collect all compound expressions in a function body.
   */
  void CollectCompoundExprsInFunction(const FunctionNode* func) {
    CompoundExprCollector collector;
    collector.VisitExpr(func->body);
  }

  /*!
   * \brief Scan an expression for ShapeExpr nodes and emit MatchCast bindings
   *        for any compound PrimExpr dimensions.
   */
  void EmitMatchCastForCompoundExprs(const Expr& expr) {
    CompoundExprCollector collector;
    collector.VisitExpr(expr);

    for (const PrimExpr& compound_expr : collector.compound_exprs_) {
      EmitMatchCastIfNeeded(compound_expr);
    }
  }

  /*!
   * \brief Substitute compound PrimExpr in a StructInfo with fresh variables.
   */
  StructInfo SubstituteStructInfo(const StructInfo& sinfo) {
    if (expr_to_var_.empty()) {
      return sinfo;
    }
    StructInfoPrimExprMutator mutator(expr_to_var_);
    return mutator.VisitStructInfo(sinfo);
  }

  /*!
   * \brief Emit a MatchCast binding for a compound PrimExpr if not already done.
   */
  void EmitMatchCastIfNeeded(const PrimExpr& expr) {
    if (IsTrivialPrimExpr(expr)) {
      return;
    }

    if (expr_to_var_.count(expr)) {
      return;
    }

    // Create a fresh tir::Var to hold the computed value
    std::string var_name = "_s" + std::to_string(sym_var_counter_++);
    tir::Var fresh_tir_var(var_name, expr->dtype);

    // Record the mapping for substitution
    expr_to_var_[expr] = fresh_tir_var;

    // Create a PrimValue that computes the compound expression
    PrimValue prim_value(expr);

    // Create a PrimStructInfo that declares the fresh variable as the value
    PrimStructInfo target_sinfo(fresh_tir_var);

    // Create a Relax Var to hold the MatchCast result
    std::string relax_var_name = var_name + "_pv";
    relax::Var match_var(relax_var_name, target_sinfo);

    // Emit the MatchCast binding
    builder_->EmitNormalized(MatchCast(match_var, prim_value, target_sinfo));
  }

  BlockBuilder builder_;
  int sym_var_counter_ = 0;
  std::unordered_map<PrimExpr, tir::Var, StructuralHash, StructuralEqual> expr_to_var_;
};

}  // namespace

Expr CanonicalizeShapeExpr(Expr expr) { return ShapeExprCanonicalizer()(std::move(expr)); }

namespace transform {

Pass CanonicalizeShapeExpr() {
  auto pass_func = [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(relax::CanonicalizeShapeExpr(f));
  };
  return CreateFunctionPass(/*pass_function=*/pass_func,
                            /*opt_level=*/0,
                            /*pass_name=*/"CanonicalizeShapeExpr",
                            /*required=*/{});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.CanonicalizeShapeExpr", CanonicalizeShapeExpr);
}

}  // namespace transform

}  // namespace relax
}  // namespace tvm
