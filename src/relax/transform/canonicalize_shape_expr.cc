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
 * \brief Cannonicalize ShapeExpr by lifting compound PrimExpr into separate bindings.
 *
 * VMShapeLower can only handle ShapeExpr where each dimension is either:
 * - IntImm (concrete integer constant)
 * - tir::Var (symbolic variable)
 *
 * This pass lifts compound PrimExpr (e.g., n+1, 4*n*m, etc.) into separate shape bindings
 * with MatchCast to extract symbolic variables, ensuring VMShapeLower receives only
 * cannonical shape expressions.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/struct_info_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/analysis.h>

#include <unordered_map>

namespace tvm {
namespace relax {

namespace {

/*!
 * \brief Check if a PrimExpr is cannonical for VMShapeLower
 *
 * VMShapeLower can only handle:
 * - IntImm: concrete integer constants
 * - tir::Var: symbolic variables that can be stored/loaded at runtime
 *
 * Any other expression (arithmetic, casts, etc.) is compound and needs canonicalization.
 */
bool IsCanonicalPrimExpr(const PrimExpr& expr) {
  return expr->IsInstance<IntImmNode>() || expr->IsInstance<tir::VarNode>();
}

/*!
 * \brief Mutator to canonicalize ShapeExpr in struct info
 *
 * This pass handles ShapeExpr canonicalization by:
 * 1. Detecting compound PrimExpr in variable struct_info
 * 2. Emitting ShapeExpr bindings to compute expressions
 * 3. Using MatchCast to extract values into fresh symbolic tir::Var
 * 4. Replacing compound expressions with these canonical vars in struct_info
 */
class ShapeExprCanonicalizer : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionNode* func) override {
    // Reset state for each function
    symbolic_var_counter_ = 0;
    compound_expr_to_var_.clear();
    emitted_bindings_.clear();

    // Visit params to populate var_remap_
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

    // Process the function body with proper scope setup
    Expr new_body = this->VisitWithNewScope(func->body, params);

    if (all_params_unchanged && new_body.same_as(func->body)) {
      return ffi::GetRef<Function>(func);
    }

    return Function(params, new_body, func->ret_struct_info, func->is_pure, func->attrs,
                    func->span);
  }

  Expr VisitExpr_(const ShapeExprNode* op) override {
    // Just cannonicalize ShapeExpr values by replacing compound expression with symbolic vars
    // The bindings should have been emitted earlier by EmitBindingsForExpr

    // Mark a copy of values to avoid any reference issues
    std::vector<PrimExpr> original_dims;
    for (const PrimExpr& dim : op->values) {
      original_dims.push_back(dim);
    }

    ffi::Array<PrimExpr> canonical_values;
    bool changed = false;

    for (const PrimExpr& dim : original_dims) {
      PrimExpr canonical_dim = GetCanonicalDimension(dim);
      canonical_values.push_back(canonical_dim);
      changed |= !canonical_dim.same_as(dim);
    }

    if (!changed) {
      return ffi::GetRef<ShapeExpr>(op);
    }

    return ShapeExpr(canonical_values, op->span);
  }

  /*!
   * \brief Scan an expression for ShapeExprs and emit bindings for compound expressions.
   * This must  be called BEFORE visiting the expression to ensure bindings are emitted first.
   */
  void EmitBindingsForExpr(const Expr& expr) {
    // Use a simple visitor to find ShapeExpr nodes
    class ShapeExprScanner : public ExprVisitor {
     public:
      explicit ShapeExprScanner(ShapeExprCanonicalizer* canonicalizer)
          : canonicalizer_(canonicalizer) {}

      void VisitExpr_(const ShapeExprNode* op) override {
        // Make a copy of values to avoid reference issues during emission
        std::vector<PrimExpr> dims;
        for (const PrimExpr& dim : op->values) {
          dims.push_back(dim);
        }
        for (const PrimExpr& dim : dims) {
          if (!IsCanonicalPrimExpr(dim)) {
            canonicalizer_->CanonicalizeDimension(dim);
          }
        }
      }

     private:
      ShapeExprCanonicalizer* canonicalizer_;
    };

    ShapeExprScanner scanner(this);
    scanner.VisitExpr(expr);
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    // Emit canonicalization bindings before processing the binding.
    // Scan the binding's value for ShapeExprs with compound expressions.
    EmitBindingsForExpr(binding->value);

    // Let the base class handle the rest
    ExprMutator::VisitBinding_(binding);
  }

  void VisitBinding_(const MatchCastNode* binding) override {
    // Scan the binding's value for ShapeExprs with compound expressions
    EmitBindingsForExpr(binding->value);

    // Delegate to base handling
    ExprMutator::VisitBinding_(binding);
  }

  Var VisitVarDef(const Var& var) override {
    // Don't canonicalize struct_info - just delegate to base
    return ExprMutator::VisitVarDef(var);
  }

 private:
  /*!
   * \brief Get the canonical form of a dimension (returns the symbolic var if already emitted)
   */
  PrimExpr GetCanonicalDimension(const PrimExpr& dim) {
    // If already canonical, return as is
    if (IsCanonicalPrimExpr(dim)) {
      return dim;
    }

    // Check if we've already canonicalized this expression
    if (auto it = compound_expr_to_var_.find(dim); it != compound_expr_to_var_.end()) {
      return it->second;
    }

    // Create a fresh symbolic variable, but don't emit yet
    tir::Var symbolic_var = CreateFreshSymbolicVar(dim->dtype);

    compound_expr_to_var_[dim] = symbolic_var;

    return symbolic_var;
  }

  /*!
   * \brief Emit bindings for a single compound dimension
   *
   * If the dimension is a compound PrimExpr:
   * 1. Create a fresh symbolic tir::Var for the compound expression
   * 2. Emit a MatchCast from a PrimValue to define the symbolic var
   */
  void CanonicalizeDimension(const PrimExpr& dim) {
    // If already canonical, nothing to emit
    if (IsCanonicalPrimExpr(dim)) {
      return;
    }

    // Check if we've already emitted the bindings
    if (emitted_bindings_.count(dim)) {
      return;
    }

    // Mark as emitted BEFORE emitting to prevent infinite recursion
    emitted_bindings_.insert(dim);

    // Get or create the symbolic var for this compound expression
    tir::Var symbolic_var;
    auto it = compound_expr_to_var_.find(dim);
    if (it != compound_expr_to_var_.end()) {
      symbolic_var = it->second;
    } else {
      DataType dtype = dim->dtype;
      symbolic_var = CreateFreshSymbolicVar(dtype);
      compound_expr_to_var_[dim] = symbolic_var;
    }

    // Emit a PrimValue binding with the compound expression
    // This will be processed by VMShapeLower to compute the value
    PrimValue prim_value(dim);
    PrimStructInfo prim_sinfo(dim->dtype);
    std::string prim_var_name = "_prim" + std::to_string(symbolic_var_counter_ - 1);
    Var prim_var(prim_var_name, prim_sinfo);
    builder_->EmitNormalized(VarBinding(prim_var, prim_value));

    // Emit MatchCast to extract the computed value into the symbolic variable
    // The pattern uses the symbolic var which will be defined by this MatchCast
    PrimStructInfo match_sinfo(symbolic_var);
    std::string match_var_name = "_match" + std::to_string(symbolic_var_counter_ - 1);
    Var match_cast_var(match_var_name, match_sinfo);
    builder_->EmitNormalized(MatchCast(match_cast_var, prim_var, match_sinfo));
  }

  /*!
   * \brief Create a fresh symbolic TIR variable
   */
  tir::Var CreateFreshSymbolicVar(DataType dtype) {
    std::string name = "s" + std::to_string(symbolic_var_counter_++);
    return tir::Var(name, dtype);
  }

  // Cache to avoid creating duplicate bindings for the same compound expression
  std::unordered_map<PrimExpr, tir::Var, StructuralHash, StructuralEqual> compound_expr_to_var_;

  // Track which compound expressions have had their bindings emitted
  std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> emitted_bindings_;

  // Counter for generating unique symbolic variable names
  int symbolic_var_counter_ = 0;
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
