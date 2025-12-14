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
 * 1. Detecting compound PrimExpr in ShapeExpr dimensions
 * 2. Lifting them into separate ShapeExpr bindings
 * 3. Using MatchCast to extract values into fresh symbolic tir::Var
 * 4. Replacing compound expressions with these canonical vars
 */
class ShapeExprCanonicalizer : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionNode* func) override {
    // Reset state for each function
    auto cached_compound_to_var = compound_expr_to_var_;
    auto cached_counter = symbolic_var_counter_;

    auto result = ExprMutator::VisitExpr_(func);

    compound_expr_to_var_ = cached_compound_to_var;
    symbolic_var_counter_ = cached_counter;

    return result;
  }

  /*!
   * \brief Override VisitVarDef to canonicalize struct_info
   *
   * This is where we intercept variable definitions and canonicalize any
   * compound PrimExpr in their TensorStructInfo shapes.
   */
  Var VisitVarDef(const Var& var) override {
    auto sinfo = GetStructInfo(var);

    // Check if we need to canonicalize the struct_info
    auto canonical_sinfo = CanonicalizeStructInfo(sinfo);

    if (canonical_sinfo.same_as(sinfo)) {
      // No changes needed
      return ExprMutator::VisitVarDef(var);
    }

    // Create a new var with canonicalized strcut_info
    if (var->IsInstance<DataflowVarNode>()) {
      return DataflowVar(var->vid, canonical_sinfo, var->span);
    }
    return Var(var->vid, canonical_sinfo, var->span);
  }

 private:
  /*!
   * \brief Canonicalize struct info by lifting compound shape expressions
   */
  StructInfo CanonicalizeStructInfo(const StructInfo& sinfo) {
    if (auto tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
      return CanonicalizeTensorStructInfo(ffi::GetRef<TensorStructInfo>(tensor_sinfo));
    } else if (auto tuple_sinfo = sinfo.as<TupleStructInfoNode>()) {
      return CanonicalizeTupleStructInfo(ffi::GetRef<TupleStructInfo>(tuple_sinfo));
    }
    return sinfo;
  }

  /*!
   * \brief Canonicalize TensorStructInfo by handling compound shape expressions
   */
  TensorStructInfo CanonicalizeTensorStructInfo(const TensorStructInfo& sinfo) {
    if (!sinfo->shape.defined()) {
      return sinfo;
    }

    auto shape_expr = sinfo->shape.as<ShapeExprNode>();
    if (!shape_expr) {
      // Shape is Var, not a ShapeExpr - no canonicalization needed
      return sinfo;
    }

    // Canonicalize each dimension
    ffi::Array<PrimExpr> canonical_dims;
    bool changed = false;

    for (const PrimExpr& dim : shape_expr->values) {
      PrimExpr canonical_dim = CanonicalizeDimension(dim);
      canonical_dims.push_back(canonical_dim);
      changed |= !canonical_dim.same_as(dim);
    }

    if (!changed) {
      return sinfo;
    }

    // Create new TensorStructInfo with canonicalized shape
    return TensorStructInfo(ShapeExpr(canonical_dims), sinfo->dtype, sinfo->vdevice, sinfo->span);
  }

  /*!
   * \brief Canonicalize TupleStructInfo recursively
   */
  TupleStructInfo CanonicalizeTupleStructInfo(const TupleStructInfo& sinfo) {
    ffi::Array<StructInfo> canonical_fields;
    bool changed = false;

    for (const StructInfo& field : sinfo->fields) {
      StructInfo canonical_field = CanonicalizeStructInfo(field);
      canonical_fields.push_back(canonical_field);
      changed |= !canonical_field.same_as(field);
    }

    if (!changed) {
      return sinfo;
    }

    return TupleStructInfo(canonical_fields, sinfo->span);
  }

  /*!
   * \brief Canonicalize a single shape dimension
   *
   * If the dimension is a compound PrimExpr:
   * 1. Emit a ShapeExpr binding containing the compound expression
   * 2. Create a fresh symbolic tir::Var
   * 3. Emit a MatchCast to bind the computed value to the symbolic var
   * 4. Return the symbolic var
   */
  PrimExpr CanonicalizeDimension(const PrimExpr& dim) {
    // If already canonical, return as is
    if (IsCanonicalPrimExpr(dim)) {
      return dim;
    }

    // Check if we've already canonicalized this expression
    if (auto it = compound_expr_to_var_.find(dim); it != compound_expr_to_var_.end()) {
      return it->second;
    }

    // Create a fresh symbolic variable
    tir::Var symbolic_var = CreateFreshSymbolicVar(dim->dtype);

    // Emit shape binding: shape_var = R.shape([compound_expr])
    ShapeExpr shape_value({dim});
    Var shape_var = builder_->Emit(shape_value);

    // Emit MatchCast to extract the computed value into the symbolic variable
    // match_cast_var: R.Shape([symbolic_var]) = shape_var
    ShapeStructInfo match_sinfo(ffi::Array<PrimExpr>{symbolic_var});
    Var match_cast_var("_", match_sinfo);
    builder_->EmitNormalized(MatchCast(match_cast_var, shape_var, match_sinfo));

    // Cache the mapping to avoid duplicate bindings
    compound_expr_to_var_[dim] = symbolic_var;

    return symbolic_var;
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
