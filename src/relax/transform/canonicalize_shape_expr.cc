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
 * \brief Canonicalize ShapeExpr by replacing composite PrimExpr dimensions with symbolic vars.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/expr.h>

#include <string>
#include <unordered_map>

namespace tvm {
namespace relax {

namespace {

bool IsSimpleShapeDim(const PrimExpr& expr) {
  return expr->IsInstance<IntImmNode>() || expr->IsInstance<tir::VarNode>();
}

class ShapeExprCanonicalizer : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const ShapeExprNode* op) final {
    ffi::Array<PrimExpr> new_values;
    bool changed = false;
    for (const PrimExpr& dim : op->values) {
      if (IsSimpleShapeDim(dim)) {
        new_values.push_back(dim);
        continue;
      }

      changed = true;
      new_values.push_back(GetOrCreateSymbol(dim));
    }

    if (!changed) {
      return ffi::GetRef<ShapeExpr>(op);
    }
    return ShapeExpr(new_values, op->span);
  }

 private:
  tir::Var GetOrCreateSymbol(const PrimExpr& expr) {
    auto it = expr_to_var_.find(expr);
    if (it != expr_to_var_.end()) {
      return it->second;
    }

    std::string base_name = "shape_expr_symbol_" + std::to_string(symbol_counter_++);
    tir::Var sym_var(base_name, expr->dtype);
    expr_to_var_.emplace(expr, sym_var);

    PrimStructInfo target_sinfo(sym_var);
    Var match_var(base_name + "_pv", target_sinfo);
    builder_->EmitNormalized(MatchCast(match_var, PrimValue(expr), target_sinfo));

    return sym_var;
  }

  int symbol_counter_ = 0;
  std::unordered_map<PrimExpr, tir::Var, StructuralHash, StructuralEqual> expr_to_var_;
};

}  // namespace

namespace transform {

Pass CanonicalizeShapeExpr() {
  auto pass_func = [](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(ShapeExprCanonicalizer()(std::move(f)));
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
