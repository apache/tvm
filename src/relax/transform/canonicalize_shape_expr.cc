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
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace relax {

namespace {

bool IsSimpleShapeDim(const PrimExpr& expr) {
  return expr->IsInstance<IntImmNode>() || expr->IsInstance<tirx::VarNode>();
}

class ShapeExprCanonicalizer : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const FunctionNode* op) final {
    bool prev_collecting = collecting_param_bindings_;
    std::vector<Binding> prev_bindings = std::move(param_bindings_);
    param_bindings_.clear();

    collecting_param_bindings_ = true;
    ffi::Array<Var> params;
    bool all_params_unchanged = true;
    for (const Var& param : op->params) {
      Var new_param = this->VisitVarDef(param);
      params.push_back(new_param);
      if (!param.same_as(new_param)) {
        var_remap_[param->vid] = new_param;
        all_params_unchanged = false;
      }
    }
    collecting_param_bindings_ = false;

    Expr body = this->VisitWithNewScope(op->body, params);

    if (!param_bindings_.empty()) {
      body = builder_->Normalize(PrependBindings(body, param_bindings_));
      all_params_unchanged = false;
    }

    // Restore the outer state so nested functions are handled correctly.
    param_bindings_ = std::move(prev_bindings);
    collecting_param_bindings_ = prev_collecting;

    if (all_params_unchanged && body.same_as(op->body)) {
      return ffi::GetRef<Expr>(op);
    } else if (IsBaseOf(GetStructInfo(body), op->ret_struct_info)) {
      return Function(params, body, op->ret_struct_info, op->is_pure, op->attrs);
    } else {
      return Function(params, body, std::nullopt, op->is_pure, op->attrs);
    }
  }

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
  // Prepend the collected parameter-level bindings as a fresh BindingBlock at the start of the
  // (already normalized) function body.
  static Expr PrependBindings(const Expr& body, const std::vector<Binding>& bindings) {
    BindingBlock block(ffi::Array<Binding>(bindings.begin(), bindings.end()));
    if (const auto* seq = body.as<SeqExprNode>()) {
      ffi::Array<BindingBlock> blocks;
      blocks.push_back(block);
      for (const BindingBlock& b : seq->blocks) {
        blocks.push_back(b);
      }
      return SeqExpr(blocks, seq->body, seq->span);
    }
    return SeqExpr({block}, body);
  }

  tirx::Var GetOrCreateSymbol(const PrimExpr& expr) {
    auto it = expr_to_var_.find(expr);
    if (it != expr_to_var_.end()) {
      return it->second;
    }

    std::string base_name = "shape_expr_symbol_" + std::to_string(symbol_counter_++);
    tirx::Var sym_var(base_name, expr->dtype);
    expr_to_var_.emplace(expr, sym_var);

    PrimStructInfo target_sinfo(sym_var);
    Var match_var(base_name + "_pv", target_sinfo);
    MatchCast binding(match_var, PrimValue(expr), target_sinfo);
    if (collecting_param_bindings_) {
      // No active binding block exists while visiting parameters; defer emission.
      param_bindings_.push_back(binding);
    } else {
      builder_->EmitNormalized(binding);
    }

    return sym_var;
  }

  int symbol_counter_ = 0;
  bool collecting_param_bindings_ = false;
  std::vector<Binding> param_bindings_;
  std::unordered_map<PrimExpr, tirx::Var, ffi::StructuralHash, ffi::StructuralEqual> expr_to_var_;
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
