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
 * \file tvm/relax/transform/adjust_matmul_order.cc
 * \brief Re-order `matmul(matmul(A,B), x)` to `matmul(A, matmul(B,x))`
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tirx/op.h>

#include <optional>
#include <unordered_set>
#include <vector>

#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, ffi::TypedFunction<Expr(Expr, ffi::Map<DFPattern, Expr>)>> CreatePatterns(
    const Function& func) {
  auto compile_time_arr = ComputableAtCompileTime(func);
  std::unordered_set<Var> compile_time_lookup(compile_time_arr.begin(), compile_time_arr.end());

  ffi::TypedFunction<bool(Expr)> is_compile_time = [compile_time_lookup](Expr arg) -> bool {
    if (auto as_var = arg.as<Var>()) {
      return compile_time_lookup.count(as_var.value());
    } else {
      return false;
    }
  };
  ffi::TypedFunction<bool(Expr)> is_runtime = [is_compile_time](Expr arg) -> bool {
    return !is_compile_time(arg);
  };

  DFPattern pat_a = WildcardPattern();
  DFPattern pat_b = WildcardPattern();
  DFPattern pat_c = WildcardPattern();

  auto pat_matmul = IsOp("relax.matmul");
  auto pat_permute_dims = IsOp("relax.permute_dims");

  auto pat_matmul_on_lhs = pat_matmul(pat_matmul(pat_a, pat_b), pat_c);
  auto pat_matmul_on_rhs = pat_matmul(pat_a, pat_matmul(pat_b, pat_c));

  auto pat_permuted_matmul_on_lhs = pat_matmul(pat_permute_dims(pat_matmul(pat_b, pat_a)), pat_c);
  auto pat_permuted_matmul_on_rhs = pat_matmul(pat_a, pat_permute_dims(pat_matmul(pat_c, pat_b)));

  auto pat = pat_matmul_on_lhs | pat_matmul_on_rhs | pat_permuted_matmul_on_lhs |
             pat_permuted_matmul_on_rhs;

  PrimExpr symbolic_var_constraints = tirx::const_true();
  auto upper_bounds = func->GetAttr<ffi::Map<ffi::String, Any>>("tir_var_upper_bound");
  auto lower_bounds = func->GetAttr<ffi::Map<ffi::String, Any>>("tir_var_lower_bound");

  if (upper_bounds || lower_bounds) {
    ffi::Map<ffi::String, tirx::Var> name_lookup;
    for (const auto& tir_var : TIRVarsInStructInfo(GetStructInfo(func))) {
      name_lookup.Set(tir_var->name_hint, tir_var);
      symbolic_var_constraints = symbolic_var_constraints && (0 <= tir_var);
    }

    // Add lower bound constraints
    if (lower_bounds) {
      for (const auto& [key, obj_bound] : lower_bounds.value()) {
        auto tir_var_name = Downcast<ffi::String>(key);
        if (auto opt_var = name_lookup.Get(tir_var_name)) {
          auto var = opt_var.value();
          auto expr_bound = Downcast<PrimExpr>(obj_bound);
          symbolic_var_constraints = symbolic_var_constraints && (expr_bound <= var);
        }
      }
    }

    // Add upper bound constraints
    if (upper_bounds) {
      for (const auto& [key, obj_bound] : upper_bounds.value()) {
        auto tir_var_name = Downcast<ffi::String>(key);
        if (auto opt_var = name_lookup.Get(tir_var_name)) {
          auto var = opt_var.value();
          auto expr_bound = Downcast<PrimExpr>(obj_bound);
          symbolic_var_constraints = symbolic_var_constraints && (var < expr_bound);
        }
      }
    }
  }

  auto rewriter = [=](Expr expr, ffi::Map<DFPattern, Expr> matches) -> Expr {
    auto expr_a = matches[pat_a];
    auto expr_b = matches[pat_b];
    auto expr_c = matches[pat_c];

    // If all three components are compile-time, the order doesn't
    // matter as the entire expression can be lifted out and
    // pre-computed.
    if (is_compile_time(expr_a) && is_compile_time(expr_b) && is_compile_time(expr_c)) {
      return expr;
    }

    auto get_shape = [](Expr expr) -> ffi::Optional<ffi::Array<PrimExpr>> {
      auto sinfo = expr->struct_info_.as<TensorStructInfoNode>();
      if (sinfo) {
        return sinfo->GetShape();
      } else {
        return std::nullopt;
      }
    };

    auto opt_shape_a = get_shape(expr_a);
    if (!opt_shape_a) return expr;
    auto opt_shape_b = get_shape(expr_b);
    if (!opt_shape_b) return expr;
    auto opt_shape_c = get_shape(expr_c);
    if (!opt_shape_c) return expr;

    auto shape_a = opt_shape_a.value();
    auto shape_b = opt_shape_b.value();
    auto shape_c = opt_shape_c.value();

    auto permute_last_two_dims = [&](Expr expr) -> Expr {
      auto opt_shape = get_shape(expr);
      if (!opt_shape) return expr;

      size_t ndim = opt_shape.value().size();
      TVM_FFI_ICHECK_GE(ndim, 2);

      ffi::Optional<ffi::Array<int64_t>> axes;

      if (ndim == 2) {
        // Pass none axes to permute_dims for simple transpose of 2D tensors.
        axes = std::nullopt;
      } else {
        ffi::Array<int64_t> axes_array;
        for (size_t i = 0; i < ndim; ++i) axes_array.push_back(i);
        axes_array.Set(ndim - 1, ndim - 2);
        axes_array.Set(ndim - 2, ndim - 1);
        axes = ffi::Optional<ffi::Array<int64_t>>(axes_array);
      }
      return permute_dims(std::move(expr), axes);
    };

    auto transpose_shape_last_two_dims = [&](ffi::Array<PrimExpr>& shape) {
      PrimExpr last_dim_shape = shape[shape.size() - 1];
      shape.Set(shape.size() - 1, shape[shape.size() - 2]);
      shape.Set(shape.size() - 2, last_dim_shape);
    };

    if (matches.count(pat_permuted_matmul_on_lhs)) {
      expr_a = permute_last_two_dims(expr_a);
      expr_b = permute_last_two_dims(expr_b);
      transpose_shape_last_two_dims(shape_a);
      transpose_shape_last_two_dims(shape_b);
    } else if (matches.count(pat_permuted_matmul_on_rhs)) {
      expr_b = permute_last_two_dims(expr_b);
      expr_c = permute_last_two_dims(expr_c);
      transpose_shape_last_two_dims(shape_b);
      transpose_shape_last_two_dims(shape_c);
    }

    // If two of the three are compile-time, group those two values
    // together, to allow them to be lifted out and pre-computed.
    if (is_compile_time(expr_a) && is_compile_time(expr_b)) {
      return matmul(matmul(expr_a, expr_b, DataType::Void()), expr_c, DataType::Void());
    } else if (is_compile_time(expr_b) && is_compile_time(expr_c)) {
      return matmul(expr_a, matmul(expr_b, expr_c, DataType::Void()), DataType::Void());
    }

    // Otherwise, select the order that reduces the total number of
    // operations required, assuming a naive matmul (see below).

    if (shape_a.size() == 1) {
      shape_a = {IntImm(shape_a[0].dtype(), 1), shape_a[0]};
    }
    if (shape_b.size() == 1) {
      if (matches.count(pat_matmul_on_lhs)) {
        shape_b = {shape_b[0], IntImm(shape_b[0].dtype(), 1)};
      } else if (matches.count(pat_matmul_on_rhs)) {
        shape_b = {IntImm(shape_b[0].dtype(), 1), shape_b[0]};
      } else {
        TVM_FFI_THROW(InternalError)
            << "OrPattern " << pat << " matched, but neither " << pat_matmul_on_lhs << " nor "
            << pat_matmul_on_rhs << " matched";
      }
    }
    if (shape_c.size() == 1) {
      shape_c = {shape_c[0], IntImm(shape_c[0].dtype(), 1)};
    }

    PrimExpr size_N = shape_a[shape_a.size() - 2];  // row of A
    PrimExpr size_R = shape_a[shape_a.size() - 1];  // col of A and row of B
    PrimExpr size_M = shape_c[shape_c.size() - 2];  // row of C and col of B
    PrimExpr size_B = shape_c[shape_c.size() - 1];  // col of C

    auto calculate_batch = [](ffi::Array<PrimExpr>& shape) {
      PrimExpr batch = 1;
      for (size_t i = 0; i < shape.size() - 2; ++i) {
        batch *= shape[i];
      }
      return batch;
    };

    PrimExpr batch_A = calculate_batch(shape_a);
    PrimExpr batch_B = calculate_batch(shape_b);
    PrimExpr batch_C = calculate_batch(shape_c);

    // Compare naive matmul FLOPs for two evaluation orders of
    //   matmul(A, matmul(B, C))  vs  matmul(matmul(A, B), C)
    //
    // Matrix dims (last two axes of each operand):
    //   A: [N, R]   B: [R, M]   C: [M, B_last]
    // Batch prefixes (product of all leading axes):
    //   batch_A, batch_B, batch_C
    //
    // LHS first — matmul(matmul(A, B), C):
    //   inner  matmul(A, B): batch_A * batch_B * N * R * M
    //   outer  matmul(., C): batch_A * batch_B * batch_C * N * M * B_last
    //   total: batch_A * batch_B * N * M * (R + batch_C * B_last)
    PrimExpr ops_with_lhs_first = (size_R + batch_C * size_B) * size_N * size_M * batch_A * batch_B;
    // RHS first — matmul(A, matmul(B, C)):
    //   inner  matmul(B, C): batch_B * batch_C * R * M * B_last
    //   outer  matmul(A, .): batch_A * batch_B * batch_C * N * R * B_last
    //   total: batch_B * batch_C * R * B_last * (M + batch_A * N)
    PrimExpr ops_with_rhs_first = (size_M + batch_A * size_N) * size_R * size_B * batch_B * batch_C;

    arith::Analyzer analyzer;
    analyzer.rewrite_simplify.SetEnabledExtensions(static_cast<arith::RewriteSimplifier::Extension>(
        analyzer.rewrite_simplify.GetEnabledExtensions() |
        arith::RewriteSimplifier::Extension::kComparisonOfProductAndSum));
    With<arith::ConstraintContext> func_attr_constraint(&analyzer, symbolic_var_constraints);
    With<arith::ConstraintContext> analyzer_constraint(
        &analyzer, size_N > 0 && size_R > 0 && size_M > 0 && size_B > 0);

    if (analyzer.CanProve(ops_with_lhs_first < ops_with_rhs_first)) {
      return matmul(matmul(expr_a, expr_b, DataType::Void()), expr_c, DataType::Void());
    } else if (analyzer.CanProve(ops_with_rhs_first < ops_with_lhs_first)) {
      return matmul(expr_a, matmul(expr_b, expr_c, DataType::Void()), DataType::Void());
    }

    // If we cannot determine which order is best, keep the existing order.
    return expr;
  };

  return {pat, rewriter};
}

}  // namespace

namespace transform {
Pass AdjustMatmulOrder() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    auto [pattern, rewriter] = CreatePatterns(func);
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "AdjustMatmulOrder", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.transform.AdjustMatmulOrder", AdjustMatmulOrder);
}

}  // namespace transform
}  // namespace relax
}  // namespace tvm
