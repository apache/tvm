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

#include "../op/op_common.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

namespace {

ffi::Array<PrimExpr> GetBatchPrefix(const ffi::Array<PrimExpr>& shape) {
  if (shape.size() <= 2) return {};
  return {shape.begin(), shape.end() - 2};
}

PrimExpr ProductDims(const ffi::Array<PrimExpr>& dims) {
  PrimExpr product = IntImm::Int64(1);
  for (const auto& dim : dims) product = product * dim;
  return product;
}

ffi::Optional<ffi::Array<PrimExpr>> InferBatchedMatmulBroadcastPrefix(
    arith::AnalyzerObj* analyzer, const ffi::Array<PrimExpr>& x1, const ffi::Array<PrimExpr>& x2) {
  auto infer_result = InferBinaryBroadcastShape(analyzer, x1, x2);
  if (infer_result.status == BinaryBroadcastShapeInferResult::Status::kSuccess) {
    return infer_result.shape;
  }
  return std::nullopt;
}

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

  PrimExpr symbolic_var_constraints = IntImm::Bool(true);
  auto upper_bounds = func->GetAttr<ffi::Map<ffi::String, Any>>("tir_var_upper_bound");
  auto lower_bounds = func->GetAttr<ffi::Map<ffi::String, Any>>("tir_var_lower_bound");

  if (upper_bounds || lower_bounds) {
    ffi::Map<ffi::String, tirx::Var> name_lookup;
    for (const auto& tir_var : TIRVarsInType(GetType(func))) {
      name_lookup.Set(tir_var->name_hint, tir_var);
      symbolic_var_constraints = symbolic_var_constraints && (0 <= tir_var);
    }

    // Add lower bound constraints
    if (lower_bounds) {
      for (const auto& [key, obj_bound] : lower_bounds.value()) {
        auto tir_var_name = key;
        if (auto opt_var = name_lookup.Get(tir_var_name)) {
          auto var = opt_var.value();
          auto expr_bound = (obj_bound).as_or_throw<PrimExpr>();
          symbolic_var_constraints = symbolic_var_constraints && (expr_bound <= var);
        }
      }
    }

    // Add upper bound constraints
    if (upper_bounds) {
      for (const auto& [key, obj_bound] : upper_bounds.value()) {
        auto tir_var_name = key;
        if (auto opt_var = name_lookup.Get(tir_var_name)) {
          auto var = opt_var.value();
          auto expr_bound = (obj_bound).as_or_throw<PrimExpr>();
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
      auto ty = expr->ty.as<TensorTypeNode>();
      if (ty) {
        return ty->GetShape();
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
      if (shape_a.size() < 2 || shape_b.size() < 2) return expr;
      expr_a = permute_last_two_dims(expr_a);
      expr_b = permute_last_two_dims(expr_b);
      transpose_shape_last_two_dims(shape_a);
      transpose_shape_last_two_dims(shape_b);
    } else if (matches.count(pat_permuted_matmul_on_rhs)) {
      if (shape_b.size() < 2 || shape_c.size() < 2) return expr;
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

    arith::Analyzer analyzer;
    auto prefix_a = GetBatchPrefix(shape_a);
    auto prefix_b = GetBatchPrefix(shape_b);
    auto prefix_c = GetBatchPrefix(shape_c);

    auto opt_prefix_ab = InferBatchedMatmulBroadcastPrefix(analyzer.get(), prefix_a, prefix_b);
    if (!opt_prefix_ab) return expr;
    auto opt_prefix_bc = InferBatchedMatmulBroadcastPrefix(analyzer.get(), prefix_b, prefix_c);
    if (!opt_prefix_bc) return expr;
    auto opt_prefix_outer_lhs =
        InferBatchedMatmulBroadcastPrefix(analyzer.get(), opt_prefix_ab.value(), prefix_c);
    if (!opt_prefix_outer_lhs) return expr;
    auto opt_prefix_outer_rhs =
        InferBatchedMatmulBroadcastPrefix(analyzer.get(), prefix_a, opt_prefix_bc.value());
    if (!opt_prefix_outer_rhs) return expr;

    PrimExpr batch_ab = ProductDims(opt_prefix_ab.value());
    PrimExpr batch_bc = ProductDims(opt_prefix_bc.value());
    PrimExpr batch_outer_lhs = ProductDims(opt_prefix_outer_lhs.value());
    PrimExpr batch_outer_rhs = ProductDims(opt_prefix_outer_rhs.value());

    // Compare naive matmul FLOPs for two evaluation orders of
    //   matmul(A, matmul(B, C))  vs  matmul(matmul(A, B), C)
    //
    // Matrix dims (last two axes): A [N, R], B [R, M], C [M, B_last]
    // Each matmul uses the broadcasted batch prefix of its operands.
    //
    // LHS first — matmul(matmul(A, B), C):
    //   batch_ab * N * R * M + batch_outer_lhs * N * M * B_last
    PrimExpr ops_with_lhs_first =
        batch_ab * size_N * size_R * size_M + batch_outer_lhs * size_N * size_M * size_B;
    // RHS first — matmul(A, matmul(B, C)):
    //   batch_bc * R * M * B_last + batch_outer_rhs * N * R * B_last
    PrimExpr ops_with_rhs_first =
        batch_bc * size_R * size_M * size_B + batch_outer_rhs * size_N * size_R * size_B;

    analyzer->rewrite_simplify.SetEnabledExtensions(
        static_cast<arith::RewriteSimplifier::Extension>(
            analyzer->rewrite_simplify.GetEnabledExtensions() |
            arith::RewriteSimplifier::Extension::kComparisonOfProductAndSum));
    With<arith::ConstraintContext> func_attr_constraint(analyzer, symbolic_var_constraints);
    With<arith::ConstraintContext> analyzer_constraint(
        analyzer, batch_ab > 0 && batch_bc > 0 && batch_outer_lhs > 0 && batch_outer_rhs > 0 &&
                      size_N > 0 && size_R > 0 && size_M > 0 && size_B > 0);

    if (analyzer->CanProve(ops_with_lhs_first < ops_with_rhs_first)) {
      return matmul(matmul(expr_a, expr_b, DataType::Void()), expr_c, DataType::Void());
    } else if (analyzer->CanProve(ops_with_rhs_first < ops_with_lhs_first)) {
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
