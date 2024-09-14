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

#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_set>
#include <vector>

#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)>> CreatePatterns(
    const Function& func) {
  auto compile_time_arr = ComputableAtCompileTime(func);
  std::unordered_set<Var> compile_time_lookup(compile_time_arr.begin(), compile_time_arr.end());

  TypedPackedFunc<bool(Expr)> is_compile_time = [compile_time_lookup](Expr arg) -> bool {
    if (auto as_var = arg.as<Var>()) {
      return compile_time_lookup.count(as_var.value());
    } else {
      return false;
    }
  };
  TypedPackedFunc<bool(Expr)> is_runtime = [is_compile_time](Expr arg) -> bool {
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

  PrimExpr symbolic_var_constraints = Bool(true);
  if (auto upper_bounds = func->GetAttr<Map<ObjectRef, ObjectRef>>("tir_var_upper_bound")) {
    Map<String, tir::Var> name_lookup;
    for (const auto& tir_var : TIRVarsInStructInfo(GetStructInfo(func))) {
      name_lookup.Set(tir_var->name_hint, tir_var);
      symbolic_var_constraints = symbolic_var_constraints && (0 <= tir_var);
    }

    for (const auto& [key, obj_bound] : upper_bounds.value()) {
      auto tir_var_name = Downcast<String>(key);
      if (auto opt_var = name_lookup.Get(tir_var_name)) {
        auto var = opt_var.value();
        auto expr_bound = Downcast<PrimExpr>(obj_bound);
        symbolic_var_constraints = symbolic_var_constraints && (var < expr_bound);
      }
    }
  }

  auto rewriter = [=](Expr expr, Map<DFPattern, Expr> matches) -> Expr {
    auto expr_a = matches[pat_a];
    auto expr_b = matches[pat_b];
    auto expr_c = matches[pat_c];

    // If all three components are compile-time, the order doesn't
    // matter as the entire expression can be lifted out and
    // pre-computed.
    if (is_compile_time(expr_a) && is_compile_time(expr_b) && is_compile_time(expr_c)) {
      return expr;
    }

    auto get_shape = [](Expr expr) -> Optional<Array<PrimExpr>> {
      auto sinfo = expr->struct_info_.as<TensorStructInfoNode>();
      if (sinfo) {
        return sinfo->GetShape();
      } else {
        return NullOpt;
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

    if (matches.count(pat_permuted_matmul_on_lhs)) {
      expr_a = permute_dims(expr_a, NullOpt);
      expr_b = permute_dims(expr_b, NullOpt);
      CHECK_EQ(shape_a.size(), 2);
      CHECK_EQ(shape_b.size(), 2);
      shape_a = {shape_a[1], shape_a[0]};
      shape_b = {shape_b[1], shape_b[0]};
    } else if (matches.count(pat_permuted_matmul_on_rhs)) {
      expr_b = permute_dims(expr_b, NullOpt);
      expr_c = permute_dims(expr_c, NullOpt);
      CHECK_EQ(shape_b.size(), 2);
      CHECK_EQ(shape_c.size(), 2);
      shape_b = {shape_b[1], shape_b[0]};
      shape_c = {shape_c[1], shape_c[0]};
    }

    // If two of the three are compile-time, group those two values
    // together, to allow them to be lifted out and pre-computed.
    if (is_compile_time(expr_a) && is_compile_time(expr_b)) {
      return matmul(matmul(expr_a, expr_b, DataType::Void()), expr_c, DataType::Void());
    } else if (is_compile_time(expr_b) && is_compile_time(expr_c)) {
      return matmul(expr_a, matmul(expr_b, expr_c, DataType::Void()), DataType::Void());
    }

    // Otherwise, select the order that reduces the total number of
    // operations required, assuming a naive matmul.

    // Matmul on LHS: ([N,R]*[R,M]) * [M,batch]
    // Matmul on RHS: [N,R] * ([R,M]*[M,batch])
    //
    // LHS first: `N*R*M + N*M*batch = N*M*(R+batch)`
    // RHS first: `N*R*batch + R*M*batch = (N+M)*R*batch`

    if (shape_a.size() == 1) {
      shape_a = {IntImm(shape_a[0].dtype(), 1), shape_a[0]};
    }
    if (shape_b.size() == 1) {
      if (matches.count(pat_matmul_on_lhs)) {
        shape_b = {shape_b[0], IntImm(shape_b[0].dtype(), 1)};
      } else if (matches.count(pat_matmul_on_rhs)) {
        shape_b = {IntImm(shape_b[0].dtype(), 1), shape_b[0]};
      } else {
        LOG(FATAL) << "InternalError: "
                   << "OrPattern " << pat << " matched, but neither " << pat_matmul_on_lhs
                   << " nor " << pat_matmul_on_rhs << " matched";
      }
    }
    if (shape_c.size() == 1) {
      shape_c = {shape_c[0], IntImm(shape_c[0].dtype(), 1)};
    }

    auto size_N = shape_a[shape_a.size() - 2];
    auto size_R = shape_a[shape_a.size() - 1];
    auto size_M = shape_c[shape_c.size() - 2];
    auto size_B = shape_c[shape_c.size() - 1];

    auto ops_with_lhs_first = (size_R + size_B) * size_N * size_M;
    auto ops_with_rhs_first = (size_M + size_N) * size_R * size_B;

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

    // If we cannot determine which order is best, keep the existing
    // order.
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

TVM_REGISTER_GLOBAL("relax.transform.AdjustMatmulOrder").set_body_typed(AdjustMatmulOrder);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
