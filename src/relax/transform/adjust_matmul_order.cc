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

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)>> CreatePatterns(
    const Function& func) {
  auto compile_time_arr = ComputableAtCompileTime(func);
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> compile_time_lookup(
      compile_time_arr.begin(), compile_time_arr.end());

  auto pat_lhs = WildcardPattern();

  auto pat_rhs_a = WildcardPattern();
  auto pat_rhs_b = WildcardPattern();
  auto pat_rhs = IsOp("relax.matmul")(pat_rhs_a, pat_rhs_b);

  auto pat_matmul = IsOp("relax.matmul")(pat_lhs, pat_rhs);

  auto rewriter = [=](Expr expr, Map<DFPattern, Expr> matches) -> Expr {
    auto lhs = matches[pat_lhs];
    auto rhs_a = matches[pat_rhs_a];
    auto rhs_b = matches[pat_rhs_b];

    auto is_compile_time = [&compile_time_lookup](Expr arg) -> bool {
      if (auto as_var = arg.as<Var>()) {
        return compile_time_lookup.count(as_var.value());
      } else {
        return false;
      }
    };

    if (is_compile_time(rhs_a) && is_compile_time(rhs_b)) {
      return expr;
    } else {
      return matmul(matmul(lhs, rhs_a, DataType::Void()), rhs_b, DataType::Void());
    }
  };

  return {pat_matmul, rewriter};
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
