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
 * \file tvm/relax/transform/reorder_permute_dims_after_concat.cc
 * \brief Reorder concat(permute_dims(A), permute_dims(B)) into permute_dims(concat(A,B))
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_set>
#include <vector>

#include "../op/tensor/index.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

namespace {
std::tuple<DFPattern, TypedPackedFunc<Expr(Expr, Map<DFPattern, Expr>)>> CreatePatterns() {
  // TODO(Lunderberg): Allow pattern-matching to handle a flexible
  // number of arguments, each of which matches the same type of
  // pattern.
  //
  // Because we instantiate one DFPattern for each value in
  // `min_concat <= i <= max_concat`, we don't want to set
  // `max_concat` to an extremely high value.  The current value of 12
  // was chosen to be significantly higher than the highest value
  // required so far (3, for query/key/value in attention layers), but
  // not so high that it requires an excessive number of `DFPattern`.
  //
  // This value is deliberately *NOT* exposed, as `max_concat` may be
  // increased at any point that it is required, and other use cases
  // should not depend on its value.  If there is a use case that
  // requires more matmuls to be handled, and pattern-matching does
  // not yet support a flexible number of `Tuple` elements,
  // `max_concat` should be increased.
  size_t min_concat = 2;
  size_t max_concat = 12;

  std::vector<DFPattern> pat_args;
  std::vector<DFPattern> pat_permute_dims;
  for (size_t i = 0; i < max_concat; i++) {
    auto arg = WildcardPattern();
    pat_args.push_back(arg);
    pat_permute_dims.push_back(IsOp("relax.permute_dims")(arg));
  }

  auto make_pattern_with_num_concat = [&](size_t num_concat) -> DFPattern {
    ICHECK_LT(num_concat, pat_permute_dims.size());
    auto concat_tuple = TuplePattern(
        Array<DFPattern>(pat_permute_dims.begin(), pat_permute_dims.begin() + num_concat));
    return IsOp("relax.concat")(concat_tuple);
  };

  DFPattern pat_concat = make_pattern_with_num_concat(min_concat);
  for (size_t i = min_concat + 1; i < max_concat; i++) {
    pat_concat = pat_concat | make_pattern_with_num_concat(i);
  }

  auto get_permute_dims_optional_axes = [](const Expr& expr) -> Optional<Array<Integer>> {
    auto call = expr.as<CallNode>();
    ICHECK(call);
    auto attrs = call->attrs.as<PermuteDimsAttrs>();
    ICHECK(attrs);

    return attrs->axes;
  };

  auto get_permute_dims_axes =
      [get_permute_dims_optional_axes](const Expr& expr) -> Array<Integer> {
    if (auto opt_axes = get_permute_dims_optional_axes(expr)) {
      return opt_axes.value();
    } else {
      auto call = Downcast<Call>(expr);
      Array<Integer> permutation;
      auto arg_sinfo = call->args[0]->struct_info_.as<TensorStructInfoNode>();
      CHECK(arg_sinfo) << "Expected permute_dims to have a single tensor argument, "
                       << "but argument " << call->args[0] << " has struct info "
                       << call->args[0]->struct_info_;
      CHECK_GE(arg_sinfo->ndim, 0);
      size_t ndim = arg_sinfo->ndim;
      for (size_t i = 0; i < ndim; i++) {
        permutation.push_back(Integer(ndim - i - 1));
      }
      return permutation;
    }
  };

  auto permute_dims_axes_are_compatible = [&](const Array<Expr>& permute_dims) -> bool {
    auto first_axes = get_permute_dims_axes(permute_dims[0]);
    for (size_t i_arg = 1; i_arg < permute_dims.size(); i_arg++) {
      auto i_axes = get_permute_dims_axes(permute_dims[i_arg]);
      if (i_axes.size() != first_axes.size()) {
        return false;
      }
      for (size_t i_axis = 0; i_axis < first_axes.size(); i_axis++) {
        if (i_axes[i_axis]->value != first_axes[i_axis]->value) {
          return false;
        }
      }
    }
    return true;
  };

  auto rewriter = [=](Expr expr, Map<DFPattern, Expr> matches) -> Expr {
    Array<Expr> args;
    Array<Expr> all_permute_dims;
    for (size_t i = 0; i < max_concat; i++) {
      if (auto permute_dim_expr = matches.Get(pat_permute_dims[i])) {
        all_permute_dims.push_back(permute_dim_expr.value());
        args.push_back(matches[pat_args[i]]);
      }
    }

    ICHECK_GE(all_permute_dims.size(), min_concat)
        << "InternalError: "
        << "Pattern match should return at least " << min_concat << " items, but only found "
        << all_permute_dims.size() << ": " << all_permute_dims;

    if (!permute_dims_axes_are_compatible(all_permute_dims)) {
      return expr;
    }
    Optional<Array<Integer>> permute_axes = get_permute_dims_optional_axes(all_permute_dims[0]);

    Call concat_call = Downcast<Call>(matches[pat_concat]);
    auto concat_attrs = concat_call->attrs.as<ConcatAttrs>();
    ICHECK(concat_attrs);

    auto old_concat_axis = [&]() -> size_t {
      if (concat_attrs->axis.defined()) {
        return concat_attrs->axis.value()->value;
      } else {
        return 0;
      }
    }();
    Integer new_concat_axis = get_permute_dims_axes(all_permute_dims[0])[old_concat_axis];

    auto new_concat = concat(Tuple(args), new_concat_axis);
    auto new_permute_dims = permute_dims(new_concat, permute_axes);

    return new_permute_dims;
  };

  return {pat_concat, rewriter};
}

}  // namespace

namespace transform {
Pass ReorderPermuteDimsAfterConcat() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    auto [pattern, rewriter] = CreatePatterns();
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "ReorderPermuteDimsAfterConcat", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ReorderPermuteDimsAfterConcat")
    .set_body_typed(ReorderPermuteDimsAfterConcat);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
