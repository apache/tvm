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
 * \file tvm/relax/transform/expand_matmul_of_sum.cc
 * \brief Expand `matmul(x, A+B)` to `matmul(x, A) + matmul(x,B)`
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
  auto pat_lhs = WildcardPattern();

  auto pat_weights = WildcardPattern();
  auto pat_indices = WildcardPattern();
  auto pat_rhs = IsOp("relax.take")(pat_weights, pat_indices);

  auto pat_matmul = IsOp("relax.matmul")(pat_lhs, pat_rhs);

  auto rewriter = [=](Expr expr, Map<DFPattern, Expr> matches) -> Expr {
    auto lhs = matches[pat_lhs];
    auto weights = matches[pat_weights];
    auto indices = matches[pat_indices];

    const auto* take_call = matches[pat_rhs].as<CallNode>();
    ICHECK(take_call) << "InternalError: "
                      << "Match of relax.take operator should produce Call, "
                      << "but instead produces " << matches[pat_rhs] << " with type "
                      << matches[pat_rhs]->GetTypeKey();
    const auto* attrs = take_call->attrs.as<TakeAttrs>();
    ICHECK(attrs) << "InternalError: "
                  << "Attributes for relax.take operator should be TakeAttrs, "
                  << "but were instead " << take_call->attrs << " with type "
                  << take_call->GetTypeKey();

    const auto* lhs_sinfo = lhs->struct_info_.as<TensorStructInfoNode>();
    if (!lhs_sinfo) return expr;

    const auto* weights_sinfo = weights->struct_info_.as<TensorStructInfoNode>();
    if (!weights_sinfo) return expr;

    const auto* indices_sinfo = indices->struct_info_.as<TensorStructInfoNode>();
    if (!indices_sinfo) return expr;

    const auto* matmul_sinfo = expr->struct_info_.as<TensorStructInfoNode>();
    if (!matmul_sinfo) return expr;

    if (!attrs->axis.defined()) return expr;
    auto axis = attrs->axis.value()->value;

    if (lhs_sinfo->IsUnknownNdim() || indices_sinfo->IsUnknownNdim() ||
        matmul_sinfo->IsUnknownNdim() || weights_sinfo->IsUnknownNdim())
      return expr;

    if (indices_sinfo->ndim == 1 && axis + 1 == weights_sinfo->ndim) {
      // Simpler case.  The activations may have batch dimensions, but
      // the weights do not.

      // lhs.shape = [*batch, infeatures]
      // weights.shape = [infeatures, table_size]
      // indices.shape = [outfeatures]

      // out_table.shape = [*batch, table_size]
      auto out_table = matmul(lhs, weights, DataType::Void());
      // new_output.shape = [*batch, outfeatures]
      auto new_output = take(out_table, indices, Integer(matmul_sinfo->ndim - 1));

      return new_output;
    } else if (lhs_sinfo->ndim == 3 && weights_sinfo->ndim == 3 && indices_sinfo->ndim == 1 &&
               axis == 0 && weights_sinfo->GetShape().defined() &&
               lhs_sinfo->GetShape().defined()) {
      // More complicated case, used for batched LoRA.  The conditions
      // on the argument dimensions can probably be relaxed, but would
      // probably need to remove the use of the einsum operator.

      auto lhs_shape = lhs_sinfo->GetShape().value();
      auto weight_shape = weights_sinfo->GetShape().value();

      // lhs.shape = [batch1, batch2, infeatures]
      // weights.shape = [table_size, infeatures, outfeatures]
      // indices.shape = [batch1]

      // reordered_weight.shape = [infeatures, table_size, outfeatures]
      auto reordered_weight = permute_dims(weights, Array{Integer(1), Integer(0), Integer(2)});
      // fused_weight.shape = [infeatures, table_size * outfeatures]
      auto fused_weight = reshape(reordered_weight,
                                  ShapeExpr({weight_shape[1], weight_shape[0] * weight_shape[2]}));
      // fused_output.shape = [batch1, batch2, table_size * outfeatures]
      auto fused_output = matmul(lhs, fused_weight, DataType::Void());
      // indexed_output.shape = [batch1, batch2, table_size, outfeatures]
      auto indexed_output = reshape(
          fused_output, ShapeExpr({lhs_shape[0], lhs_shape[1], weight_shape[0], weight_shape[2]}));

      // TODO(Lunderberg): Find a better way to express these last two
      // steps.  For an output at [i,j,k], the value is
      // `indexed_output[i, j, indices[i], k]`, but there doesn't seem
      // to be a good way to express that in relax.  It could be
      // written using `call_te`, but that would prevent later
      // optimizations from recognizing the high-level relax
      // operations.

      // duplicated_output.shape = [batch1, batch2, batch1, outfeatures]
      auto duplicated_output = take(indexed_output, indices, Integer(2));
      // new_output.shape = [batch1, batch2, outfeatures]
      auto new_output = einsum(Tuple({duplicated_output}), "ijik->ijk");

      return new_output;
    } else {
      return expr;
    }
  };

  return {pat_matmul, rewriter};
}

}  // namespace

namespace transform {
Pass ReorderTakeAfterMatmul() {
  auto pass_func = [=](Function func, IRModule mod, PassContext pc) {
    auto [pattern, rewriter] = CreatePatterns();
    return RewriteCall(pattern, rewriter, func);
  };
  return CreateFunctionPass(pass_func, 1, "ReorderTakeAfterMatmul", {});
}

TVM_REGISTER_GLOBAL("relax.transform.ReorderTakeAfterMatmul")
    .set_body_typed(ReorderTakeAfterMatmul);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
