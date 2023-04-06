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
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>

#include <unordered_map>
#include <vector>

#include "../op/tensor/index.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

using runtime::Map;

static auto matmul_op = Op::Get("relax.matmul");

Function Rewrite(Function f, int num_branches, int slice_axis) {
  PatternContext ctx;
  ctx.EnterWithScope();

  auto input_pattern = Wildcard();
  std::vector<WildcardPattern> weight_patterns;
  std::vector<CallPattern> matmul_patterns;

  for (int i = 0; i < num_branches; ++i) {
    auto w_pat = Wildcard();
    CallPattern matmul_pat{ExprPattern(matmul_op), {input_pattern, w_pat}};
    weight_patterns.push_back(w_pat);
    matmul_patterns.push_back(matmul_pat);
    ctx.add_constraint(input_pattern, matmul_pat, PairCons(PairCons::kUsedBy, 0));
    ctx.add_constraint(w_pat, matmul_pat, PairCons(PairCons::kUsedBy, 1));
  }

  runtime::TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>)> rewriter =
      [=](Map<DFPattern, Var> matchings) {
        auto inp = matchings[input_pattern];

        Array<Expr> weights;
        for (const auto& weight_pat : weight_patterns) {
          weights.push_back(matchings[weight_pat]);
        }

        auto concat_weights = concat(Tuple(weights), Integer(1));  // TODO: axis
        auto out_dtype =
            Downcast<TensorStructInfo>(GetStructInfo(matchings[matmul_patterns[0]]))->dtype;
        auto matmul_combined = matmul(inp, concat_weights, out_dtype);

        Map<Var, Expr> replacements;
        PrimExpr begin{0};
        Array<PrimExpr> strides{1};

        for (size_t i = 0; i < num_branches; ++i) {
          auto sinfo = GetStructInfo(weights[i]);
          auto width = Downcast<TensorStructInfo>(sinfo)->GetShape().value()[1];
          auto bound_var = matchings[matmul_patterns[i]];
          auto slice =
              strided_slice(matmul_combined, {slice_axis}, {begin}, {begin + width}, strides);
          replacements.Set(bound_var, slice);
          begin += width;
        }

        return replacements;
      };

  auto rewritten = RewriteBindings(ctx, rewriter, f);
  ctx.ExitWithScope();
  return rewritten;
}

struct BranchInfo {
  int num_branches;
  int slice_axis;
};

std::vector<BranchInfo> GetBranchInfo(Function f) {
  std::unordered_map<const VarNode*, BranchInfo> groups;
  PostOrderVisit(f, [&](const Expr& e) {
    if (auto call = e.as<CallNode>(); call && call->op.same_as(matmul_op)) {
      auto lhs = Downcast<Var>(call->args[0]);
      if (auto it = groups.find(lhs.get()); it == groups.end()) {
        auto sinfo = GetStructInfo(e);
        auto slice_axis = Downcast<TensorStructInfo>(sinfo)->ndim - 1;
        groups[lhs.get()] = {1, slice_axis};
      } else {
        it->second.num_branches += 1;
      }
    }
  });

  std::vector<BranchInfo> info;

  for (const auto& group : groups) {
    if (group.second.num_branches > 1) {
      info.push_back(group.second);
    }
  }

  std::sort(info.begin(), info.end(),
            [](const auto& b1, const auto& b2) { return b1.num_branches > b2.num_branches; });

  return info;
}

Function CombineParallelMatmul(Function f) {
  auto branches = GetBranchInfo(f);
  for (const auto& branch : branches) {
    f = Rewrite(f, branch.num_branches, branch.slice_axis);
  }
  return f;
}

namespace transform {

Pass CombineParallelMatmul() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return relax::CombineParallelMatmul(f); };
  return CreateFunctionPass(/*pass_function=*/pass_func,            //
                            /*opt_level=*/0,                        //
                            /*pass_name=*/"CombineParallelMatmul",  //
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.CombineParallelMatmul").set_body_typed(CombineParallelMatmul);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
