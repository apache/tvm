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
#include <tvm/relax/expr_functor.h>
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

std::unordered_map<size_t, std::vector<size_t>> GroupShapes(
    const std::vector<Array<PrimExpr>>& shapes) {
  std::unordered_map<size_t, std::vector<size_t>> indices_map;
  for (size_t i = 0; i < shapes.size(); ++i) {
    indices_map[shapes[i].size()].push_back(i);
  }
  return indices_map;
}

inline TensorStructInfo GetTensorSInfo(Expr e) {
  return Downcast<TensorStructInfo>(GetStructInfo(e));
}

struct BranchInfo {
  int num_branches;
  bool has_bias;
  std::optional<std::string> activation;
};

Function Rewrite(Function f, const BranchInfo& branch_info) {
  PatternContext ctx;
  ctx.EnterWithScope();

  auto input_pattern = Wildcard();
  std::vector<WildcardPattern> rhs_patterns;
  std::vector<WildcardPattern> bias_patterns;
  std::vector<CallPattern> matmul_patterns, bias_add_patterns, activation_patterns;

  for (int i = 0; i < branch_info.num_branches; ++i) {
    auto w_pat = Wildcard();
    rhs_patterns.push_back(w_pat);
    auto matmul_pat = IsOp("relax.matmul")(input_pattern, w_pat);
    matmul_patterns.push_back(matmul_pat);
    ctx.add_constraint(input_pattern, matmul_pat, PairCons(PairCons::kUsedBy, 0));
    ctx.add_constraint(w_pat, matmul_pat, PairCons(PairCons::kUsedBy, 1));

    CallPattern matmul_out = matmul_pat;

    if (branch_info.has_bias) {
      auto bias_pat = Wildcard();
      bias_patterns.push_back(bias_pat);
      auto bias_add = IsOp("relax.add")(matmul_pat, bias_pat);
      bias_add_patterns.push_back(bias_add);
      matmul_out = bias_add;
    }

    if (branch_info.activation) {
      activation_patterns.push_back(IsOp(*branch_info.activation)(matmul_out));
    }

  }

  runtime::TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>)> rewriter =
      [=](Map<DFPattern, Var> matchings) {
        auto inp = matchings[input_pattern];
        auto lhs_dim = GetTensorSInfo(inp)->ndim;

        std::vector<Array<PrimExpr>> rhs_shapes;
        for (const auto& rhs_pat : rhs_patterns) {
          auto r = matchings[rhs_pat];
          rhs_shapes.push_back(GetTensorSInfo(r)->GetShape().value());
        }

        auto shape_groups = GroupShapes(rhs_shapes);

        Map<Var, Expr> replacements;

        for (const auto& [rhs_dim, indices] : shape_groups) {
          if (indices.size() == 1) continue;

          Array<Expr> rhs;
          for (auto ind : indices) {
            rhs.push_back(matchings[rhs_patterns[ind]]);
          }

          auto concat_rhs = concat(Tuple(rhs), Integer(rhs_dim - 1));
          auto out_dtype = GetTensorSInfo(matchings[matmul_patterns[indices[0]]])->dtype;
          auto matmul_combined = matmul(inp, concat_rhs, out_dtype);

          PrimExpr begin{0};
          Array<PrimExpr> strides{1};
          int slice_axis = std::max<int>(lhs_dim, rhs_dim) - 1;

          for (size_t i = 0; i < indices.size(); ++i) {
            auto width = GetTensorSInfo(rhs[i])->GetShape().value()[rhs_dim - 1];
            auto bound_var = matchings[matmul_patterns[indices[i]]];
            auto slice =
                strided_slice(matmul_combined, {slice_axis}, {begin}, {begin + width}, strides);
            replacements.Set(bound_var, slice);
            begin += width;
          }
        }

        return replacements;
      };

  auto rewritten = RewriteBindings(ctx, rewriter, f);
  ctx.ExitWithScope();
  return rewritten;
}

std::vector<BranchInfo> GetBranchInfo(Function f) {
  std::unordered_map<const VarNode*, BranchInfo> groups;
  static auto matmul_op = Op::Get("relax.matmul");

  PostOrderVisit(f, [&](const Expr& e) {
    if (auto call = e.as<CallNode>(); call && call->op.same_as(matmul_op)) {
      auto lhs = Downcast<Var>(call->args[0]);
      if (auto it = groups.find(lhs.get()); it == groups.end()) {
        groups[lhs.get()] = {1, false, std::nullopt};
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
    f = Rewrite(f, branch);
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
