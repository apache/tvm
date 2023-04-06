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

#include <tvm/arith/analyzer.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/dataflow_matcher.h>
#include <tvm/relax/dataflow_pattern.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>
#include <tvm/relax/transform.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "../op/nn/nn.h"
#include "../op/tensor/binary.h"
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
  std::optional<int> bias_dim;
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

    if (branch_info.bias_dim) {
      auto bias_pat = Wildcard();
      bias_patterns.push_back(bias_pat);
      auto bias_add = IsOp("relax.add")(matmul_pat, bias_pat);
      ctx.add_constraint(matmul_pat, bias_add, PairCons(PairCons::kUsedBy, 0));
      ctx.add_constraint(bias_pat, bias_add, PairCons(PairCons::kUsedBy, 1));
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
          auto rhs_shape_opt = GetTensorSInfo(r)->GetShape();
          if (!rhs_shape_opt) {
            return Map<Var, Expr>{};
          }
          rhs_shapes.push_back(rhs_shape_opt.value());
        }

        auto batch_dims_compatible = [&rhs_shapes](int rhs_dim,
                                                   const std::vector<size_t>& indices) {
          arith::Analyzer ana;
          for (auto ind : indices) {
            ICHECK_EQ(static_cast<int>(rhs_shapes[ind].size()), rhs_dim);
            // -2 for reduction and concat axes
            for (size_t i = 0; i < rhs_dim - 2; ++i) {
              if (!ana.CanProve(rhs_shapes[indices[0]][i] == rhs_shapes[ind][i])) {
                return false;
              }
            }
          }
          return true;
        };

        Map<Var, Expr> replacements;

        for (const auto& [rhs_dim, indices] : GroupShapes(rhs_shapes)) {
          if (indices.size() == 1 || !batch_dims_compatible(rhs_dim, indices)) continue;

          Array<Expr> rhs, bias;
          for (auto ind : indices) {
            rhs.push_back(matchings[rhs_patterns[ind]]);
            if (branch_info.bias_dim) {
              ICHECK(matchings.count(bias_patterns[ind]));
              bias.push_back(matchings[bias_patterns[ind]]);
            }
          }

          auto concat_rhs = concat(Tuple(rhs), Integer(rhs_dim - 1));
          auto out_dtype = GetTensorSInfo(matchings[matmul_patterns[indices[0]]])->dtype;
          auto matmul_combined = matmul(inp, concat_rhs, out_dtype);

          auto pattern_to_replace = &matmul_patterns;

          if (branch_info.bias_dim) {
            auto bias_dim = GetTensorSInfo(bias[0])->ndim;
            for (auto b : bias) {
              ICHECK(GetTensorSInfo(b)->ndim == bias_dim);
            }
            auto concat_bias = concat(Tuple(bias), Integer(bias_dim - 1));
            matmul_combined = add(matmul_combined, concat_bias);
            pattern_to_replace = &bias_add_patterns;
          }

          if (branch_info.activation) {
            pattern_to_replace = &activation_patterns;
            if (*branch_info.activation == "relu") {
              matmul_combined = relu(matmul_combined);
            } else if (*branch_info.activation == "gelu") {
              matmul_combined = gelu(matmul_combined);
            } else if (*branch_info.activation == "silu") {
              matmul_combined = silu(matmul_combined);
            } else {
              LOG(FATAL) << "Unsupported activation: " << *branch_info.activation;
            }
          }

          PrimExpr begin{0};
          Array<PrimExpr> strides{1};
          int slice_axis = std::max<int>(lhs_dim, rhs_dim) - 1;

          for (size_t i = 0; i < indices.size(); ++i) {
            auto width = GetTensorSInfo(rhs[i])->GetShape().value()[rhs_dim - 1];
            auto bound_var = matchings[(*pattern_to_replace)[indices[i]]];
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
  auto lhs_pat = Wildcard();
  auto rhs_pat = Wildcard();
  auto bias_pat = Wildcard();

  auto matmul_pat = IsOp("relax.matmul")(lhs_pat, rhs_pat);
  auto bias_add_pat = IsOp("relax.add")(matmul_pat, bias_pat);

  std::vector<std::string> activations{"relax.nn.relu", "relax.nn.gelu", "relax.nn.silu"};

  std::vector<DFPattern> activation_pat, bias_activation_pat;
  for (const auto& act : activations) {
    activation_pat.push_back(IsOp(act)(matmul_pat));
    bias_activation_pat.push_back(IsOp(act)(bias_add_pat));
  }

  auto bindings = AnalyzeVar2Value(f);

  std::unordered_map<const VarNode*, BranchInfo> groups_activation, groups_bias, groups_matmul;

  PostOrderVisit(f, [&](const Expr& e) {
    if (!e->IsInstance<CallNode>()) return;
    if (auto match = ExtractMatchedExpr(bias_add_pat, e, bindings)) {
      auto matmul_call = Downcast<Call>(match.value()[matmul_pat]);
      auto matmul_lhs = Downcast<Var>(matmul_call->args[0]);
      auto bias_dim = GetTensorSInfo(match.value()[bias_pat])->ndim;
      if (auto it = groups_bias.find(matmul_lhs.get()); it == groups_bias.end()) {
        groups_bias[matmul_lhs.get()] = {1, bias_dim, std::nullopt};
      } else {
        it->second.num_branches += 1;
        if (it->second.bias_dim && *it->second.bias_dim != bias_dim) {
          it->second.bias_dim = std::nullopt;
        }
      }
      return;
    }
  });

  PostOrderVisit(f, [&](const Expr& e) {
    if (!e->IsInstance<CallNode>()) return;
    if (auto match = ExtractMatchedExpr(matmul_pat, e, bindings)) {
      auto matmul_call = Downcast<Call>(match.value()[matmul_pat]);
      auto matmul_lhs = Downcast<Var>(matmul_call->args[0]);
      if (groups_bias.count(matmul_lhs.get()) || groups_activation.count(matmul_lhs.get())) return;
      if (auto it = groups_matmul.find(matmul_lhs.get()); it == groups_matmul.end()) {
        groups_matmul[matmul_lhs.get()] = {1, std::nullopt, std::nullopt};
      } else {
        it->second.num_branches += 1;
      }
      return;
    }
  });

  // for (size_t i = 0; i < activations.size(); ++i) {
  //   if (auto match = ExtractMatchedExpr(bias_activation_pat[i], e, bindings)) {
  //     auto matmul_lhs = Downcast<Var>(match.value()[lhs_pat]);
  //     auto bias_dim = GetTensorSInfo(match.value()[bias_pat])->ndim;
  //     if (auto it = groups.find(matmul_lhs.get()); it == groups.end()) {
  //       groups[matmul_lhs.get()] = {1, bias_dim, activations[i]};
  //     } else {
  //       it->second.num_branches += 1;

  //       if (it->second.bias_dim != bias_dim) {
  //         it->second.bias_dim = std::nullopt;
  //       }

  //       if (!it->second.activation || (*it->second.activation != activations[i])) {
  //         it->second.activation = std::nullopt;
  //       }
  //     }

  //     for (auto pat : {matmul_pat, bias_add_pat}) {
  //       seen.insert(match.value()[pat].get());
  //     }
  //     return;
  //   }
  //   if (auto match = ExtractMatchedExpr(activation_pat[i], e, bindings)) {
  //     auto matmul = match.value()[matmul_pat];
  //     auto matmul_lhs = Downcast<Var>(match.value()[lhs_pat]);
  //     if (auto it = groups.find(matmul_lhs.get()); it == groups.end()) {
  //       groups[matmul_lhs.get()] = {1, std::nullopt, activations[i]};
  //     } else {
  //       it->second.num_branches += 1;

  //       if (!it->second.activation || (*it->second.activation != activations[i])) {
  //         it->second.activation = std::nullopt;
  //       }
  //     }
  //     return;
  //   }
  // }

  std::vector<BranchInfo> info;

  for (auto groups : {groups_matmul, groups_activation, groups_bias}) {
    for (const auto& group : groups) {
      if (group.second.num_branches > 1) {
        info.push_back(group.second);
      }
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
