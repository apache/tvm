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

using FCheck = runtime::TypedPackedFunc<bool(Var, Array<Var>, Array<Var>, Map<Var, Expr>)>;

/*! \brief Group shapes of the RHS matrices by rank. Matrices in a group whose batch sizes
  are compatible are combined.
*/
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

struct Patterns {
  Patterns() : input(Wildcard()) { ctx.EnterWithScope(); }

  PatternContext ctx;
  WildcardPattern input;
  std::vector<WildcardPattern> rhs;
  std::vector<WildcardPattern> bias;
  std::vector<CallPattern> matmul;
  std::vector<CallPattern> bias_add;
  std::vector<CallPattern> activation;
};

struct SplitInfo {
  Var rhs;
  Optional<Var> bias;
  PrimExpr split_size;
  DFPattern pattern_to_replace;
};

Patterns CreatePatterns(const BranchInfo& branch_info) {
  Patterns patterns;

  for (int i = 0; i < branch_info.num_branches; ++i) {
    auto w_pat = Wildcard();
    auto matmul_pat = IsOp("relax.matmul")(patterns.input, w_pat);
    patterns.rhs.push_back(w_pat);
    patterns.matmul.push_back(matmul_pat);
    patterns.ctx.add_constraint(patterns.input, matmul_pat, PairCons(PairCons::kUsedBy, 0));
    patterns.ctx.add_constraint(w_pat, matmul_pat, PairCons(PairCons::kUsedBy, 1));

    CallPattern matmul_out = matmul_pat;

    if (branch_info.bias_dim) {
      auto bias_pat = Wildcard();
      auto bias_add_pat = IsOp("relax.add")(matmul_pat, bias_pat);
      patterns.bias.push_back(bias_pat);
      patterns.bias_add.push_back(bias_add_pat);
      patterns.ctx.add_constraint(matmul_pat, bias_add_pat, PairCons(PairCons::kUsedBy, 0));
      patterns.ctx.add_constraint(bias_pat, bias_add_pat, PairCons(PairCons::kUsedBy, 1));
      matmul_out = bias_add_pat;
    }

    if (branch_info.activation) {
      auto act_pat = IsOp(*branch_info.activation)(matmul_out);
      patterns.activation.push_back(act_pat);
      patterns.ctx.add_constraint(matmul_out, act_pat, PairCons(PairCons::kUsedBy, 0));
    }
  }

  return patterns;
}

/*! \brief Create a rewriter for the given parallel matmul branches. */
runtime::TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>, Map<Var, Expr>)> GetRewriter(
    const Patterns& patterns, const BranchInfo& branch_info, FCheck check) {
  auto batch_dims_compatible = [](size_t rhs_dim, const std::vector<size_t>& indices,
                                  const std::vector<Array<PrimExpr>>& rhs_shapes) {
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

  return [=](Map<DFPattern, Var> matchings, Map<Var, Expr> bindings) {
    std::vector<Array<PrimExpr>> rhs_shapes;
    for (const auto& rhs_pat : patterns.rhs) {
      auto rhs_shape_opt = GetTensorSInfo(matchings[rhs_pat])->GetShape();
      if (!rhs_shape_opt) {
        return Map<Var, Expr>{};
      }
      rhs_shapes.push_back(rhs_shape_opt.value());
    }

    Map<Var, Expr> replacements;

    for (const auto& [rhs_dim, indices] : GroupShapes(rhs_shapes)) {
      if (indices.size() == 1 || !batch_dims_compatible(rhs_dim, indices, rhs_shapes)) continue;

      auto lhs = matchings[patterns.input];

      const auto& patterns_to_replace = [&patterns, &branch_info]() {
        if (branch_info.activation) return patterns.activation;
        if (branch_info.bias_dim) return patterns.bias_add;
        return patterns.matmul;
      }();

      std::vector<SplitInfo> splits;
      for (auto index : indices) {
        Var rhs = matchings[patterns.rhs[index]];
        Optional<Var> bias = NullOpt;
        if (branch_info.bias_dim.has_value()) {
          bias = matchings[patterns.bias[index]];
        }
        PrimExpr split_size = GetTensorSInfo(rhs)->GetShape().value()[rhs_dim - 1];
        DFPattern pattern_to_replace = patterns_to_replace[index];
        splits.push_back(SplitInfo{rhs, bias, split_size, pattern_to_replace});
      }
      // At most one dynamic output shape can be part of the combined
      // matmul, and it must be the last item in the split.  Use
      // `std::stable_sort` instead of `std::sort` to maintain a
      // consistent order for all static shapes, and to consistently
      // select the same dynamic weight to participate.
      auto is_dynamic_split = [](const SplitInfo& split) -> bool {
        return !split.split_size->IsInstance<IntImmNode>();
      };
      std::stable_sort(splits.begin(), splits.end(),
                       [&is_dynamic_split](const auto& a, const auto& b) {
                         return is_dynamic_split(a) < is_dynamic_split(b);
                       });
      // Remove anything after the first dynamic shape participating
      // in the combined matmul.
      if (auto it = std::find_if(splits.begin(), splits.end(), is_dynamic_split);
          it != splits.end()) {
        splits.erase(it + 1, splits.end());
      }

      if (splits.size() == 1) {
        continue;
      }

      Array<Var> rhs;
      Array<Var> bias;
      for (const auto& split : splits) {
        rhs.push_back(split.rhs);
        if (split.bias) {
          bias.push_back(split.bias.value());
        }
      }

      if (!check(lhs, rhs, bias, bindings)) {
        continue;
      }

      auto concat_rhs = concat(Tuple(rhs), Integer(rhs_dim - 1));
      auto out_dtype = GetTensorSInfo(matchings[patterns.matmul[indices[0]]])->dtype;
      auto matmul_combined = matmul(lhs, concat_rhs, out_dtype);

      if (branch_info.bias_dim) {
        auto bias_dim = GetTensorSInfo(bias[0])->ndim;
        auto concat_bias = concat(Tuple(bias), Integer(bias_dim - 1));
        matmul_combined = add(matmul_combined, concat_bias);
      }

      if (branch_info.activation) {
        if (*branch_info.activation == "relax.nn.relu") {
          matmul_combined = relu(matmul_combined);
        } else if (*branch_info.activation == "relax.nn.gelu") {
          matmul_combined = gelu(matmul_combined);
        } else if (*branch_info.activation == "relax.nn.gelu_tanh") {
          matmul_combined = gelu_tanh(matmul_combined);
        } else if (*branch_info.activation == "relax.nn.silu") {
          matmul_combined = silu(matmul_combined);
        } else {
          LOG(FATAL) << "Unsupported activation: " << *branch_info.activation;
        }
      }

      int split_index = 0;
      Array<IntImm> sections;
      for (size_t i = 0; i + 1 < splits.size(); i++) {
        auto width = splits[i].split_size.as<IntImmNode>();
        ICHECK(width) << "InternalError: "
                      << "All splits except the last one must have a static shape";
        split_index += width->value;
        sections.push_back(IntImm(DataType::Int(64), split_index));
      }

      int lhs_dim = GetTensorSInfo(lhs)->ndim;
      int split_axis = std::max<int>(lhs_dim, rhs_dim) - 1;
      auto chunks = split(matmul_combined, sections, split_axis);

      for (size_t i = 0; i < splits.size(); i++) {
        const auto& split = splits[i];
        auto bound_var = matchings[split.pattern_to_replace];
        replacements.Set(bound_var, TupleGetItem(chunks, i));
      }
    }

    return replacements;
  };
}

Function Rewrite(Function f, const BranchInfo& branch_info, FCheck check) {
  auto patterns = CreatePatterns(branch_info);
  auto rewriter = GetRewriter(patterns, branch_info, check);
  return RewriteBindings(patterns.ctx, rewriter, f);
}

/*! \brief Look for subtrees with parallel matmul and return information about
  them (the number of branches and the kind of fused ops)
*/
std::vector<BranchInfo> GetBranchInfo(Function f) {
  auto bias_pat = Wildcard();
  auto matmul_pat = IsOp("relax.matmul")(Wildcard(), Wildcard());
  auto bias_add_pat = IsOp("relax.add")(matmul_pat, bias_pat);

  std::vector<std::string> activations{"relax.nn.relu", "relax.nn.gelu", "relax.nn.gelu_tanh",
                                       "relax.nn.silu"};

  std::vector<DFPattern> activation_pat, bias_activation_pat;
  for (const auto& act : activations) {
    activation_pat.push_back(IsOp(act)(matmul_pat));
    bias_activation_pat.push_back(IsOp(act)(bias_add_pat));
  }

  auto bindings = AnalyzeVar2Value(f);

  auto create_group = [&](DFPattern pat) {
    // Maps a LHS matrix to consumer parallel matmuls
    std::unordered_map<const VarNode*, BranchInfo> groups;

    PostOrderVisit(f, [&](const Expr& e) {
      if (!e->IsInstance<CallNode>()) return;

      auto match = ExtractMatchedExpr(pat, e, bindings);
      if (!match) return;

      auto matmul_call = Downcast<Call>(match.value()[matmul_pat]);
      auto matmul_lhs = Downcast<Var>(matmul_call->args[0]);

      std::optional<int> bias_dim = std::nullopt;
      std::optional<std::string> activation = std::nullopt;

      if (match.value().count(bias_pat)) {
        bias_dim = GetTensorSInfo(match.value()[bias_pat])->ndim;
      }

      for (size_t i = 0; i < activations.size(); ++i) {
        if (match.value().count(activation_pat[i]) || match.value().count(bias_activation_pat[i])) {
          activation = activations[i];
        }
      }

      if (auto it = groups.find(matmul_lhs.get()); it != groups.end()) {
        // Create a new branch in the existing parallel matmul subtree, and
        // invalidate bias and activation information when needed.
        BranchInfo* branch = &it->second;

        branch->num_branches += 1;

        if (!bias_dim || (branch->bias_dim && *branch->bias_dim != *bias_dim)) {
          branch->bias_dim = std::nullopt;
        }

        if (!activation || (branch->activation && *branch->activation != *activation)) {
          branch->activation = std::nullopt;
        }
      } else {
        // Create a new subgraph with one matmul
        groups[matmul_lhs.get()] = {1, bias_dim, activation};
      }
    });

    return groups;
  };

  std::unordered_map<const VarNode*, BranchInfo> groups_activation;
  for (size_t i = 0; i < activations.size(); ++i) {
    auto groups = create_group(bias_activation_pat[i]);
    groups_activation.merge(std::move(groups));
  }

  for (size_t i = 0; i < activations.size(); ++i) {
    auto groups = create_group(activation_pat[i]);
    groups_activation.merge(std::move(groups));
  }

  auto groups_bias = create_group(bias_add_pat);
  auto groups_matmul = create_group(matmul_pat);

  for (const auto& groups : {groups_bias, groups_activation}) {
    for (const auto& [lhs, branch] : groups) {
      // Prefer combining more matmuls than combining fewer ones and leaving additional uncombined
      // matmuls followed by bias or activation. So we combine matmuls + fused ops patterns only
      // when all branches have the same fused ops.
      if (auto it = groups_matmul.find(lhs);
          it != groups_matmul.end() && it->second.num_branches == branch.num_branches) {
        it->second = branch;
      }
    }
  }

  std::vector<BranchInfo> info;

  for (const auto& groups : {groups_matmul, groups_activation, groups_bias}) {
    for (const auto& group : groups) {
      if (group.second.num_branches > 1) {
        info.push_back(group.second);
      }
    }
  }

  return info;
}

Function CombineParallelMatmul(Function f, FCheck check) {
  auto branches = GetBranchInfo(f);
  std::sort(branches.begin(), branches.end(),
            [](const auto& b1, const auto& b2) { return b1.num_branches > b2.num_branches; });

  for (const auto& branch : branches) {
    f = Rewrite(f, branch, check);
  }
  return f;
}

namespace transform {

Pass CombineParallelMatmul(FCheck check) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return relax::CombineParallelMatmul(f, check);
      };
  return CreateFunctionPass(/*pass_function=*/pass_func,            //
                            /*opt_level=*/0,                        //
                            /*pass_name=*/"CombineParallelMatmul",  //
                            /*required=*/{});
}

TVM_REGISTER_GLOBAL("relax.transform.CombineParallelMatmul").set_body_typed(CombineParallelMatmul);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
