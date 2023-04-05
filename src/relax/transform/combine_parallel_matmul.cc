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

#include <vector>

#include "../op/tensor/index.h"
#include "../op/tensor/linear_algebra.h"
#include "../op/tensor/manipulate.h"

namespace tvm {
namespace relax {

using runtime::Map;

Function CombineParallelMatmul(Function f) {
  PatternContext ctx;
  WildcardPattern input_pattern;
  std::vector<WildcardPattern> weight_patterns;
  std::vector<CallPattern> matmul_patterns;
  const int num_branches = 32;

  runtime::TypedPackedFunc<Map<Var, Expr>(Map<DFPattern, Var>)> rewriter =
      [=](Map<DFPattern, Var> matchings) {
        auto inp = matchings[input_pattern];

        Array<Expr> weights;
        for (const auto& weight_pat : weight_patterns) {
          weights.push_back(matchings[weight_pat]);
        }

        auto concat_weights = concat(Tuple(weights), Integer(1));
        auto matmul_combined = matmul(inp, concat_weights, DataType::Float(16));

        Map<Var, Expr> replacements;
        PrimExpr begin{0};
        int slice_axis = 2;
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
  return RewriteBindings(ctx, rewriter, f);
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

}  // namespace transform

}  // namespace relax
}  // namespace tvm
