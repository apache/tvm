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
 *
 * \file combine_parallel_batch_matmul.cc
 * \brief Combine parallel batch matmuls into a single one.
 *
 * This pass replaces batch_matmul that share the same lhs node with a
 * single batch matmul.Elemwise and broadcast ops following batch_matmul are also
 * combined if possible.
 *
 * This prevents launching multiple kernels in networks with multiple
 * convolution branches, such as Inception block.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "./combine_parallel_op.h"
#include "./expr_subst.h"
#include "pattern_util.h"

namespace tvm {
namespace relay {

class ParallelBatchMatmulCombiner : public ParallelOpCombiner {
 public:
  explicit ParallelBatchMatmulCombiner(uint64_t min_num_branches)
      : ParallelOpCombiner("nn.batch_matmul", min_num_branches) {}

 protected:
  bool IsSupportedOp(const CallNode* n) { return true; }

  bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    StructuralEqual eq;
    const auto* rhs_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* rhs_b = b->args[1]->type_as<TensorTypeNode>();
    const auto* restype_a = a->type_as<TensorTypeNode>();
    const auto* restype_b = b->type_as<TensorTypeNode>();
    // shape[2] is the contraction axis and automatically consistent
    // if it were valid batch_matmul ops
    auto res = eq(rhs_a->dtype, rhs_b->dtype) && eq(restype_a->dtype, restype_b->dtype) &&
               (rhs_a->shape.size() == 3) && (rhs_b->shape.size() == 3) &&
               eq(rhs_a->shape[0], rhs_b->shape[0]);
    return res;
  }

  Call MakeCombinedOp(const Group& branches) {
    const Op& batch_matmul = Op::Get("nn.batch_matmul");
    Expr data = branches[0][0]->args[0];

    Array<Expr> weights;
    for (const auto& branch : branches) {
      auto batch_matmul = branch[0];
      weights.push_back(batch_matmul->args[1]);
    }
    Expr new_weight = MakeConcatenate(Tuple(weights), 1);
    return Call(batch_matmul, {data, new_weight}, {}, {});
  }

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) { return true; }

  Call MakeCombinedCallFromFollowingOps(const Expr& data, const Group& branches, size_t depth,
                                        size_t parent_index) {
    Array<Expr> new_args;
    const CallNode* call = branches[0][depth];

    for (size_t i = 0; i < call->args.size(); i++) {
      if (i == parent_index) {
        new_args.push_back(data);
        continue;
      }

      Array<Expr> tuple;
      for (const auto& branch : branches) {
        tuple.push_back(branch[depth]->args[i]);
      }

      auto concat = MakeConcatenate(Tuple(tuple), -1);
      new_args.push_back(std::move(concat));
    }

    return Call(call->op, new_args, call->attrs, {});
  }

  void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth,
                         ExprSubstMap* subst_map) {
    int64_t index = 0;

    for (const auto& branch : branches) {
      const CallNode* batch_matmul = branch[0];
      auto feature_dim = batch_matmul->args[1]->type_as<TensorTypeNode>()->shape[1];
      auto fpp = tir::as_const_int(feature_dim);
      int64_t features = *fpp;
      Array<Integer> begin;
      Array<Integer> end;
      for (size_t i = 0; i < 2; i++) {
        begin.push_back(0);
        end.push_back(-1);
      }
      begin.push_back(index);
      index += features;
      end.push_back(features);
      Array<Integer> strides(begin.size(), 1);
      auto slice = MakeStridedSlice(data, begin, end, strides, "size");
      subst_map->insert({GetRef<Expr>(branch[depth]), slice});
    }
  }
};

/*! \brief Combine parallel batch_matmul if number of branches >= min_num_branches */
Expr CombineParallelBatchMatmul(const Expr& expr, uint64_t min_num_branches) {
  return ParallelBatchMatmulCombiner(min_num_branches).Combine(expr);
}

namespace transform {

Pass CombineParallelBatchMatmul(uint64_t min_num_branches) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CombineParallelBatchMatmul(f, min_num_branches));
      };
  return CreateFunctionPass(pass_func, 4, "CombineParallelBatchMatmul", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CombineParallelBatchMatmul")
    .set_body_typed(CombineParallelBatchMatmul);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
