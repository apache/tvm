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
 * \file combine_parallel_dense.cc
 * \brief Combine parallel dense ops into a single dense.
 *
 * This pass replaces dense ops that share the same input node, same shape,
 * and don't have "units" defined with a single batch matrix multiplication.
 * The inputs of the new batch_matmul is the stack of the original inputs.
 * Elemwise and broadcast ops following dense are also combined if possible.
 *
 * This prevents launching multiple kernels in networks with multiple
 * dense branches, such as BERT.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <unordered_map>
#include <unordered_set>
#include "./expr_subst.h"
#include "pattern_util.h"
#include "./combine_parallel_op_batch.h"

namespace tvm {
namespace relay {

class ParallelDenseCombiner : public ParallelOpBatchCombiner {
 public:
  explicit ParallelDenseCombiner(uint64_t min_num_branches)
    : ParallelOpBatchCombiner("nn.dense", "nn.batch_matmul", min_num_branches) {
  }

 protected:
  virtual bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    StructuralEqual eq;
    const auto* attrs_a = a->attrs.as<DenseAttrs>();
    const auto* attrs_b = b->attrs.as<DenseAttrs>();
    CHECK(attrs_a);
    CHECK(attrs_b);
    const auto* weight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* weight_b = b->args[1]->type_as<TensorTypeNode>();

    return eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
           eq(weight_a->shape[0], weight_b->shape[0]) &&
           eq(weight_a->shape[1], weight_b->shape[1]);
  }
};

/*! \brief Combine parallel dense if number of branches >= min_num_branches */
Expr CombineParallelDense(const Expr& expr, uint64_t min_num_branches) {
  return ParallelDenseCombiner(min_num_branches).Combine(expr);
}

namespace transform {

Pass CombineParallelDense(uint64_t min_num_branches) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(CombineParallelDense(f, min_num_branches));
  };
  return CreateFunctionPass(pass_func, 4, "CombineParallelDense", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CombineParallelDense")
.set_body_typed(CombineParallelDense);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
