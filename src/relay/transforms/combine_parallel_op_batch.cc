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
 * \file combine_parallel_op_batch.cc
 * \brief Combine parallel ops into a single batch op.
 *
 * This pass replaces ops that share the same input node and same shape
 * with a single op that takes in batched input. The inputs of the new
 * batched op are the stack of the original inputs. Elementwise and
 * broadcast ops following the original op are also stacked
 * and fused if possible. For example:
 *
 *            data
 *         /         \
 *    add (2,2)     add (2,2)
 *      |            |
 * elemwise (2,2)   elemwise (2,2)
 *      |            |
 *
 * Would become:
 *
 *            data
 *              |
 *       add+elemwise (2,2,2)
 *          /       \
 *
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
#include "./combine_parallel_op.h"
#include "./combine_parallel_op_batch.h"
#include "pattern_util.h"

namespace tvm {
namespace relay {

ParallelOpBatchCombiner::ParallelOpBatchCombiner(const std::string& op_name,
                                                 const std::string& batch_op_name,
                                                 uint64_t min_num_branches)
  : ParallelOpCombiner(op_name, min_num_branches),
    batch_op_name_(batch_op_name) {
}

bool ParallelOpBatchCombiner::IsSupportedOp(const CallNode* n) {
  return true;
}

bool ParallelOpBatchCombiner::CanOpsBeCombined(const CallNode* a, const CallNode* b) {
  if (a->args.size() != b->args.size()) {
    return false;
  }

  StructuralEqual eq;
  for (size_t i = 0; i < a->args.size(); i++) {
    auto ta = a->args[i]->type_as<TensorTypeNode>();
    auto tb = b->args[i]->type_as<TensorTypeNode>();
    if (ta->shape.size() != tb->shape.size() || !eq(ta->dtype, tb->dtype)) {
      return false;
    }

    for (size_t j = 0; j < ta->shape.size(); j++) {
      if (!eq(ta->shape[j], tb->shape[j])) {
        return false;
      }
    }
  }

  return true;
}

Call ParallelOpBatchCombiner::MakeCombinedOp(const Group& branches) {
  const Op& batch_op = Op::Get(batch_op_name_);

  Array<Expr> new_args;
  size_t num_args = branches[0][0]->args.size();
  for (size_t i = 0; i < num_args; i++) {
    Array<Expr> arg_from_all_branches;
    for (const auto& branch : branches) {
      arg_from_all_branches.push_back(branch[0]->args[i]);
    }

    new_args.push_back(MakeStack(Tuple(arg_from_all_branches), 0));
  }

  return Call(batch_op, new_args, Attrs(), {});
}

bool ParallelOpBatchCombiner::IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) {
  StructuralEqual eq;
  auto ta = a->args[index]->type_as<TensorTypeNode>();
  auto tb = b->args[index]->type_as<TensorTypeNode>();

  if (!eq(ta->dtype, tb->dtype) || ta->shape.size() != tb->shape.size())
    return false;

  for (size_t i = 0; i < ta->shape.size(); i++) {
    if (!eq(ta->shape[i], tb->shape[i]))
      return false;
  }
  return true;
}

Call ParallelOpBatchCombiner::MakeCombinedCallFromFollowingOps(const Expr& data,
                                                               const Group& branches,
                                                               size_t depth,
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
      // if the shape of the arg is of shape (j,),
      // expand it to (1,j) so it can be properly broadcasted.
      Expr arg = branch[depth]->args[i];
      const TensorTypeNode* arg_tensor = arg->type_as<TensorTypeNode>();
      if (arg_tensor->shape.size() == 1) {
        Expr expanded_arg = MakeExpandDims(arg, 0, 1);
        tuple.push_back(expanded_arg);
      } else {
        tuple.push_back(arg);
      }
    }

    auto stack = MakeStack(Tuple(tuple), 0);
    new_args.push_back(std::move(stack));
  }

  return Call(call->op, new_args, call->attrs, {});
}

void ParallelOpBatchCombiner::UpdateGroupOutput(const Expr& data,
                        const Group& branches,
                        size_t depth,
                        ExprSubstMap* subst_map) {
  int index = 0;
  auto split = MakeSplit(data, Integer(branches.size()), 0);
  for (const auto& branch : branches) {
    auto split_data = TupleGetItem(split, index++);
    auto squeezed_data = MakeSqueeze(split_data, {0});
    subst_map->insert({GetRef<Expr>(branch[depth]), squeezed_data});
  }
}

/*! \brief Combine parallel op into batched op if number of branches >= min_num_branches */
Expr CombineParallelOpBatch(const Expr& expr,
                            const std::string& op_name,
                            const std::string& batch_op_name,
                            uint64_t min_num_branches) {
  return ParallelOpBatchCombiner(op_name, batch_op_name, min_num_branches).Combine(expr);
}

namespace transform {

Pass CombineParallelOpBatch(const std::string& op_name,
                            const std::string& batch_op_name,
                            uint64_t min_num_branches) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(CombineParallelOpBatch(f,
                                                       op_name,
                                                       batch_op_name,
                                                       min_num_branches));
  };
  return CreateFunctionPass(pass_func, 4, "CombineParallelOpBatch", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CombineParallelOpBatch")
.set_body_typed(CombineParallelOpBatch);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
