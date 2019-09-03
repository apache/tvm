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
 * Copyright (c) 2019 by Contributors
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
*          add (2,2,2)
*              |
*         elemwise (2,2,2)
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
#include "./pattern_util.h"
#include "./combine_parallel_op.h"

namespace tvm {
namespace relay {

class ParallelOpBatchCombiner : public ParallelOpCombiner {
 public:
  ParallelOpBatchCombiner(const std::string& op_name, uint64_t min_num_branches)
    : ParallelOpBatchCombiner(op_name, op_name, min_num_branches) {
  }

  ParallelOpBatchCombiner(const std::string& op_name, const std::string& batched_op_name, uint64_t min_num_branches)
    : ParallelOpCombiner(op_name, min_num_branches),
      batched_op_name_(batched_op_name) {
  }

 protected:
  bool IsSupportedOp(const CallNode* n) {
    return true;
  }

  bool CanOpsBeCombined(const CallNode* a, const CallNode* b) {
    if (!eq(a->args.size(), b->args.size())) {
      return false;
    }

    for (size_t i = 0; i < a->args.size(); i++) {
      const auto* ta = a->args[i]->type_as<TensorTypeNode>();
      const auto* tb = b->args[i]->type_as<TensorTypeNode>();
      if (!eq(ta->shape.size(), tb->shape.size()) || !eq(ta->dtype, tb->dtype)) {
        return false;
      }

      for (size_t j = 0; j < ta->shape.size(); j++) {
        if (!eq(ta->shape[j], tb->shape[j])) {
          return false;
        }
      }
    }
  }

  Call MakeCombinedOp(const Group& branches) {
    const Op& batch_op = Op::Get(batched_op_name_);
    
    Array<Expr> stacked_args;
    size_t num_args = branches[0][0]->args.size();
    for (size_t i = 0; i < num_args; i++) {
      Array<Expr> new_arg;
      for (const auto& branch : branches) {
        new_arg.push_back(branch[0]->args[i]);
      }

      Expr new_arg_stack = MakeStack(TupleNode::make(new_arg), 0);
      stacked_args.push_back(new_arg_stack);
    }

    return CallNode::make(batch_op, stacked_args, Attrs(), {});
  }

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) {
    AttrsEqual eq;
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

  Call MakeCombinedCallFromFollowingOps(const Expr& data,
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
        // if the shape of the arg is 1D, expand it to (1,j) so it can be properly broadcasted.
        Expr arg = branch[depth]->args[i];
        const TensorTypeNode* arg_tensor = arg->type_as<TensorTypeNode>();
        if (arg_tensor->shape.size() == 1) {
          Expr expanded_arg = MakeExpandDims(arg, 0, 1);
          tuple.push_back(expanded_arg);
        } else {
          tuple.push_back(arg);
        }
      }

      auto stack = MakeStack(TupleNode::make(tuple), 0);
      new_args.push_back(std::move(stack));
    }

    return CallNode::make(call->op, new_args, call->attrs, {});
  }

  void UpdateGroupOutput(const Expr& data,
                         const Group& branches,
                         size_t depth,
                         ExprSubstMap* subst_map) {
    int index = 0;
    auto split = MakeSplit(data, Integer(branches.size()), 0);
    for (const auto& branch : branches) {
      auto split_data = TupleGetItemNode::make(split, index++);
      auto squeezed_data = MakeSqueeze(split_data, {0});
      subst_map->insert({GetRef<Expr>(branch[depth]), squeezed_data});
    }
  }

private:
  std::string batched_op_name_;
};

/*! \brief Combine parallel dense if number of branches >= min_num_branches */
Expr CombineParallelDense(const Expr& expr, uint64_t min_num_branches) {
  return ParallelDenseCombiner(min_num_branches).Combine(expr);
}

namespace transform {

Pass CombineParallelDense(uint64_t min_num_branches) {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(CombineParallelDense(f, min_num_branches));
  };
  return CreateFunctionPass(pass_func, 4, "CombineParallelDense",
                            {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.CombineParallelDense")
.set_body_typed(CombineParallelDense);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
