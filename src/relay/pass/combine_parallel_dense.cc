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
#include "./pattern_util.h"
#include "./combine_parallel_op.h"

namespace tvm {
namespace relay {

class ParallelDenseCombiner : public ParallelOpCombiner {
 public:
  ParallelDenseCombiner(uint64_t min_num_branches) : ParallelOpCombiner("nn.dense", min_num_branches) {
  }

 protected:
  virtual bool IsSupportedOp(const CallNode* n) {
    const auto* attrs = n->attrs.as<DenseAttrs>();
    return !attrs->units.defined();
  }

  virtual bool AreCompatibleOps(const CallNode* a, const CallNode* b) {
    AttrsEqual eq;
    const auto* attrs_a = a->attrs.as<DenseAttrs>();
    const auto* attrs_b = b->attrs.as<DenseAttrs>();
    CHECK(attrs_a);
    CHECK(attrs_b);
    const auto* weight_a = a->args[1]->type_as<TensorTypeNode>();
    const auto* weight_b = b->args[1]->type_as<TensorTypeNode>();

    return eq(attrs_a->out_dtype, attrs_b->out_dtype) &&
           eq(weight_a->shape[0], weight_b->shape[0]) &&
           eq(weight_a->shape[1], weight_b->shape[1]) &&
           eq(attrs_a->units.defined(), attrs_b->units.defined());
  }

  virtual void CombineBranches(const Group& branches, ExprSubstMap& subst_map) {
    Call combined = MakeCombinedDense(branches);
    auto it = std::min_element(branches.begin(), branches.end(),
                               [](const Branch& branch_a,
                                  const Branch& branch_b) {
                                    return branch_a.size() < branch_b.size();
                                  });
    size_t depth = it->size();
    size_t i;
    // starting from 1 to skip the dense
    for (i = 1; i < depth; i++) {
      size_t parent_index;
      for (parent_index = 0; parent_index < branches[0][i]->args.size(); parent_index++) {
        if (branches[0][i]->args[parent_index].get() == branches[0][i - 1]) break;
      }
      CHECK_NE(parent_index, branches[0][i]->args.size());
      if (!CheckLevel(branches, i, parent_index)) break;
      combined = MakeCombinedCall(combined, branches, i, parent_index);
    }
    UpdateGroupOutput(combined, branches, i - 1, subst_map);
  }

 private:
  // Combine dense into batch matmul.
  Call MakeCombinedDense(const Group& branches) {
    static const Op& batch_matmul = Op::Get("nn.batch_matmul");
    Array<Expr> datas;
    Array<Expr> weights;
    for (const auto& branch : branches) {
      auto dense = branch[0];
      auto data = dense->args[0];
      auto weight = dense->args[1];
      datas.push_back(data);
      weights.push_back(weight);
    }

    Expr new_data = MakeStack(TupleNode::make(datas), 0);
    Expr new_weight = MakeStack(TupleNode::make(weights), 0);
    return CallNode::make(batch_matmul, {new_data, new_weight}, Attrs(), {});
  }

  bool IsArgCompatible(const CallNode* a, const CallNode* b, size_t index) {
    AttrsEqual eq;
    auto ta = a->args[index]->type_as<TensorTypeNode>();
    auto tb = b->args[index]->type_as<TensorTypeNode>();
    auto toutput_a = a->type_as<TensorTypeNode>();
    auto toutput_b = b->type_as<TensorTypeNode>();

    if (!eq(ta->dtype, tb->dtype) || ta->shape.size() != tb->shape.size())
      return false;

    for (size_t i = 0; i < ta->shape.size(); i++) {
      if (!eq(ta->shape[i], tb->shape[i]))
        return false;
    }
    return true;
  }

  // Check if ops in depth-th level can be combined
  bool CheckLevel(const Group& branches, size_t depth, size_t parent_index) {
    const CallNode* call = branches[0][depth];
    AttrsEqual attrs_equal;
    // check if all branches in current depth can be combined
    for (auto it = branches.begin() + 1; it != branches.end(); it++) {
      const Branch& branch = *it;
      if (!branch[depth]->op.same_as(call->op) ||
          !attrs_equal(branch[depth]->attrs, call->attrs) ||
          branch[depth]->args.size() != call->args.size()) {
        return false;
      }

      if (branch[depth]->args[parent_index].get() != branch[depth - 1])
        return false;

      // Check args
      for (size_t i = 0; i < call->args.size(); i++) {
        if (i == parent_index) continue;

        if (!IsArgCompatible(call, branch[depth], i) ||
            !attrs_equal(call->attrs, branch[depth]->attrs)) {
          return false;
        }
      }
    }
    return true;
  }

  // Combine args and make the combined CallNode
  Call MakeCombinedCall(const Expr& data, const Group& branches, size_t depth, size_t parent_index) {
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

  // Replace output of each branch with slices of the combined output
  void UpdateGroupOutput(const Expr& data, const Group& branches, size_t depth, ExprSubstMap& subst_map) {
    int index = 0;
    auto split = MakeSplit(data, Integer(branches.size()), 0);
    for (const auto& branch : branches) {
      const CallNode* dense = branch[0];
      auto split_data = TupleGetItemNode::make(split, index++);
      auto squeezed_data = MakeSqueeze(split_data, {0});
      subst_map[GetRef<Expr>(branch[depth])] = squeezed_data;
    }
  }
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
