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
 * \file split_args.cc
 */
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include "../op/annotation/annotation.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

class ArgumentSplitter : public ExprRewriter {
 public:
  explicit ArgumentSplitter(size_t max_function_args)
      : max_function_args_(max_function_args), concat_op_(Op::Get("concatenate")) {}

  Expr ConcatSplitter(const TupleNode* tuple_node, const tvm::Array<relay::Expr>& args, int axis,
                      size_t limit) {
    tvm::Array<relay::Expr> new_args;
    size_t added_args = 0;
    for (const auto& it : args) {
      size_t curr_args = 1;
      if (const auto* ttype = it->checked_type().as<TensorTypeNode>()) {
        ICHECK(additional_args_cache_.count(ttype));
        curr_args += additional_args_cache_[ttype];
      }
      if (added_args + curr_args > limit) {
        Tuple new_tuple = WithFields(GetRef<Tuple>(tuple_node), new_args);
        Expr stop = StopFusion(new_tuple);
        Expr lastExpr = MakeConcatenate(stop, axis);
        new_args.clear();
        new_args.push_back(lastExpr);
        added_args = curr_args;
      }
      added_args += curr_args;
      new_args.push_back(it);
    }
    Tuple new_tuple = WithFields(GetRef<Tuple>(tuple_node), new_args);
    Expr stop = StopFusion(new_tuple);
    Expr lastExpr = MakeConcatenate(stop, axis);
    return lastExpr;
  }

  // In the case of dynamic shape in tensor, the sizes of any_dims and strides are passed as
  // function args
  size_t CalculateNumberOfAdditionalArgs_(const TensorTypeNode* arg, bool isOutput = false) {
    size_t num = 0;
    for (const auto& dim : arg->shape) {
      if (dim.as<AnyNode>()) {
        num++;
      }
    }
    // In the case of dynamic shape, strides are also passed to a function as arguments. The number
    // of strides equals the rank of the tensor.
    if (num > 0 && isOutput)
      return arg->shape.size();
    else if (num > 0)
      num += arg->shape.size();
    return num;
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (max_function_args_ == 0) return post;
    if (call->op == concat_op_) {
      auto tuple_node = call->args[0].as<TupleNode>();
      if (tuple_node == nullptr) return post;
      const auto param = call->attrs.as<ConcatenateAttrs>();
      size_t outputsNum = 1;
      if (const auto* tuple_type = call->checked_type().as<TupleTypeNode>()) {
        outputsNum = tuple_type->fields.size();
        for (const auto& it : tuple_type->fields) {
          if (const auto* ttype = it.as<TensorTypeNode>()) {
            outputsNum += CalculateNumberOfAdditionalArgs_(ttype, true);
          }
        }
      } else if (const auto* ttype = call->checked_type().as<TensorTypeNode>()) {
        outputsNum += CalculateNumberOfAdditionalArgs_(ttype, true);
      }
      CHECK_GT(max_function_args_, outputsNum);
      size_t limit = max_function_args_ - outputsNum;

      size_t argsNum = tuple_node->fields.size();
      for (const auto& it : tuple_node->fields) {
        if (const auto* ttype = it->checked_type().as<TensorTypeNode>()) {
          size_t any_dims = CalculateNumberOfAdditionalArgs_(ttype);
          argsNum += any_dims;
          additional_args_cache_[ttype] = any_dims;
        }
      }
      if (argsNum < limit) return post;
      return ConcatSplitter(tuple_node, tuple_node->fields, param->axis, limit);
    }
    return post;
  }

 private:
  const size_t max_function_args_;
  const Op& concat_op_;
  std::unordered_map<const TensorTypeNode*, size_t> additional_args_cache_;
};

Expr SplitArgs(const Expr& expr, size_t max_function_args) {
  auto rewriter = ArgumentSplitter(max_function_args);
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass SplitArgs(uint64_t max_function_args) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        auto r = Downcast<Function>(SplitArgs(f, max_function_args));
        return m->attrs.defined() ? WithAttrs(r, {m->attrs->dict}) : r;
      };
  return CreateFunctionPass(pass_func, 1, "SplitArgs", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SplitArgs").set_body_typed(SplitArgs);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
