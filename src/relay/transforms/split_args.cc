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
  explicit ArgumentSplitter(int max_function_args)
      : max_function_args_(max_function_args), concat_op_(Op::Get("concatenate")) {}

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (max_function_args_ < 0) return post;
    if (call->op == concat_op_) {
      auto op = call->args[0].as<TupleNode>();
      const auto param = call->attrs.as<ConcatenateAttrs>();
      int outputsNum = 1;
      if (const auto* tuple_type = call->checked_type().as<TupleTypeNode>()) {
        outputsNum = tuple_type->fields.size();
      }
      const int limit = max_function_args_ - outputsNum;
      int argsNum = op->fields.size();
      if (argsNum < limit) return post;
      int splitNum = argsNum / limit;
      splitNum = (argsNum % limit) ? splitNum + 1 : splitNum;

      std::vector<Expr> splitted(splitNum);
      for (int i = 0; i < splitNum; ++i) {
        int startIdx = i * limit;
        int argsCount = std::min(limit, argsNum - startIdx);
        tvm::Array<Expr> args;
        for (int j = 0; j < argsCount; ++j) {
          args.push_back(op->fields[j + startIdx]);
        }
        Tuple tuple(args);
        Expr body = MakeConcatenate(tuple, param->axis);
        splitted[i] = StopFusion(body);
      }
      tvm::Array<Expr> tupleArgs(splitted);
      Tuple tuple(tupleArgs);
      return MakeConcatenate(tuple, param->axis);
    }
    return post;
  }

 private:
  const int max_function_args_;
  const Op& concat_op_;
};

Expr SplitArgs(const Expr& expr, int max_function_args) {
  auto rewriter = ArgumentSplitter(max_function_args);
  return PostOrderRewrite(expr, &rewriter);
}

namespace transform {

Pass SplitArgs(int max_function_args) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SplitArgs(f, max_function_args));
      };
  return CreateFunctionPass(pass_func, 1, "SplitArgs", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SplitArgs").set_body_typed(SplitArgs);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
