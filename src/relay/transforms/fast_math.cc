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
 * \file fast_math.cc
 * \brief Replaces non linear activation functions with their fast but approximate counterparts.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class FastMathMutator : public ExprRewriter {
 public:
  FastMathMutator()
      : exp_op_(Op::Get("exp")),
        erf_op_(Op::Get("erf")),
        tanh_op_(Op::Get("tanh")) {}

  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (pre->op == exp_op_) {
      return FastExp(post.as<CallNode>()->args[0]);
    } else if (pre->op == erf_op_) {
      return FastErf(post.as<CallNode>()->args[0]);
    } else if (pre->op == tanh_op_) {
      return FastTanh(post.as<CallNode>()->args[0]);
    }
    return post;
  }

 private:
  // Cache the following ops. They will be used in the passes repeatedly for
  // operator equivalence checking so that the registry lookup overhead can be
  // reduced.
  const Op& exp_op_;
  const Op& erf_op_;
  const Op& tanh_op_;
};

Expr FastMath(const Expr& e) {
  auto rewriter = FastMathMutator();
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass FastMath() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(FastMath(f));
  };
  return CreateFunctionPass(pass_func, 4, "FastMath", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.FastMath")
.set_body_typed(FastMath);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
