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

class FastMathMutator : public ExprMutator {
 public:
  FastMathMutator()
      : exp_op_(Op::Get("exp")),
        tanh_op_(Op::Get("tanh")) {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op == exp_op_) {
      return FastExp(new_n.as<CallNode>()->args[0]);
    } else if (n->op == tanh_op_) {
      return FastTanh(new_n.as<CallNode>()->args[0]);
    }
    return new_n;
  }

 private:
  // Cache the following ops. They will be used in the passes repeatedly for
  // operator equivalence checking so that the registry lookup overhead can be
  // reduced.
  const Op& exp_op_;
  const Op& tanh_op_;
};

Expr FastMath(const Expr& e) {
  return FastMathMutator().Mutate(e);
}

namespace transform {

Pass FastMath() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(FastMath(f));
  };
  return CreateFunctionPass(pass_func, 4, "FastMath",
                            {tir::StringImmNode::make("InferType")});
}

TVM_REGISTER_GLOBAL("relay._transform.FastMath")
.set_body_typed(FastMath);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
