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
 * \file canonicalize_ops.cc
 * \brief Canonicalize special operators to basic operators.
    This can simplify latter analysis. (e.g. Expand bias_add to expand_dims and broadcast_add.)
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class BiasAddSimplifier : public ExprRewriter {
 public:
  BiasAddSimplifier() : bias_add_op_(Op::Get("nn.bias_add")) {}

  Expr Rewrite_(const CallNode* n, const Expr& post) override {
    auto new_n = post;
    if (n->op == bias_add_op_) {
      Call call = Downcast<Call>(new_n);
      CHECK_EQ(call->args.size(), 2);
      const BiasAddAttrs* param = call->attrs.as<BiasAddAttrs>();

      auto ttype = n->args[0]->type_as<TensorTypeNode>();
      size_t n_dim = ttype->shape.size();
      int axis = param->axis;
      if (axis < 0) {
        axis += n_dim;
      }
      Expr expanded_bias = ExpandBiasToMatchAxis(call->args[1], n_dim, {axis});
      Expr ret = Add(call->args[0], expanded_bias);
      ret->checked_type_ = n->checked_type_;
      return ret;
    }
    return new_n;
  }

 private:
  // Cache the bias_add for equivalence checking.
  const Op& bias_add_op_;
};

Expr CanonicalizeOps(const Expr& e) {
  auto rewriter = BiasAddSimplifier();
  return PostOrderRewrite(e, &rewriter);
}

namespace transform {

Pass CanonicalizeOps() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
    return Downcast<Function>(CanonicalizeOps(f));
  };
  return CreateFunctionPass(pass_func, 3, "CanonicalizeOps", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.CanonicalizeOps")
.set_body_typed(CanonicalizeOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
