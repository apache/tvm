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
 * Copyright (c) 2018 by Contributors
 * \file canonicalize_ops.cc
 * \brief Canonicalize special operators to basic operators.
    This can simplify latter analysis. (e.g. Expand bias_add to expand_dims and broadcast_add.)
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/transform.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

class BiasAddSimplifier : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* n) {
    static const Op& bias_add = Op::Get("nn.bias_add");
    auto new_n = ExprMutator::VisitExpr_(n);
    if (n->op.same_as(bias_add)) {
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
};

Expr CanonicalizeOps(const Expr& e) {
  return BiasAddSimplifier().Mutate(e);
}

namespace transform {

Pass CanonicalizeOps() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
    return Downcast<Function>(CanonicalizeOps(f));
  };
  return CreateFunctionPass(pass_func, 3, "CanonicalizeOps",
                            {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay._transform.CanonicalizeOps")
.set_body_typed(CanonicalizeOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
