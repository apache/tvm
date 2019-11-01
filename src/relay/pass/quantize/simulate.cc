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
 *
 * \file annotate.cc
 *
 * \brief Annotating the graph with simulated quantize operators.
 */

#include <tvm/relay/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include "./quantize.h"

namespace tvm {
namespace relay {
namespace quantize {

using namespace relay::transform;

Expr MakeSimulatedQuantize(Expr data) {
  static const Op& op = Op::Get("relay.op.annotation.simulated_quantize");
  Expr dom_scale = VarNode::make("dom_scale", Type());
  Expr clip_min = VarNode::make("clip_min", Type());
  Expr clip_max = VarNode::make("clip_max", Type());
  auto attrs = make_node<SimulatedQuantizeAttrs>();
  attrs->sign = true;
  attrs->rounding = "round";
  return CallNode::make(op, {data, dom_scale, clip_min, clip_max}, Attrs(attrs), {});

}

class Simulator : public ExprMutator {
 public:
  Expr VisitExpr_(const CallNode* call) final {
    Expr new_call = ExprMutator::VisitExpr_(call);
    call = new_call.as<CallNode>();

    Array<Expr> new_args;
    for (auto arg : call->args) {
      new_args.push_back(MakeSimulatedQuantize(arg));
    }
    return CallNode::make(call->op, new_args, call->attrs,
                          call->type_args);
  }
};

Expr Simulate(Expr e) {
  return Simulator().VisitExpr(e);
}

TVM_REGISTER_API("relay._quantize.Simulate")
.set_body_typed(Simulate);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
