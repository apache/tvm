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
 * \file calibrate.cc
 *
 * \brief Create profile graph and calibrate on dataset
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include "./quantize.h"

namespace tvm {
namespace relay {
namespace quantize {

class StatsCollector : private ExprMutator {
 public:
  StatsCollector() : simulated_quantize_op_(Op::Get("relay.op.annotation.simulated_quantize")) {}

  Expr Collect(const Expr& expr) {
    auto new_e = this->Mutate(expr);
    const FunctionNode* func = new_e.as<FunctionNode>();
    CHECK(func) << "Input shoule be Function";
    Expr new_body = TupleNode::make(std::move(profile_data_));
    return FunctionNode::make(FreeVars(new_body), new_body, NullValue<Type>(), func->type_params,
            func->attrs);
  }

 private:
  Array<Expr> profile_data_;
  const Op& simulated_quantize_op_;

  Expr VisitExpr_(const CallNode* call) {
    Expr new_e = ExprMutator::VisitExpr_(call);
    const CallNode* new_call = new_e.as<CallNode>();
    CHECK(new_call);
    if (new_call->op == simulated_quantize_op_) {
      auto attrs = new_call->attrs.as<SimulatedQuantizeAttrs>();
      // rewrite the annotation
      auto new_attrs = make_object<SimulatedQuantizeAttrs>();
      const Expr& quantize_input = new_call->args[0];  // expression being quantized
      auto placeholder = MakeConstantScalar(DataType::Float(32), 0.);  // unused argument
      Array<Expr> new_args{quantize_input, placeholder, placeholder, placeholder};
      new_attrs->kind = QAnnotateKind::kQIdentity;
      new_attrs->sign = attrs->sign;
      new_attrs->rounding = attrs->rounding;
      Expr identity_quantize = CallNode::make(new_call->op, new_args, Attrs{new_attrs}, {});

      // add non-const expressions to profile data
      if (attrs->kind != QAnnotateKind::kQWeight) {
        CHECK(!quantize_input.as<ConstantNode>());
        profile_data_.push_back(identity_quantize);
      }
      return identity_quantize;
    } else {
      return new_e;
    }
  }
};

/*
 * \brief Given an annotated graph, create a profile graph to collect profile data from the
 * calibration dataset.
 *
 * This pass collects simulated_quantize op into a tuple. Simulated_quantize ops are rewritten to
 * identity mode. The tuple is the output of the profile graph. Both input and output of this pass
 * are relay::Function.
 *
 * \param expr The simulation graph after annotation.
 * \return The profile graph.
 */
Expr CreateStatsCollector(const Expr& expr) {
  return StatsCollector().Collect(expr);
}

TVM_REGISTER_API("relay._quantize.CreateStatsCollector")
.set_body_typed(CreateStatsCollector);

}  // namespace quantize
}  // namespace relay
}  // namespace tvm
