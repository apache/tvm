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
 *  Copyright (c) 2018 by Contributors
 * \file quantize_rewrite.cc
 * \brief Lower quantized ops to exisiting Relay ops.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/qnn.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/quantize_util.h>
#include <tvm/relay/transform.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {

Expr QuantizeForwardRewrite(const Call& ref_call, const Array<Expr>& new_args, const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  Expr data = new_args[0];
  const auto* attrs = ref_call->attrs.as<QuantizeAttrs>();
  const auto out_dtype = attrs->out_dtype;
  const auto* new_tensor = data.operator->()->checked_type().as<TensorTypeNode>();
  CHECK(new_tensor) << "Expected TensorTypeNode but was " << data.operator->()->checked_type();
  const auto output_zero_point = MakeConstantScalar(Int(32), attrs->output_zero_point);
  const auto scale = MakeConstantScalar(Float(32), attrs->output_scale);
  const int32_t min_val = get_qmin(out_dtype);
  const int32_t max_val = get_qmax(out_dtype);
  auto scale_data = Cast(Round(Divide(data, scale)), Int(32));
  // we are trying to do - std::min(std::max(unclamped, min_val), max_val);
  auto add_zero_point = Add(scale_data, output_zero_point);
  auto clamped_output = Clip(add_zero_point, min_val, max_val);
  auto clamp_out_dtype = Cast(clamped_output, out_dtype);
  return clamp_out_dtype;
}

RELAY_REGISTER_OP("qnn.quantize")
    .set_attr<FForwardRewrite>("FQuantizeForwardRewrite", QuantizeForwardRewrite);

Expr DequantizeForwardRewrite(const Call& ref_call, const Array<Expr>& new_args,
                              const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  Expr data = new_args[0];
  const auto* attrs = ref_call->attrs.as<DequantizeAttrs>();
  const auto* new_tensor = data.operator->()->checked_type().as<TensorTypeNode>();
  CHECK(new_tensor) << "Expected TensorTypeNode but was " << data.operator->()->checked_type();
  const auto input_zero_point = MakeConstantScalar(Int(32), attrs->input_zero_point);
  const auto input_scale = MakeConstantScalar(Float(32), attrs->input_scale);
  auto shift = Subtract(Cast(data, Int(32)), input_zero_point);
  auto scale = Multiply(Cast(shift, Float(32)), input_scale);
  return scale;
}

RELAY_REGISTER_OP("qnn.dequantize")
    .set_attr<FForwardRewrite>("FQuantizeForwardRewrite", DequantizeForwardRewrite);

TVM_REGISTER_API("relay._quantize.rewrite").set_body_typed<Expr(Expr)>([](const Expr& e) {
  Expr ret = ForwardRewrite(e, "FQuantizeForwardRewrite", nullptr, nullptr);
  return ret;
});

}  // namespace relay
}  // namespace tvm
