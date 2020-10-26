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
 * \file quantize.cc
 *
 * \brief transform a graph to a low-bit graph
 *   for compression and acceleration.
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/data_type.h>
#include <tvm/relay/type.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <stack>
#include "./quantize.h"

namespace tvm {
namespace hago {

using namespace ::tvm::relay;
using ::tvm::relay::Expr;
using ::tvm::relay::Type;

TVM_REGISTER_NODE_TYPE(SimulatedQuantizeAttrs);

bool SimulatedQuantizeRel(const Array<Type>& types,
                          int num_inputs,
                          const Attrs& attrs,
                          const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 6);
  const auto param = attrs.as<SimulatedQuantizeAttrs>();
  CHECK(param != nullptr);

  const auto* data = types[0].as<TensorTypeNode>();
  CHECK(data != nullptr);
  // CHECK_NE(data->shape.size(), 0) << "Input shape cannot be empty";

  // FIXME - Use axis to do type checking for scale
  // Skip for supporting per-channel scales
  // reporter->Assign(types[1], TensorType({}, DataType::Float(32)));     // in_scale
  // reporter->Assign(types[2], TensorType({}, DataType::Float(32)));     // out_scale
  reporter->Assign(types[3], TensorType({}, DataType::Float(32)));     // clip_min
  reporter->Assign(types[4], TensorType({}, DataType::Float(32)));     // clip_max
  reporter->Assign(types[5], types[0]);                                // output
  return true;
}


RELAY_REGISTER_OP("nn.simulated_quantize")
.describe(R"code(simulated quantize op)code" TVM_ADD_FILELINE)
.set_num_inputs(5)
.add_argument("data", "Tensor", "The input data.")
.add_argument("in_scale", "Scalar", "The scale of input.")
.add_argument("out_scale", "Scalar", "The scale of output.")
.add_argument("clip_min", "Scalar", "The clip min.")
.add_argument("clip_max", "Scalar", "The clip max.")
.set_attrs_type<SimulatedQuantizeAttrs>()
.set_support_level(11)
.add_type_rel("simulated_quantize", SimulatedQuantizeRel);


Expr create_simulated_quantize(Expr data,
                               Expr in_scale, Expr out_scale,
                               Expr clip_min, Expr clip_max,
                               DataType in_dtype, DataType out_dtype,
                               bool sign, std::string rounding,
                               Optional<Integer> axis) {
  auto attrs = make_object<SimulatedQuantizeAttrs>();
  attrs->in_dtype = in_dtype;
  attrs->out_dtype = out_dtype;
  attrs->sign = sign;
  attrs->rounding = rounding;
  attrs->axis = axis;
  static const Op& op = Op::Get("nn.simulated_quantize");
  return relay::Call(op, {data, in_scale, out_scale, clip_min, clip_max}, Attrs(attrs));
}

TVM_REGISTER_GLOBAL("hago.quantize.simulated_quantize").set_body_typed(create_simulated_quantize);

}  // namespace hago
}  // namespace tvm
