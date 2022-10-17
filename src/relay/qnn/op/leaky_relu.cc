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
 * \file src/relay/qnn/op/leaky_relu.cc
 * \brief QNN leaky relu operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>

#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

bool QnnLeakyReluRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  // Expected Types: data, input_scale, input_zero_point, output_scale, output_zero_point, out_type
  ICHECK_EQ(types.size(), 6);
  const auto* x = types[0].as<TensorTypeNode>();
  if (x == nullptr) return false;
  ICHECK(x->dtype == DataType::Int(8) || x->dtype == DataType::UInt(8))
      << "Expected quantized leaky_relu type(int8, uint8) for input but was " << x->dtype;
  const auto* param = attrs.as<LeakyReluAttrs>();
  ICHECK(param != nullptr) << "LeakyReluAttrs cannot be nullptr.";

  // Check the types of scale and zero points.
  for (size_t i = 1; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }

  ICHECK(IsScalarType(types[1], DataType::Float(32)));  // input_scale
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // input_zero_point
  ICHECK(IsScalarType(types[3], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[4], DataType::Int(32)));    // output_zero_point

  // Assign types for scale and zero points.
  reporter->Assign(types[1], TensorType({}, DataType::Float(32)));  // input_scale
  reporter->Assign(types[2], TensorType({}, DataType::Int(32)));    // input_zero_point
  reporter->Assign(types[3], TensorType({}, DataType::Float(32)));  // output_scale
  reporter->Assign(types[4], TensorType({}, DataType::Int(32)));    // output_zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // IdentityRel infer type function.
  Array<Type> tensor_types = {types[0], types[5]};
  return IdentityRel(tensor_types, 2, attrs, reporter);
}

// Positional relay function to create quantized leaky relu operator used by frontend FFI.
Expr MakeQuantizedLeakyRelu(Expr x, double alpha, Expr input_scale, Expr input_zero_point,
                            Expr output_scale, Expr output_zero_point) {
  auto attrs = make_object<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op& op = Op::Get("qnn.leaky_relu");
  return Call(op, {x, input_scale, input_zero_point, output_scale, output_zero_point}, Attrs(attrs),
              {});
}

/*
 * \brief Canonicalizes the QNN leaky relu op.
 * \param attrs The empty attribute.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for leaky relu op.
 */
Expr QnnLeakyReluCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                              const Array<tvm::relay::Type>& arg_types) {
  // We rely on fixed point arithmetic to preserve the precision of multiplication
  // by a small alpha value < 1.
  //
  // We assume the same scale and zero point for alpha and the input tensor.
  // LeakyReLU can be written in terms of respective quantized tensors, scales and
  // zero points as
  //
  //    scale_o * (Q_o - zp_o) = alpha * scale_i * (Q_i - zp_i)  when Q_i < zp_i  (1)
  //    scale_o * (Q_o - zp_o) = scale_i * (Q_i - zp_i)  when Q_i >= zp_i  (2)
  //
  // Since the input qnn params can be different than output qnn params, we first requantize the
  // input tensor to the output qnn params. After requantizing Q_i, equation (1) becames equation
  // (3) where Q_i' is the requantized data from Q_i.
  //
  //    scale_o * (Q_o - zp_o) = alpha * scale_o * (Q_i' - zp_o)  when Q_i < zp_i  (3)
  //                       Q_o = alpha * Q_i' + (1 - alpha) * zp_o  when Q_i < zp_i  (4)
  //
  // It is equal to requantize Q_i to Q_o using scale_o and zp_o in equation (2).
  // So equation (2) becomes
  //
  //                       Q_o = requantize(Q_i)  when Q_i >= zp_i  (5)
  //
  // Finnally, Q_o could be calculated by equation (4) and equation (5).
  ICHECK_EQ(new_args.size(), 5);
  Expr data = Cast(new_args[0], DataType::Int(32));
  Expr input_scale = new_args[1];
  Expr input_zero_point = Cast(new_args[2], DataType::Int(32));
  Expr output_scale = new_args[3];
  Expr output_zero_point = Cast(new_args[4], DataType::Int(32));

  const auto* q_attrs = attrs.as<LeakyReluAttrs>();
  auto alpha = q_attrs->alpha;

  const auto input_shape = get_shape(arg_types[0]);
  const auto input_dtype = arg_types[0].as<TensorTypeNode>()->dtype;

  // requantize the input to Q_i'
  auto requantized_expr = RequantizeOrUpcast(data, input_scale, input_zero_point, output_scale,
                                             output_zero_point, input_shape);

  // alpha * Q_i'
  auto [fixed_point_multiplier, shift] = GetFixedPointMultiplierShift(alpha);
  auto prod = FixedPointMultiply(requantized_expr, fixed_point_multiplier, shift);

  // (1 - alpha) * zp_o
  auto [fixed_point_multiplier_z, shift_z] = GetFixedPointMultiplierShift(1 - alpha);
  auto scaled_z = FixedPointMultiply(output_zero_point, fixed_point_multiplier_z, shift_z);

  // alpha * Q_i' + (1 - alpha) * zp_o
  auto add = Add(prod, scaled_z);
  auto output = Where(Less(data, input_zero_point), add, requantized_expr);

  return ConvertDtype(output, input_dtype);
}

RELAY_REGISTER_OP("qnn.leaky_relu")
    .describe("Leaky relu for quantized tensors.")
    .set_attrs_type<LeakyReluAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Quantized Tensor", "The input data.")
    .add_argument("input_scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("input_zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("QLeakyRelu", QnnLeakyReluRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnLeakyReluCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.leaky_relu").set_body_typed(MakeQuantizedLeakyRelu);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
