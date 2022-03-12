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
  // Expected Types: data, scale, zero_point
  ICHECK_EQ(types.size(), 4);
  const auto* x = types[0].as<TensorTypeNode>();
  if (x == nullptr) return false;
  ICHECK(x->dtype == DataType::Int(8) || x->dtype == DataType::UInt(8))
      << "Expected quantized leaky_relu type(int8, uint8) for input but was " << x->dtype;
  const auto* param = attrs.as<LeakyReluAttrs>();
  ICHECK(param != nullptr) << "LeakyReluAttrs cannot be nullptr.";

  // Check the types of scale and zero points.
  for (size_t i = 1; i < 3; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }

  ICHECK(IsScalarType(types[1], DataType::Float(32)));  // scale
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // zero_point

  // Assign types for scale and zero points.
  reporter->Assign(types[1], TensorType({}, DataType::Float(32)));  // scale
  reporter->Assign(types[2], TensorType({}, DataType::Int(32)));    // zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // IdentityRel infer type function.
  Array<Type> tensor_types = {types[0], types[3]};
  return IdentityRel(tensor_types, 2, attrs, reporter);
}

// Positional relay function to create quantized leaky relu operator used by frontend FFI.
Expr MakeQuantizedLeakyRelu(Expr x, double alpha, Expr scale, Expr zero_point) {
  auto attrs = make_object<LeakyReluAttrs>();
  attrs->alpha = alpha;
  static const Op& op = Op::Get("qnn.leaky_relu");
  return Call(op, {x, scale, zero_point}, Attrs(attrs), {});
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
  // Let T = s(q_t - z) where q_t is the input arg[0]
  // Then, the quantized value of alpha * T is:
  // q(a * T, s, z) = [(a * T) / s] + z = a * s(q_t - z) / s + z = a * (q_t - z) + z
  // = a * q_t + (1 - a) * z
  //
  // We return the quantized value of alpha * T for all values q_t < input_zero_point.

  ICHECK_EQ(new_args.size(), 3);
  Expr quantized_data = Cast(new_args[0], DataType::Int(32));
  Expr input_zero_point = Cast(new_args[2], DataType::Int(32));

  const auto* q_attrs = attrs.as<LeakyReluAttrs>();
  auto alpha = q_attrs->alpha;

  int32_t fixed_point_multiplier, shift;
  std::tie(fixed_point_multiplier, shift) = GetFixedPointMultiplierShift(alpha);
  auto prod = FixedPointMultiply(quantized_data, fixed_point_multiplier, shift);

  int32_t fixed_point_multiplier_z, shift_z;
  std::tie(fixed_point_multiplier_z, shift_z) = GetFixedPointMultiplierShift(1 - alpha);
  auto scaled_z = FixedPointMultiply(input_zero_point, fixed_point_multiplier_z, shift_z);

  auto add = Add(prod, scaled_z);
  auto output = Where(Less(quantized_data, input_zero_point), add, quantized_data);

  const auto* input_type = arg_types[0].as<TensorTypeNode>();
  return ConvertDtype(output, input_type->dtype);
}

RELAY_REGISTER_OP("qnn.leaky_relu")
    .describe("Leaky relu for quantized tensors.")
    .set_attrs_type<LeakyReluAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "Quantized Tensor", "The input data.")
    .add_argument("scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .set_support_level(11)
    .add_type_rel("QLeakyRelu", QnnLeakyReluRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnLeakyReluCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.leaky_relu").set_body_typed(MakeQuantizedLeakyRelu);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
