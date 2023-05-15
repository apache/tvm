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
 * \file src/relay/qnn/op/softmax.cc
 * \brief QNN softmax operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>

#include "op_common.h"
#include "tvm/ir/expr.h"
#include "tvm/relay/attrs/nn.h"
#include "tvm/relay/type.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/logging.h"
#include "tvm/topi/reduction.h"

namespace tvm {
namespace relay {
namespace qnn {

bool QnnSoftmaxRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  // Expected Types: input, scale, zero_point, output_scale, output_zero_point, output
  ICHECK_EQ(types.size(), 6);
  const auto* x = types[0].as<TensorTypeNode>();
  if (x == nullptr) return false;
  ICHECK(x->dtype == DataType::Int(8))
      << "Expected quantized softmax type(int8) for input but was " << x->dtype;

  // Check the types of scale and zero points.
  for (size_t i = 1; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }

  ICHECK(IsScalarType(types[1], DataType::Float(32)));  // scale
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // zero_point
  ICHECK(IsScalarType(types[3], DataType::Float(32)));  // scale
  ICHECK(IsScalarType(types[4], DataType::Int(32)));    // zero_point

  // Assign types for scale and zero points.
  reporter->Assign(types[1], TensorType({}, DataType::Float(32)));  // scale
  reporter->Assign(types[2], TensorType({}, DataType::Int(32)));    // zero_point
  reporter->Assign(types[3], TensorType({}, DataType::Float(32)));  // scale
  reporter->Assign(types[4], TensorType({}, DataType::Int(32)));    // zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // IdentityRel infer type function.
  Array<Type> tensor_types = {types[0], types[5]};
  return IdentityRel(tensor_types, 2, attrs, reporter);
}

// Positional relay function to create quantized softmax operator used by frontend FFI.
Expr MakeQuantizedSoftmax(Expr x, int axis, Expr scale, Expr zero_point, Expr output_scale,
                          Expr output_zero_point) {
  auto attrs = make_object<SoftmaxAttrs>();
  attrs->axis = axis;
  static const Op& op = Op::Get("qnn.softmax");
  return Call(op, {x, scale, zero_point, output_scale, output_zero_point}, Attrs(attrs), {});
}

/*
 * \brief Canonicalizes the QNN softmax op.
 * \param attrs The Softmax attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for softmax op.
 * \note This op is highly experimental and sometimes lacks accuracy.
 *       Be aware that the input scale must be in the range of 0 to 1.
 */
Expr QnnSoftmaxCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                            const Array<tvm::relay::Type>& arg_types) {
  // Expected: input, scale, zero_point, output_scale, output_zero_point
  ICHECK_EQ(new_args.size(), 5);

  const auto const_i32 = [&](int32_t val) { return MakeConstantScalar(DataType::Int(32), val); };
  const auto const_f32 = [&](float val) { return MakeConstantScalar(DataType::Float(32), val); };

  const auto const_input_scale = new_args[1].as<ConstantNode>();
  ICHECK(const_input_scale) << "Input scale should be constant.";
  ICHECK(const_input_scale->is_scalar()) << "Input scale should be scalar.";
  const float input_scale = static_cast<float*>(const_input_scale->data->data)[0];
  ICHECK(input_scale <= 1.f) << "Input scale should be less than or equal to 1.";

  const Expr input_zero_point = new_args[2];
  const Expr output_scale = new_args[3];
  const Expr output_zero_point = new_args[4];
  const int axis = attrs.as<SoftmaxAttrs>()->axis;

  // Refer to the Algorithm 1 in https://arxiv.org/pdf/2207.01405.pdf

  const Expr quantized_data = Subtract(Cast(new_args[0], DataType::Int(32)), input_zero_point);

  const Expr x_0 = ConvertDtype(const_f32(std::round(1.f / input_scale)), DataType::Int(32));
  const Expr max = Max(quantized_data, {axis}, true, false);
  const Expr x = Subtract(quantized_data, max);

  const int m = 30;
  const int bits = 8;
  const Expr x_p = Subtract(Add(x, RightShift(x, const_i32(1))), RightShift(x, const_i32(4)));
  const Expr q = Clip(Divide(x_p, Negative(x_0)), 0, 20);
  const Expr max_q = Max(q, {axis}, true, false);
  const Expr r = Subtract(x_p, Multiply(q, Negative(x_0)));
  const Expr x_b = Add(RightShift(r, const_i32(1)), x_0);
  const Expr exps = LeftShift(x_b, Subtract(max_q, q));
  const Expr sums = Sum(exps, {axis}, true, false);
  const Expr output =
      RightShift(Multiply(Divide(const_i32(1 << m), sums), exps), const_i32(m - (bits - 1)));
  const Expr requantized = Requantize(output, arg_types[0].as<TensorTypeNode>()->shape,
                                      const_f32(1.f / (1 << (bits - 1))), const_i32(0),
                                      output_scale, output_zero_point, DataType::Int(bits), 0);

  return requantized;
}

RELAY_REGISTER_OP("qnn.softmax")
    .describe("Softmax for quantized tensors.")
    .set_attrs_type<SoftmaxAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Quantized Tensor", "The input data.")
    .add_argument("scale", "Tensor", "The quantization scale of the input tensor.")
    .add_argument("zero_point", "Tensor", "The quantization zero_point of the input tensor.")
    .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")
    .add_argument("output_zero_point", "Tensor",
                  "The quantization zero_point of the output tensor.")
    .set_support_level(11)
    .add_type_rel("QSoftmax", QnnSoftmaxRel)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnSoftmaxCanonicalize);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.softmax").set_body_typed(MakeQuantizedSoftmax);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
