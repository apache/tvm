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
 * \file src/relay/qnn/op/unary_elementwise_op.cc
 * \brief QNN unary elementwise operators.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>

#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

bool QnnElementwiseUnaryFuncRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                                const TypeReporter& reporter) {
  // Expected Types: data, scale, zero_point, output_scale, output_zero_point
  ICHECK_EQ(types.size(), 6);
  const auto* x = types[0].as<TensorTypeNode>();
  if (x == nullptr) return false;
  ICHECK(x->dtype == DataType::Int(8) || x->dtype == DataType::UInt(8))
      << "Expected quantized type(int8, uint8) for input but was " << x->dtype;

  // Check the types of scale and zero points.
  for (size_t i = 1; i < 5; ++i) {
    if (types[i].as<IncompleteTypeNode>()) {
      return false;
    }
  }
  ICHECK(IsScalarType(types[1], DataType::Float(32)));  // scale
  ICHECK(IsScalarType(types[2], DataType::Int(32)));    // zero_point
  ICHECK(IsScalarType(types[3], DataType::Float(32)));  // output_scale
  ICHECK(IsScalarType(types[4], DataType::Int(32)));    // output_zero_point

  // Assign types for scale and zero points.
  reporter->Assign(types[1], TensorType({}, DataType::Float(32)));  // scale
  reporter->Assign(types[2], TensorType({}, DataType::Int(32)));    // zero_point
  reporter->Assign(types[3], TensorType({}, DataType::Float(32)));  // output_scale
  reporter->Assign(types[4], TensorType({}, DataType::Int(32)));    // output_zero_point

  // Collect the input tensor and output tensor devoid of scale and zero points to reuse Relay
  // IdentityRel infer type function.
  Array<Type> tensor_types = {types[0], types[5]};
  return IdentityRel(tensor_types, 2, attrs, reporter);
}

Expr MakeQuantizedElmentwiseUnaryFunc(Expr x, Expr scale, Expr zero_point, Expr output_scale,
                                      Expr output_zero_point, const Op& op) {
  return Call(op, {x, scale, zero_point, output_scale, output_zero_point}, Attrs(), {});
}

/*! TODO
 *
 */
#define QNN_CREATE_UNARY_ELEMENTWISE_OP(OpName)                                                 \
  RELAY_REGISTER_OP("qnn." OpName)                                                              \
      .describe("Elementwise " OpName " for quantized tensors.")                                \
      .set_num_inputs(5)                                                                        \
      .add_argument("data", "Quantized Tensor", "The input data.")                              \
      .add_argument("scale", "Tensor", "The quantization scale of the input tensor.")           \
      .add_argument("zero_point", "Tensor", "The quantization zero_point of the input tensor.") \
      .add_argument("output_scale", "Tensor", "The quantization scale of the output tensor.")   \
      .add_argument("output_zero_point", "Tensor",                                              \
                    "The quantization zero_point of the output tensor.")                        \
      .set_support_level(11)                                                                    \
      .add_type_rel("qnn." OpName, QnnElementwiseUnaryFuncRel)                                  \
      .set_attr<TNonComputational>("TNonComputational", true);                                  \
                                                                                                \
  TVM_REGISTER_GLOBAL("relay.qnn.op._make." OpName)                                             \
      .set_body_typed(                                                                          \
          [](Expr x, Expr scale, Expr zero_point, Expr output_scale, Expr output_zero_point) {  \
            return MakeQuantizedElmentwiseUnaryFunc(x, scale, zero_point, output_scale,         \
                                                    output_zero_point, Op::Get("qnn." OpName)); \
          });

QNN_CREATE_UNARY_ELEMENTWISE_OP("rsqrt");
QNN_CREATE_UNARY_ELEMENTWISE_OP("erf");
QNN_CREATE_UNARY_ELEMENTWISE_OP("tanh");
QNN_CREATE_UNARY_ELEMENTWISE_OP("exp");
QNN_CREATE_UNARY_ELEMENTWISE_OP("sqrt");

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
