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
 * \file src/relay/qnn/op/mul.cc
 * \brief QNN mul operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../../pass/pattern_util.h"
#include "../util.h"
#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

/*
 * \brief Canonicalizes the QNN mul op.
 * \param attrs The QNN concatenate attrs.
 * \param new_args The new mutated args to the call node.
 * \param arg_types The types of input and output.
 * \return The sequence of Relay ops for mul op.
 */
Expr QnnMulCanonicalize(const Attrs& attrs, const Array<Expr>& new_args,
                        const Array<tvm::relay::Type>& arg_types) {
  // Get the attrs.
  CHECK_EQ(new_args.size(), 8);
  auto& lhs = new_args[0];
  auto& rhs = new_args[1];
  auto& lhs_scale = new_args[2];
  auto& lhs_zero_point = new_args[3];
  auto& rhs_scale = new_args[4];
  auto& rhs_zero_point = new_args[5];
  auto& output_scale = new_args[6];
  auto& output_zero_point = new_args[7];

  // Get the input dtype and shape.
  CHECK_EQ(arg_types.size(), 9);
  auto tensor_type = arg_types[0].as<TensorTypeNode>();
  CHECK(tensor_type != nullptr);
  auto input_dtype = tensor_type->dtype;
  auto input_shape = tensor_type->shape;

  /*
  A tensor multiplication c = a * b can be written in terms of respective
  quantized tensors, scales and zero points as
  S_c * (Q_c - zp_c) = S_a * (Q_a - zp_a) * S_b * (Q_b - zp_b).

  We can consider the product (Q_a - zp_a) * (Q_b - zp_b) as a different
  quantized tensor of c, Q', with corresponding scale S' = S_a * S_b and zp' =
  0. The quantized multiplication then becomes
  Q_c = S'/S_c Q' + z_c,
  which is essentially a requantization of tensor Q' into tensor Q_c.
  */

  auto lhs_shifted = Cast(lhs, DataType::Int(32));
  auto rhs_shifted = Cast(rhs, DataType::Int(32));

  auto zero_scalar = MakeConstantScalar(DataType::Int(32), 0);
  if (!IsEqualScalar(lhs_zero_point, zero_scalar)) {
    lhs_shifted = Subtract(lhs_shifted, lhs_zero_point);
  }

  if (!IsEqualScalar(rhs_zero_point, zero_scalar)) {
    rhs_shifted = Subtract(rhs_shifted, rhs_zero_point);
  }

  // Create a new tensor Q'
  auto output = Multiply(lhs_shifted, rhs_shifted);

  // Get the adjusted new scale and zero points.
  float lhs_scale_float = GetScalarFromConstant<float>(lhs_scale);
  float rhs_scale_float = GetScalarFromConstant<float>(rhs_scale);
  float new_scale_float = lhs_scale_float * rhs_scale_float;
  auto new_input_scale = MakeConstantScalar(DataType::Float(32), new_scale_float);
  auto new_input_zero_point = zero_scalar;

  // Requantize to get Q_c
  output = Requantize(output, input_shape, new_input_scale, new_input_zero_point, output_scale,
                      output_zero_point, input_dtype);

  return output;
}

// QNN Multiplication operator.
QNN_REGISTER_BINARY_OP("mul")
.describe("Elementwise mul with with broadcasting for quantized tensors.")
.set_support_level(11)
.set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnMulCanonicalize);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
