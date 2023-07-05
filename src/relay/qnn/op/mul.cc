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

#include "../../transforms/pattern_utils.h"
#include "../utils.h"
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
  Expr output;

  // Get the attrs.
  QnnBinaryOpArguments args(new_args);

  // Get the input dtype and shape.
  QnnBinaryOpTensorType input_type(arg_types, 0);
  // data types
  const auto int32_dtype = DataType::Int(32);
  const auto float32_dtype = DataType::Float(32);

  const auto* broadcast_attrs = attrs.as<BroadcastAttrs>();
  ICHECK(broadcast_attrs != nullptr);

  auto lhs_axis = broadcast_attrs->lhs_axis;
  auto rhs_axis = broadcast_attrs->rhs_axis;

  if (IsConstScalar(args.lhs_scale) && IsConstScalar(args.rhs_scale)) {
    /*
    This is per-tensor quantized multiply.

    A tensor multiplication c = a * b can be written in terms of respective
    quantized tensors, scales and zero points as
    S_c * (Q_c - zp_c) = S_a * (Q_a - zp_a) * S_b * (Q_b - zp_b).

    We can consider the product (Q_a - zp_a) * (Q_b - zp_b) as a different
    quantized tensor of c, Q', with corresponding scale S' = S_a * S_b and zp' =
    0. The quantized multiplication then becomes
    Q_c = S'/S_c Q' + z_c,
    which is essentially a requantization of tensor Q' into tensor Q_c.
    */

    auto lhs_shifted = Cast(args.lhs, int32_dtype);
    auto rhs_shifted = Cast(args.rhs, int32_dtype);

    auto zero_scalar = MakeConstantScalar(int32_dtype, 0);
    if (!IsEqualScalar(args.lhs_zero_point, zero_scalar)) {
      lhs_shifted = Subtract(lhs_shifted, args.lhs_zero_point);
    }

    if (!IsEqualScalar(args.rhs_zero_point, zero_scalar)) {
      rhs_shifted = Subtract(rhs_shifted, args.rhs_zero_point);
    }

    // Create a new tensor Q'
    output = Multiply(lhs_shifted, rhs_shifted);

    // Get the adjusted new scale and zero points.
    float lhs_scale_float = GetScalarFromConstant<float>(args.lhs_scale);
    float rhs_scale_float = GetScalarFromConstant<float>(args.rhs_scale);
    float new_scale_float = lhs_scale_float * rhs_scale_float;
    auto new_input_scale = MakeConstantScalar(float32_dtype, new_scale_float);
    auto new_input_zero_point = zero_scalar;

    // Requantize to get Q_c
    output = Requantize(output, input_type.shape, new_input_scale, new_input_zero_point,
                        args.output_scale, args.output_zero_point, input_type.dtype);
  } else if (lhs_axis == rhs_axis) {
    /*
    This is per-channel quantized multiply, assumming lhs_axis and rhs_axis are the same.
    The subtract is done on the specified axis via broadcast. Then, we multiply lhs and rhs.
    The output is requantized using new scale and axis. TODO: support different axes.
    */

    auto lhs_data = Cast(args.lhs, int32_dtype);
    auto rhs_data = Cast(args.rhs, int32_dtype);

    auto zero_scalar = MakeConstantScalar(int32_dtype, 0);
    if (!IsEqualScalar(args.lhs_zero_point, zero_scalar)) {
      // Broadcast lhs zero point if needed
      int rank = static_cast<int>(input_type.shape.size());
      int axis = (lhs_axis < 0) ? ((rank > 0) ? rank + lhs_axis : 0) : lhs_axis;
      Expr lhs_zero_broadcast = ExpandBiasToMatchAxis(Reshape(args.lhs_zero_point,
                                                              {
                                                                  -1,
                                                              }),
                                                      rank, {axis});
      lhs_data = Subtract(lhs_data, Cast(lhs_zero_broadcast, DataType::Int(32)));
    }

    if (!IsEqualScalar(args.rhs_zero_point, zero_scalar)) {
      // Broadcast rhs zero point if needed
      int rank = static_cast<int>(input_type.shape.size());
      int axis = (rhs_axis < 0) ? ((rank > 0) ? rank + rhs_axis : 0) : rhs_axis;
      Expr rhs_zero_broadcast = ExpandBiasToMatchAxis(Reshape(args.rhs_zero_point,
                                                              {
                                                                  -1,
                                                              }),
                                                      rank, {axis});
      rhs_data = Subtract(rhs_data, Cast(rhs_zero_broadcast, DataType::Int(32)));
    }

    // Create a new tensor Q'
    output = Multiply(lhs_data, rhs_data);

    // Requantize to get Q_c
    auto lhs_scales = GetFloatVectorFromConstant(args.lhs_scale);
    auto rhs_scales = GetFloatVectorFromConstant(args.rhs_scale);
    std::vector<double> output_multipliers;
    for (size_t i = 0; i < lhs_scales.size(); i++) {
      double multiplier = static_cast<double>(lhs_scales[i]) * static_cast<double>(rhs_scales[i]);
      output_multipliers.push_back(multiplier);
    }
    auto new_input_scale = MakeConstantTensor(
        DataType::Float(32), {(int64_t)output_multipliers.size()}, output_multipliers);

    output = Requantize(output, input_type.shape, new_input_scale, zero_scalar, args.output_scale,
                        args.output_zero_point, input_type.dtype, lhs_axis);

  } else {
    LOG(FATAL) << "Not supported: lhs_axis and rhs_axis are not the same.";
  }

  return output;
}

// QNN Multiplication operator.
QNN_REGISTER_BINARY_OP("mul")
    .describe("Elementwise mul with broadcasting for quantized tensors.")
    .set_support_level(11)
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize", QnnMulCanonicalize)
    .set_attr<TOpPattern>("TOpPattern", kBroadcast);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
