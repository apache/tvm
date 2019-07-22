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
 *  Copyright (c) 2019 by Contributors
 * \file qnn_lower.cc
 * \brief Lower qnn ops to a sequence of exisiting Relay ops.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include "../util.h"
#include "../../pass/pattern_util.h"

namespace tvm {
namespace relay {
namespace qnn {
/*!
 * \brief namespace of qnn lower pass.
 *
 * Use namespace to reduce potential naming conflict.
 */
namespace qnn_lower {

using runtime::TypedPackedFunc;

// Lowering of qnn.requantize op

/*
 * \brief Convert FP32 representation into fixed point representation.
 * \param double_multplier The input FP32 number.
 * \param idtype The input datatype.
 * \return The pair of multiplier and shift for fixed point representation.
 * \note Converts a floating point number so that it can be represented by
 *       integers. The representation is
 *             float_number = (significand) * 2^(exponent)
 *
 *       The significand is a number between 0.5 and 1. This is represented by
 *       an integer number. For example, if it is int32, then the decimal point
 *       exists between bit 31 and 30 from LSB (or between first and second bit
 *       from the left).
 *
 *       Some examples are
 *           0.25 = (0.5) * 2^(-1)
 *           0.125 = (0.5) * 2^(-2)
 *
 *       Credit to TFLite reference implementation.
 */
std::pair<int, int> GetFixedPointMultiplierShift(double double_multiplier,
    const DataType& idtype) {
  int significand, exponent;
  int idtype_bits = idtype.bits();

  // Get the significand (significand) and exponent (exponent)
  double significand_d = std::frexp(double_multiplier, &exponent);

  // Convert the double significand to int significand.
  significand_d = std::round(significand_d * (1ll << (idtype_bits - 1)));
  auto significand_int64 = static_cast<int64_t>(significand_d);
  CHECK_LE(significand_int64, (1ll << (idtype_bits - 1)));
  if (significand_int64 == (1ll << (idtype_bits - 1))) {
    significand_int64 /= 2;
    ++exponent;
  }
  CHECK_LE(significand_int64, std::numeric_limits<int>::max());
  significand = static_cast<int>(significand_int64);
  return std::pair<int, int>(significand, exponent);
}

/*
 * \brief Lower requantize to a sequence of ops.
 * \param input_tensor The input tensor to requantize op.
 * \param param The requantize op attrs.
 * \param idtype The dtype of the input tensor.
 * \param out_shape The output shape of the requantize op.
 * \return The sequence of existing Relay ops.
 * \note Requantization using only integer computation. Here, the computation is
 *       converted to a fixed point computation by computing output multiplier
 *       and shift. This is useful, if the target device does not support/have
 *       very expensive floating point computations.
 *
 *       Original compuation is scale_fp32 * quantized_tensor.  To convert into
 *       integer computation, the multiplication with fp32 scalar can be
 *       replaced by multiplication with an int value and then right shifting
 *       the result. This approximates the floating point computation with a
 *       fixed point computation.
 *
 *       The whole computation this can be broken down into following steps
 *       1) Calculate the integer multiplier and integer shift.
 *       2) Subtract the input integer point.
 *       3) Multiply the integer fixed point multiplier with quantized tensor.
 *       4) Round the result.
 *       5) Right shift the result.
 *       6) Add the output_zero_point.
 *       7) Cast to the out_dtype.
 */
Expr RequantizeLower(const Expr& input_tensor,
    const RequantizeAttrs* param, const DataType& idtype,
    const Array<IndexExpr>& out_shape) {

  double double_multiplier = param->input_scale/param->output_scale;

  // The multiplication will be performed in higher precision. Find the dtype.
  int idtype_bits = idtype.bits();
  DataType up_idtype = Int(2 * idtype_bits);

  // 1) Calculating the integer multiplier and integer shift
  std::pair<int, int> fixed_point_params =
      GetFixedPointMultiplierShift(double_multiplier, idtype);
  int fixed_point_multiplier = fixed_point_params.first;
  int shift = fixed_point_params.second;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;

  // 2) Subtract the input_zero_point
  auto tensor = Cast(input_tensor, up_idtype);
  if (param->input_zero_point != 0) {
    auto input_zp = MakeConstantScalar(up_idtype, param->input_zero_point);
    tensor = Subtract(tensor, input_zp);
  }

  // 3) Multiply the integer multiplier
  if (left_shift != 0) {
    tensor = Multiply(tensor, MakeConstantScalar(up_idtype, 1 << left_shift));
  }
  // Perform the multiplication in higher precision.
  // If idtype is Int(32), the scalar is a fixed point value of int32 where the
  // decimal point is between bits 31 and 30. After multiplying with
  // input_tensor, the result is in int64 where the decimal point is sitting
  // between bits 31 and 30 (from the right, rightmost bit is bit 0).
  Expr scalar = MakeConstantScalar(up_idtype, fixed_point_multiplier);
  auto multiplied_t = Multiply(tensor, scalar);

  // 4) Find the rounding scalar. This depends on where the final decimal point
  // sits. As we will be right shifting the multiplied_t, we need to first
  // calculate the total_right_shift.
  int total_right_shift = right_shift + idtype_bits - 1;

  tensor = multiplied_t;
  Expr round_scalar;
  if (param->rounding == "UPWARD") {
    auto pos_rounder = MakeConstantScalar(up_idtype, (1ll << (total_right_shift - 1)));
    round_scalar = pos_rounder;
  } else if (param->rounding == "AWAY_FROM_ZERO") {
    auto pos_rounder = MakeConstantScalar(up_idtype, (1ll << (total_right_shift - 1)));
    auto neg_rounder = MakeConstantScalar(up_idtype, (1ll << (total_right_shift - 1)) - 1);
    auto pos_rounder_t = Full(pos_rounder, out_shape, up_idtype);
    auto neg_rounder_t = Full(neg_rounder, out_shape, up_idtype);

    auto zero = MakeConstantScalar(up_idtype, 0);
    auto zero_t = Full(zero, out_shape, up_idtype);
    round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder_t,
            neg_rounder_t);
  }
  // Add the rounding scalar.
  tensor = Add(tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  auto scaled_int64_t = RightShift(tensor,
          MakeConstantScalar(up_idtype, total_right_shift));

  // 6) Add the output zero point.
  auto output_zp = MakeConstantScalar(up_idtype, param->output_zero_point);
  auto shifted_int64_t = Add(output_zp, scaled_int64_t);

  // 7) Clip to the out_dtype min/max.
  // Find the right clip min/maxes. While clipping, it is necessary that
  // clip_min and clip_max are within the dtype range of the input tensor to the
  // clip operator. For example, if the input to clip operator is int8, but the
  // out_dtype is uint8, we will get incorrect results, if we set max as 255.
  auto q_min = std::max(GetQmin(param->out_dtype), GetQmin(idtype));
  auto q_max = std::min(GetQmax(param->out_dtype), GetQmax(idtype));
  auto clipped_t = Clip(shifted_int64_t, q_min, q_max);
  auto requantized_output = Cast(clipped_t, param->out_dtype);
  return requantized_output;
}

/*
 * \brief Forward rewrite the requantize op.
 * \param ref_call The original call that will be lowered.
 * \param new_args The new mutated args to the call node.
 * \param ctx The node context.
 * \return The sequence of Relay ops for requantize op.
 * \note Lowering of the requantize operation. The requantize operator converts
 *       one quantized tensor to another quantized tensor. For the output
 *       tensor, we are provided with output scale and zero point. The
 *       computation looks like this
 *
 * Q_output = zp_output +  (scale_input)/(scale_ouptut) * (Q_input - zp_input)
 */
Expr RequantizeForwardRewrite(const Call& ref_call,
    const Array<Expr>& new_args, const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  Expr quantized_data = new_args[0];
  const auto* param = ref_call->attrs.as<RequantizeAttrs>();
  CHECK(param != nullptr);

  // Find output shape.
  Array<IndexExpr> out_shape;
  auto ref_call_t = ref_call->checked_type();
  auto output_tt = ref_call_t.as<TensorTypeNode>();
  CHECK(output_tt != nullptr) << "Type information missing."
      << " Please run infer_type pass.";
  out_shape = output_tt->shape;

  // Find input dtype.
  auto ref_input_t = ref_call->args[0]->checked_type();
  auto input_tt = ref_input_t.as<TensorTypeNode>();
  CHECK(input_tt != nullptr) << "Type information missing."
      << " Please run infer_type pass.";
  const auto input_dtype = input_tt->dtype;

  // Check rounding validity.
  CHECK(param->rounding == "UPWARD" || param->rounding == "AWAY_FROM_ZERO")
      << "QNN requantize supports two rounding modes - UPWARD and "
      << "AWAY_FROM_ZERO";
  return RequantizeLower(quantized_data, param, input_dtype, out_shape);
}

RELAY_REGISTER_OP("qnn.requantize")
.set_attr<FForwardRewrite>("FQnnForwardRewrite", RequantizeForwardRewrite);

Expr QnnLower(const Expr& expr) {
  return ForwardRewrite(expr, "FQnnForwardRewrite", nullptr, nullptr);
}
}  // namespace qnn_lower

namespace transform {
using namespace tvm::relay::transform;
Pass QnnLower() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Downcast<Function>(
          relay::qnn::qnn_lower::QnnLower(f));
  };
  return CreateFunctionPass(pass_func, 0, "QnnLower",
                            {ir::StringImm::make("InferType")});
}

TVM_REGISTER_API("relay.qnn._transform.QnnLower")
.set_body_typed(QnnLower);
}  // namespace transform

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
