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
#include <tvm/relay/transform.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/quantize_util.h>
#include <tvm/relay/attrs/qnn.h>
#include "pattern_util.h"

namespace tvm {
namespace relay {


// Lowering of qnn.requantize op
void GetFixedPointMultiplierShift(double double_multiplier,
    int32_t* fixed_point_multiplier, int* shift,
    const DataType& idtype) {

  int acc_dtype_bits = idtype.bits();

  if (double_multiplier == 0.) {
    *fixed_point_multiplier = 0;
    *shift = 0;
    return;
  }
  const double q = std::frexp(double_multiplier, shift);
  auto q_fixed = static_cast<int64_t>(std::round(q * (1ll << (acc_dtype_bits - 1))));
  CHECK_LE(q_fixed, (1ll << (acc_dtype_bits - 1)));
  if (q_fixed == (1ll << (acc_dtype_bits - 1))) {
    q_fixed /= 2;
    ++*shift;
  }
  CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  *fixed_point_multiplier = static_cast<int32_t>(q_fixed);
}

Expr MultiplyByIntegerMuliplier(const Expr& convolved_tensor,
    const int32_t fixed_point_multiplier, const int left_shift,
    const RequantizeAttrs*& param, const DataType& idtype,
    const Array<IndexExpr>& out_shape) {
  // TODO (janimesh) - How to add the overflow checks here. TFLite code snippet is
  // bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
  // return overflow ? std::numeric_limits<std::int32_t>::max() : .....;/

  // The calculations are done in upcast of idtype to retain precision.
  int acc_dtype_bits = idtype.bits();
  DataType up_idtype = Int(2 * acc_dtype_bits);

  auto tensor = convolved_tensor;
  // Typically the left_shift will be 0 if the original scale is > 0.5.
  if (left_shift != 0) {
    tensor = Multiply(tensor, MakeConstantScalar(idtype, 1 << left_shift));
  }

  // Upcast the computation to Int64 and multiply the multiplier.
  Expr scalar = MakeConstantScalar(up_idtype, fixed_point_multiplier);
  auto multiplied_t = Multiply(Cast(tensor, up_idtype), scalar);

  // Since, we are performing fixed point computation. We are only interested in
  // higher 16/32 bits. But before that, we also need to perform rounding.
  // This is fixed point rounding. So, the rounder add scalar depends if the
  // input is positive.
  auto zero = MakeConstantScalar(up_idtype, 0);
  auto pos_threshold = MakeConstantScalar(up_idtype,
          1ll << (acc_dtype_bits - 2));
  auto neg_threshold = MakeConstantScalar(up_idtype,
          (1 - (1ll << (acc_dtype_bits - 2))));
  auto pos_rounder = Full(pos_threshold, out_shape, up_idtype);
  auto neg_rounder = Full(neg_threshold, out_shape, up_idtype);
  auto rounding_scalar = Where(GreaterEqual(multiplied_t, zero), pos_rounder, neg_rounder);
  auto rounded_tensor = Add(multiplied_t, rounding_scalar);

  // Perform right shift to get the first 16/32 bits.
  // The result is first doubled and the first 15/31 bits are obtained. This is
  // done by just right shifting the result by 15/31 bits.
  auto right_shift_scalar = MakeConstantScalar(up_idtype, (acc_dtype_bits - 1));
  auto scaled_t = RightShift(rounded_tensor, right_shift_scalar);
  auto q_imin = get_qmin(idtype);
  auto q_imax = get_qmax(idtype);
  auto integer_multiplied_t = Cast(Clip(scaled_t, q_imin, q_imax),
          idtype);
  return integer_multiplied_t;
}

Expr ShiftByIntegerShift(const Expr& multiplied_t,
    const int& exponent, const RequantizeAttrs*& param,
    const DataType& idtype, const Array<IndexExpr>& out_shape) {
  CHECK_GE(exponent, 0);
  int acc_dtype_bits = idtype.bits();
  CHECK_LE(exponent, (acc_dtype_bits - 1));

  // We need to perform rounding. The rounding here is closest to the power
  // of 2. The exponent basically represents the decimal point. We need to round
  // at the decimal point.
  auto tensor = multiplied_t;
  if (exponent != 0) {
    auto pos_rounder = MakeConstantScalar(idtype, (1ll << (exponent - 1)));
    auto neg_rounder = MakeConstantScalar(idtype, (1ll << (exponent - 1)) - 1);
    auto pos_rounder_t = Full(pos_rounder, out_shape, idtype);
    auto neg_rounder_t = Full(neg_rounder, out_shape, idtype);

    auto zero = MakeConstantScalar(idtype, 0);
    auto zero_t = Full(zero, out_shape, idtype);
    auto round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder_t,
            neg_rounder_t);
    tensor = Add(tensor, round_scalar);
  }

  // Right shift by exponent to approximate the division.
  auto scaled_t = RightShift(tensor,
          MakeConstantScalar(idtype, exponent));
  return scaled_t;
}


/*
 * Requantization using only integer computation. Here, the computation is
 * converted to a fixed point computation by computing output multiplier and
 * shift. This is useful, if the target device does not support/have very
 * expensive floating point computations.
 *
 * Original compuation is scale_fp32 * quantized_tensor.  To convert into
 * integer computation, the multiplication with fp32 scalar can be replaced by
 * multiplication with an int value and then right shifting the result. This
 * approximates the floating point computation with a fixed point computation.
 *
 * The whole computaition this can be broken down into following steps 
 * 1) Calculate the integer multiplier and integer shift.
 * 2) Multiply the integer multiplier with quantized tensor.
 * 3) Right shift the result.
 *
 * The only thing complicating the above computations is the tedious approach of
 * handling rounding.
 */
Expr RequantizeInt(const Expr& convolved_tensor,
    const RequantizeAttrs*& param, const DataType& idtype,
    const Array<IndexExpr>& out_shape) {

  double double_multiplier = param->input_scale/param->output_scale;
  // 1) Calculating the integer multiplier and integer shift
  int32_t fixed_point_multiplier;
  int shift;
  GetFixedPointMultiplierShift(double_multiplier, &fixed_point_multiplier,
          &shift, idtype);

  // 2) Multiply the integer multiplier
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  auto multiplied_t = MultiplyByIntegerMuliplier(convolved_tensor,
          fixed_point_multiplier, left_shift, param, idtype, out_shape);

  // 3) Divide by the denominator or right shift the result.
  auto scaled_int32_t = ShiftByIntegerShift(multiplied_t,
          right_shift, param, idtype, out_shape);

  // 4) Clip to the out_dtype min/max.
  auto q_min = std::max(get_qmin(param->out_dtype), get_qmin(idtype));
  auto q_max = std::min(get_qmax(param->out_dtype), get_qmax(idtype));
  auto clipped_t = Clip(scaled_int32_t, q_min, q_max);
  auto requantized_output = Cast(clipped_t, param->out_dtype);
  return requantized_output;
}

/* 
 * Requantization using floating computation. Here we can multiply the scale to
 * the convolved_tensor, round to nearest integer and then cast back to int32.
 */
Expr RequantizeFloat(const Expr& convolved_tensor,
    const RequantizeAttrs*& param, const DataType& idtype,
    const Array<IndexExpr>& out_shape) {
  double double_multiplier = param->input_scale/param->output_scale;
  auto scalar_multiplier = MakeConstantScalar(Float(32), double_multiplier);

  // Multiply the convolved tensor with the new scale.
  auto casted_t = Cast(convolved_tensor, Float(32));
  auto multiplied_t = Round(Multiply(casted_t, scalar_multiplier));
  auto q_imin = get_qmin(idtype);
  auto q_imax = get_qmax(idtype);
  auto scaled_int32_t = Cast(Clip(multiplied_t, q_imin, q_imax),
          idtype);

  // Clip to the out_dtype min/max.
  // Clip limits must be smaller than the dtype of the input tensor.
  auto q_min = std::max(get_qmin(param->out_dtype), get_qmin(idtype));
  auto q_max = std::min(get_qmax(param->out_dtype), get_qmax(idtype));
  auto clipped_t = Clip(scaled_int32_t, q_min, q_max);
  auto requantized_output = Cast(clipped_t, param->out_dtype);
  return requantized_output;
}

/*
 * Lowering of the requantize operation. The requantize operator converts one
 * quantized tensor to another quantized tensor. For the output tensor, we are
 * provided with output scale and zero point. The computation looks like this
 *
 * Q_output = zp_output +  (scale_input)/(scale_ouptut) * (Q_input - zp_input)
 *
 * The above computation can be done in floating point as the scales are in
 * FP32. Alternatively, we can approximate floating point with fixed point
 * computation. This is controlled by use_int_compute.
 */
Expr RequantizeForwardRewrite(const Call& ref_call,
    const Array<Expr>& new_args, const NodeRef& ctx) {
  CHECK_EQ(new_args.size(), 1);
  Expr quantized_data = new_args[0];
  const auto* param = ref_call->attrs.as<RequantizeAttrs>();

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

  // Check for current quantization support.
  CHECK_EQ(param->input_zero_point, 0)
      << "Encountered non-zero zero point."
      << " Only symmetric quantization supported for now.";
  CHECK_EQ(param->output_zero_point, 0)
      << "Encountered non-zero zero point."
      << " Only symmetric quantization supported for now.";

  if (param->use_int_compute) {
    return RequantizeInt(quantized_data, param, input_dtype, out_shape);
  } else {
    return RequantizeFloat(quantized_data, param, input_dtype, out_shape);
  }
}


RELAY_REGISTER_OP("qnn.requantize")
.set_attr<FForwardRewrite>("FQuantizeForwardRewrite", RequantizeForwardRewrite);



TVM_REGISTER_API("relay._quantize.rewrite")
.set_body_typed<Expr(Expr)>([](const Expr& e) {
  Expr ret = ForwardRewrite(e, "FQuantizeForwardRewrite", nullptr, nullptr);
  return ret;
});


}  // namespace relay
}  // namespace tvm
