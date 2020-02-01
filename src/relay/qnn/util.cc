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
 * \file src/relay/qnn/util.cc
 * \brief Utility functions for QNN.
 */

#include "util.h"

#include <limits>

#include "../transforms/pattern_util.h"

namespace tvm {
namespace relay {
namespace qnn {

/* \brief This function implements the rounding part of ARMv7 NEON VQRDMULH
 * instruction. For code reuse, the multiplied tensor is directly passed in
 * as parameter.
 */
Expr SaturatingRoundingDoublingHigh32(const Expr& input_tensor, const Expr& multiplier_expr,
                                      const Expr& scaled_tensor,
                                      const Array<IndexExpr>& input_shape,
                                      bool possible_to_overflow = true) {
  DataType hp_dtype = DataType::Int(64);
  DataType lp_dtype = DataType::Int(32);
  int64_t pos_nudge_value = (1ll << 30);
  int64_t neg_nudge_value = 1 - (1ll << 30);
  auto pos_nudge = MakeConstantScalar(hp_dtype, pos_nudge_value);
  auto neg_nudge = MakeConstantScalar(hp_dtype, neg_nudge_value);
  auto pos_nudge_t = Full(pos_nudge, input_shape, hp_dtype);
  auto neg_nudge_t = Full(neg_nudge, input_shape, hp_dtype);

  auto dividend = MakeConstantScalar(hp_dtype, 1ll << 31);

  auto zero_t = Zeros(input_shape, hp_dtype);
  auto nudged_tensor_t =
      Add(scaled_tensor, Where(GreaterEqual(scaled_tensor, zero_t), pos_nudge_t, neg_nudge_t));
  auto high32_t = Cast(Divide(nudged_tensor_t, dividend), lp_dtype);

  if (possible_to_overflow) {
    auto int32_min = MakeConstantScalar(lp_dtype, std::numeric_limits<std::int32_t>::min());
    auto int32_max = MakeConstantScalar(lp_dtype, std::numeric_limits<std::int32_t>::max());
    auto int32_max_t = Full(int32_max, input_shape, lp_dtype);
    auto int32_min_t = Full(int32_min, input_shape, lp_dtype);

    auto overflow_t =
        LogicalAnd(Equal(input_tensor, int32_min_t), Equal(multiplier_expr, int32_min_t));
    return Where(overflow_t, int32_max_t, high32_t);
  } else {
    return high32_t;
  }
}

/*
 * \brief Convert FP32 representation into fixed point representation.
 * \param double_multplier The input FP32 number.
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
std::pair<int32_t, int32_t> GetFixedPointMultiplierShift(double double_multiplier) {
  int32_t significand, exponent;
  if (double_multiplier == 0.) {
    significand = 0;
    exponent = 0;
    return std::make_pair(significand, exponent);
  }

  // Get the significand and exponent.
  double significand_d = std::frexp(double_multiplier, &exponent);

  // Convert the double significand to int significand, i.e., convert into a
  // integer where the decimal point is between bit 31 and 30. This is done by
  // multiplying the double value with 2^31 and then casting to int.
  significand_d = std::round(significand_d * (1ll << 31));
  auto significand_int64 = static_cast<int64_t>(significand_d);
  CHECK_LE(significand_int64, (1ll << 31));
  if (significand_int64 == (1ll << 31)) {
    significand_int64 /= 2;
    ++exponent;
  }
  CHECK_LE(significand_int64, std::numeric_limits<int32_t>::max());
  significand = static_cast<int32_t>(significand_int64);
  return std::make_pair(significand, exponent);
}

Expr FixedPointMultiply(Expr tensor, double multiplier, const Array<IndexExpr>& input_shape,
                        const std::string& rounding) {
  // Choose high precision datatype to be int64. This is for avoiding overflow
  // in multiplication of two int32 values.
  DataType hp_dtype = DataType::Int(64);
  DataType lp_dtype = DataType::Int(32);
  tensor = Cast(tensor, hp_dtype);

  // 1) Calculating the integer multiplier and integer shift
  int32_t fixed_point_multiplier, shift;
  std::tie(fixed_point_multiplier, shift) = GetFixedPointMultiplierShift(multiplier);
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;

  // 2) Multiply the integer multiplier
  if (left_shift != 0) {
    tensor = LeftShift(tensor, MakeConstantScalar(hp_dtype, left_shift));
  }

  // 3) Perform the multiplication in higher precision.
  // The scalar is a fixed point value of int32 where the decimal point is
  // between bits 31 and 30. After multiplying with input_tensor, the result
  // is in int64 where the decimal point is sitting between bits 31 and 30
  // (from the right, rightmost bit is bit 0). The computation is performed in
  // higher precision to avoid overflow in multiplying two int32 values.
  Expr scalar = MakeConstantScalar(hp_dtype, fixed_point_multiplier);
  Expr scaled_tensor = Multiply(tensor, scalar);

  // 4) Find the rounding scalar. This depends on where the final decimal
  // point sits. As we will be right shifting the multiplied_t, we need to
  // first calculate the total_right_shift.
  int total_right_shift = right_shift + 31;
  int64_t pos_rounding_value = (1ll << (total_right_shift - 1));

  // This lambda function gathers some shared logic in "TONEAREST" and "TFLITE"
  // rounding scheme, which calculates a rounder tensor according to the sign
  // of values in the tensor to be rounded.
  auto nearest_rounding_scalar = [&](const Expr& input_tensor, int right_shift,
                                     DataType dtype) -> Expr {
    int64_t pos_rounding_value = (1ll << (right_shift - 1));
    auto pos_rounder = MakeConstantScalar(dtype, pos_rounding_value);
    auto neg_rounder = MakeConstantScalar(dtype, pos_rounding_value - 1);
    auto pos_rounder_t = Full(pos_rounder, input_shape, dtype);
    auto neg_rounder_t = Full(neg_rounder, input_shape, dtype);

    auto zero_t = Zeros(input_shape, dtype);
    return Where(GreaterEqual(input_tensor, zero_t), pos_rounder_t, neg_rounder_t);
  };

  Expr round_scalar;
  if (rounding == "UPWARD") {
    round_scalar = MakeConstantScalar(hp_dtype, pos_rounding_value);
  } else if (rounding == "TONEAREST") {
    round_scalar = nearest_rounding_scalar(scaled_tensor, total_right_shift, hp_dtype);
  } else if (rounding == "TFLITE") {
    auto scalar_t = Full(scalar, input_shape, hp_dtype);
    bool possible_to_overflow = fixed_point_multiplier == std::numeric_limits<int32_t>::min();
    auto high32_t = SaturatingRoundingDoublingHigh32(tensor, scalar_t, scaled_tensor, input_shape,
                                                     possible_to_overflow);

    if (right_shift <= 0) {
      scaled_tensor = high32_t;
    } else {
      auto zero_t = Zeros(input_shape, lp_dtype);
      round_scalar = nearest_rounding_scalar(high32_t, right_shift, lp_dtype);
      scaled_tensor = Add(high32_t, round_scalar);
      auto rshift_expr = MakeConstantScalar(lp_dtype, right_shift);
      scaled_tensor = RightShift(scaled_tensor, rshift_expr);
    }
    return scaled_tensor;
  } else {
    LOG(FATAL) << "Rounding mode " << rounding << " not supported.";
  }

  // Add the rounding scalar.
  scaled_tensor = Add(scaled_tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  tensor = RightShift(scaled_tensor, MakeConstantScalar(hp_dtype, total_right_shift));

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return Cast(tensor, DataType::Int(32));
}

Expr FixedPointMultiplyPerChannel(Expr tensor, std::vector<double> multipliers,
                                  const Array<IndexExpr>& input_shape, int channel_axis,
                                  const std::string& rounding) {
  // Get the n dim. This will be used to expand the multiplier to match the axis.
  size_t n_dim = input_shape.size();

  // Get the num of channels/axis along which the tensor was quantized.
  int64_t n_channels = (int64_t)multipliers.size();

  // Choose high precision datatype to be int64. This is for avoiding overflow
  // in multiplication of two int32 values.
  DataType hp_dtype = DataType::Int(64);
  DataType lp_dtype = DataType::Int(32);
  tensor = Cast(tensor, hp_dtype);

  // 1) Calculating the integer multiplier and integer shift. These are calculated per axis/per
  // channel.
  std::vector<int64_t> fixed_pt_multipliers, lshifts, rshifts;
  bool lshift_required = false;
  bool rshift_required = false;
  bool possible_to_overflow = false;
  for (auto multiplier : multipliers) {
    int64_t fixed_pt_multiplier, shift;
    std::tie(fixed_pt_multiplier, shift) = GetFixedPointMultiplierShift(multiplier);
    int64_t lshift = shift > 0 ? shift : 0;
    int64_t rshift = shift > 0 ? 0 : -shift;
    fixed_pt_multipliers.push_back(fixed_pt_multiplier);
    lshifts.push_back(lshift);
    rshifts.push_back(rshift);
    lshift_required |= (lshift != 0);
    rshift_required |= (rshift != 0);
    possible_to_overflow |= (fixed_pt_multiplier == std::numeric_limits<int32_t>::min());
  }

  // 2) Multiply the integer multiplier. Convert lefts shifts into expr and multiply.
  if (lshift_required) {
    auto lshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, lshifts);
    auto exp_lshift_expr = ExpandBiasToMatchAxis(lshift_expr, n_dim, {channel_axis});
    tensor = LeftShift(tensor, exp_lshift_expr);
  }
  auto rshift_expr = MakeConstantTensor(lp_dtype, {n_channels}, rshifts);
  auto exp_rshift_expr = ExpandBiasToMatchAxis(rshift_expr, n_dim, {channel_axis});

  // 3) Perform the multiplication in higher precision.
  // The scalar is a fixed point value of int32 where the decimal point is
  // between bits 31 and 30. After multiplying with input_tensor, the result
  // is in int64 where the decimal point is sitting between bits 31 and 30
  // (from the right, rightmost bit is bit 0). The computation is performed in
  // higher precision to avoid overflow in multiplying two int32 values.
  auto fixed_pt_multiplier_expr = MakeConstantTensor(hp_dtype, {n_channels}, fixed_pt_multipliers);
  auto exp_fixed_pt_multiplier_expr =
      ExpandBiasToMatchAxis(fixed_pt_multiplier_expr, n_dim, {channel_axis});
  auto scaled_tensor = Multiply(tensor, exp_fixed_pt_multiplier_expr);

  // 4) Find the rounding scalar. This depends on where the final decimal point sits. As we will be
  // right shifting the multiplied_t, we need to first calculate the total_rshift. Further, we can
  // calculate the pos and neg rounding offset.
  std::vector<int64_t> pos_rounding_values, total_rshifts;
  for (auto rshift : rshifts) {
    int64_t total_rshift = rshift + 31;
    total_rshifts.push_back(total_rshift);
    pos_rounding_values.push_back((1ll << (total_rshift - 1)));
  }

  // This lambda function gathers some shared logic in "TONEAREST" and "TFLITE"
  // rounding scheme, which calculates a rounder tensor according to the sign
  // of values in the tensor to be rounded.
  auto nearest_rounding_tensor = [&](const Expr& input_tensor, const std::vector<int64_t>& rshifts,
                                     DataType dtype) -> Expr {
    std::vector<int64_t> pos_rounding_values, neg_rounding_values;
    for (auto rshift : rshifts) {
      int64_t pos_rounding_val = rshift > 0 ? (1ll << (rshift - 1)) : 0;
      int64_t neg_rounding_val = rshift > 0 ? ((1ll << (rshift - 1)) - 1) : 0;
      pos_rounding_values.push_back(pos_rounding_val);
      neg_rounding_values.push_back(neg_rounding_val);
    }
    // Make a Relay expr from positive and negative rounding offset values.
    auto pos_rounding_value_expr = MakeConstantTensor(dtype, {n_channels}, pos_rounding_values);
    auto exp_pos_rounding_value_expr =
        ExpandBiasToMatchAxis(pos_rounding_value_expr, n_dim, {channel_axis});
    auto pos_rounder = MakeBroadCastTo(exp_pos_rounding_value_expr, input_shape);
    auto neg_rounding_value_expr = MakeConstantTensor(dtype, {n_channels}, neg_rounding_values);
    auto exp_neg_rounding_value_expr =
        ExpandBiasToMatchAxis(neg_rounding_value_expr, n_dim, {channel_axis});
    auto neg_rounder = MakeBroadCastTo(exp_neg_rounding_value_expr, input_shape);
    auto zero_t = Zeros(input_shape, dtype);
    return Where(GreaterEqual(input_tensor, zero_t), pos_rounder, neg_rounder);
  };

  Expr round_scalar;
  if (rounding == "UPWARD") {
    // Make a Relay expr from positive and negative rounding offset values.
    auto pos_rounding_value_expr = MakeConstantTensor(hp_dtype, {n_channels}, pos_rounding_values);
    auto exp_pos_rounding_value_expr =
        ExpandBiasToMatchAxis(pos_rounding_value_expr, n_dim, {channel_axis});
    round_scalar = exp_pos_rounding_value_expr;
  } else if (rounding == "TONEAREST") {
    round_scalar = nearest_rounding_tensor(scaled_tensor, total_rshifts, hp_dtype);
  } else if (rounding == "TFLITE") {
    auto high32_t = SaturatingRoundingDoublingHigh32(
        tensor, exp_fixed_pt_multiplier_expr, scaled_tensor, input_shape, possible_to_overflow);
    if (!rshift_required) {
      return high32_t;
    } else {
      auto zero_t = Zeros(input_shape, lp_dtype);
      round_scalar = nearest_rounding_tensor(high32_t, rshifts, lp_dtype);
      scaled_tensor = Add(high32_t, round_scalar);
      return RightShift(scaled_tensor, exp_rshift_expr);
    }
  } else {
    LOG(FATAL) << "Rounding mode " << rounding << " not supported.";
  }
  // Add the rounding scalar.
  tensor = Add(scaled_tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  auto total_rshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, total_rshifts);
  auto exp_total_rshift_expr = ExpandBiasToMatchAxis(total_rshift_expr, n_dim, {channel_axis});
  tensor = RightShift(tensor, exp_total_rshift_expr);

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return Cast(tensor, DataType::Int(32));
}

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
