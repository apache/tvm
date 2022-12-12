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
 * \file src/relay/qnn/utils.cc
 * \brief Utility functions for QNN.
 */

#include "utils.h"

#include "../transforms/pattern_utils.h"

namespace tvm {
namespace relay {
namespace qnn {

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
  ICHECK_LE(significand_int64, (1ll << 31));
  if (significand_int64 == (1ll << 31)) {
    significand_int64 /= 2;
    ++exponent;
  }
  ICHECK_LE(significand_int64, std::numeric_limits<int32_t>::max());
  significand = static_cast<int32_t>(significand_int64);
  return std::make_pair(significand, exponent);
}


std::pair<int32_t, int32_t> GetFixedPointMultiplierShift_16(double double_multiplier) {
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
  significand_d = std::round(significand_d * (1ll << 15));
  auto significand_int64 = static_cast<int64_t>(significand_d);
  ICHECK_LE(significand_int64, (1ll << 15));
  if (significand_int64 == (1ll << 15)) {
    significand_int64 /= 2;
    ++exponent;
  }
  ICHECK_LE(significand_int64, std::numeric_limits<int32_t>::max());
  significand = static_cast<int32_t>(significand_int64);
  return std::make_pair(significand, exponent);
}

std::pair<int32_t, int32_t> GetFixedPointMultiplierShift_12(double double_multiplier) {
  int32_t significand, exponent;
  if (double_multiplier == 0.) {
    significand = 0;
    exponent = 0;
    return std::make_pair(significand, exponent);
  }
  int32_t shift;
  
  // Get the significand and exponent.
  double significand_d = std::frexp(double_multiplier, &exponent);

  // Convert the double significand to int significand, i.e., convert into a
  // integer where the decimal point is between bit 31 and 30. This is done by
  // multiplying the double value with 2^31 and then casting to int.

  // if (-exponent + 11 >= 8 && -exponent + 11 < 15){
  //   shift = 8 + exponent;
  // }
  // else if (-exponent + 11 >= 15 && -exponent + 11 < 18){
  //   shift = 15 + exponent;
  // }
  // else if (-exponent + 11 >= 18 && -exponent + 11 < 21){
  //   shift = 18 + exponent;
  // }
  // else if(-exponent + 11 >= 21 &&  -exponent + 11 < 25){
  //   shift = 21 +exponent;
  // }
  // else if(-exponent + 11 >= 25 &&  -exponent + 11 < 29){
  //   shift = 25 + exponent;
  // }
  // else if(-exponent + 11 >= 29 &&  -exponent + 11 < 31){
  //   shift = 29 + exponent;
  // }
  // else if(-exponent + 11 >= 31 &&  -exponent + 11 < 39){
  //   shift = 31 + exponent;
  // }
  // else{
  //   shift = 0;
  // }

  // if (-exponent + 11 >=8 && -exponent + 11 < 10){
  //   shift = 8 + exponent;
  // }
  // else if (-exponent + 11 >= 10 && -exponent + 11 < 12){
  //   shift = 10 + exponent;
  // }
  // else if (-exponent + 11 >= 12 && -exponent + 11 < 14){
  //   shift = 12 + exponent;
  // }
  // else if (-exponent + 11 >= 14 && -exponent + 11 < 16){
  //   shift = 14 + exponent;
  // }
  // else if (-exponent + 11 >= 16 && -exponent + 11 < 18){
  //   shift = 16 + exponent;
  // }
  // else if(-exponent + 11 >= 18 &&  -exponent + 11 < 20){
  //   shift = 18 +exponent;
  // }
  // else if(-exponent + 11 >= 20 &&  -exponent + 11 < 22){
  //   shift = 20 + exponent;
  // }
  // else if (-exponent + 11 >= 22 && -exponent + 11 < 24){
  //   shift = 22 + exponent;
  // }
  // else if (-exponent + 11 >= 24 && -exponent + 11 < 26){
  //   shift = 24 + exponent;
  // }
  // else if(-exponent + 11 >= 26 &&  -exponent + 11 < 28){
  //   shift = 26 +exponent;
  // }
  // else if(-exponent + 11 >= 28 &&  -exponent + 11 < 30){
  //   shift = 28 + exponent;
  // }
  // else if (-exponent + 11 >= 30 && -exponent + 11 < 32){
  //   shift = 30 + exponent;
  // }
  // else if(-exponent + 11 >= 32 &&  -exponent + 11 < 34){
  //   shift = 32 +exponent;
  // }
  // else if(-exponent + 11 >= 34 &&  -exponent + 11 < 36){
  //   shift = 34 + exponent;
  // }
  // else if (-exponent + 11 >= 36 && -exponent + 11 < 38){
  //   shift = 36 + exponent;
  // }
  // else if(-exponent + 11 >= 38 &&  -exponent + 11 < 40){
  //   shift = 38 +exponent;
  // }
  // else{
  //   shift = 11;
  // }

  // if (-exponent + 11 >=8 && -exponent + 11 < 10){
  //   shift = 8 + exponent;
  // }
  // else if (-exponent + 11 >= 10 && -exponent + 11 < 12){
  //   shift = 10 + exponent;
  // }
  // else if (-exponent + 11 >= 12 && -exponent + 11 < 14){
  //   shift = 12 + exponent;
  // }
  // else if (-exponent + 11 >= 14 && -exponent + 11 < 16){
  //   shift = 14 + exponent;
  // }
  // else if(-exponent + 11 >= 32 &&  -exponent + 11 < 34){
  //   shift = 32 +exponent;
  // }
  // else if(-exponent + 11 >= 34 &&  -exponent + 11 < 36){
  //   shift = 34 + exponent;
  // }
  // else if (-exponent + 11 >= 36 && -exponent + 11 < 38){
  //   shift = 36 + exponent;
  // }
  // else if(-exponent + 11 >= 38 &&  -exponent + 11 < 40){
  //   shift = 38 +exponent;
  // }
  // else{
  //   shift = 11;
  // }
  // printf("exponent:%d,-exponent+shift:%d,shift:%d\n", exponent, -exponent+shift, shift);
  // significand_d = std::round(significand_d * (1ll << shift));
  // auto significand_int64 = static_cast<int64_t>(significand_d);
  // ICHECK_LE(significand_int64, (1ll << shift));
  // if (significand_int64 == (1ll << shift)) {
  //   significand_int64 /= 2;
  //   ++exponent;
  // }
  significand_d = std::round(significand_d * (1ll << 11));
  auto significand_int64 = static_cast<int64_t>(significand_d);
  ICHECK_LE(significand_int64, (1ll << 11));
  if (significand_int64 == (1ll << 11)) {
    significand_int64 /= 2;
    ++exponent;
  }
  ICHECK_LE(significand_int64, std::numeric_limits<int32_t>::max());
  significand = static_cast<int32_t>(significand_int64);
  return std::make_pair(significand, exponent);
}

Expr FixedPointMultiplyToNearest_16bit(Expr tensor, double multiplier,
                                 const Array<IndexExpr>& input_shape) {
  // Choose high precision datatype to be int64. This is for avoiding overflow
  // in multiplication of two int32 values.
  DataType hp_dtype = DataType::Int(64);
  tensor = Cast(tensor, hp_dtype);

  // 1) Calculating the integer multiplier and integer shift
  int32_t fixed_point_multiplier, shift;
  std::tie(fixed_point_multiplier, shift) = GetFixedPointMultiplierShift_16(multiplier);
  //printf("16bit_fixed_pertensor:%.10f = %d * 2^%d\n",multiplier, fixed_point_multiplier, shift-16);
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
  tensor = Multiply(tensor, scalar);

  // 4) Find the rounding scalar. This depends on where the final decimal
  // point sits. As we will be right shifting the multiplied_t, we need to
  // first calculate the total_right_shift.
  int total_right_shift = right_shift + 15;
  int64_t pos_rounding_value = (1ll << (total_right_shift - 1));

  Expr round_scalar;

  auto pos_rounder = MakeConstantScalar(hp_dtype, pos_rounding_value);
  auto neg_rounder = MakeConstantScalar(hp_dtype, pos_rounding_value - 1);
  auto pos_rounder_t = Full(pos_rounder, input_shape, hp_dtype);
  auto neg_rounder_t = Full(neg_rounder, input_shape, hp_dtype);

  auto zero_t = Zeros(input_shape, hp_dtype);
  round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder_t, neg_rounder_t);

  // Add the rounding scalar.
  tensor = Add(tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  tensor = RightShift(tensor, MakeConstantScalar(hp_dtype, total_right_shift));

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return tensor;
}

Expr FixedPointMultiplyToNearest_12bit(Expr tensor, double multiplier,
                                 const Array<IndexExpr>& input_shape) {
  // Choose high precision datatype to be int64. This is for avoiding overflow
  // in multiplication of two int32 values.
  DataType hp_dtype = DataType::Int(64);
  tensor = Cast(tensor, hp_dtype);

  // 1) Calculating the integer multiplier and integer shift
  int32_t fixed_point_multiplier, shift;
  std::tie(fixed_point_multiplier, shift) = GetFixedPointMultiplierShift_12(multiplier);
  //printf("multiplier:%lf\n",multiplier);
  //printf("12bit_fixed_pertensor:%.10f = %d * 2^%d\n",multiplier, fixed_point_multiplier, shift-11);
  int32_t real_shift;
  // if (-shift + 11 >=8 && -shift + 11 < 15){
  //   real_shift = 8 + shift;
  // }
  // else if (-shift + 11 >= 15 && -shift + 11 < 18){
  //   real_shift = 15 + shift;
  // }
  // else if (-shift + 11 >= 18 && -shift + 11 < 21){
  //   real_shift = 18 + shift;
  // }
  // else if (-shift + 11 >= 21 && -shift + 11 < 25){
  //   real_shift = 21 + shift;
  // }
  // else if (-shift + 11 >= 25 && -shift + 11 < 29){
  //   real_shift = 25 + shift;
  // }
  // else if(-shift + 11 >= 29 &&  -shift + 11 < 31){
  //   real_shift = 29 +shift;
  // }
  // else if(-shift + 11 >= 31 &&  -shift + 11 < 39){
  //   real_shift = 31 + shift;
  // }
  // else{
  //   real_shift = 0;
  // }

  // if (-shift + 11 >=8 && -shift + 11 < 10){
  //   real_shift = 8 + shift;
  // }
  // else if (-shift + 11 >= 10 && -shift + 11 < 12){
  //   real_shift = 10 + shift;
  // }
  // else if (-shift + 11 >= 12 && -shift + 11 < 14){
  //   real_shift = 12 + shift;
  // }
  // else if (-shift + 11 >= 14 && -shift + 11 < 16){
  //   real_shift = 14 + shift;
  // }
  // else if (-shift + 11 >= 16 && -shift + 11 < 18){
  //   real_shift = 16 + shift;
  // }
  // else if(-shift + 11 >= 18 &&  -shift + 11 < 20){
  //   real_shift = 18 +shift;
  // }
  // else if(-shift + 11 >= 20 &&  -shift + 11 < 22){
  //   real_shift = 20 + shift;
  // }
  // else if (-shift + 11 >= 22 && -shift + 11 < 24){
  //   real_shift = 22 + shift;
  // }
  // else if (-shift + 11 >= 24 && -shift + 11 < 26){
  //   real_shift = 24 + shift;
  // }
  // else if(-shift + 11 >= 26 &&  -shift + 11 < 28){
  //   real_shift = 26 +shift;
  // }
  // else if(-shift + 11 >= 28 &&  -shift + 11 < 30){
  //   real_shift = 28 + shift;
  // }
  // else if (-shift + 11 >= 30 && -shift + 11 < 32){
  //   real_shift = 30 + shift;
  // }
  // else if(-shift + 11 >= 32 &&  -shift + 11 < 34){
  //   real_shift = 32 +shift;
  // }
  // else if(-shift + 11 >= 34 &&  -shift + 11 < 36){
  //   real_shift = 34 + shift;
  // }
  // else if (-shift + 11 >= 36 && -shift + 11 < 38){
  //   real_shift = 36 + shift;
  // }
  // else if(-shift + 11 >= 38 &&  -shift + 11 < 40){
  //   real_shift = 38 +shift;
  // }
  // else{
  //   real_shift = 11;
  // }

  // if (-shift + 11 >=8 && -shift + 11 < 10){
  //   real_shift = 8 + shift;
  // }
  // else if (-shift + 11 >= 10 && -shift + 11 < 12){
  //   real_shift = 10 + shift;
  // }
  // else if (-shift + 11 >= 12 && -shift + 11 < 14){
  //   real_shift = 12 + shift;
  // }
  // else if (-shift + 11 >= 14 && -shift + 11 < 16){
  //   real_shift = 14 + shift;
  // }
  // else if(-shift + 11 >= 32 &&  -shift + 11 < 34){
  //   real_shift = 32 +shift;
  // }
  // else if(-shift + 11 >= 34 &&  -shift + 11 < 36){
  //   real_shift = 34 + shift;
  // }
  // else if (-shift + 11 >= 36 && -shift + 11 < 38){
  //   real_shift = 36 + shift;
  // }
  // else if(-shift + 11 >= 38 &&  -shift + 11 < 40){
  //   real_shift = 38 +shift;
  // }
  // else{
  //   real_shift = 11;
  // }
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  //  printf("pertensor\n");
  //  if(left_shift!=0){
  //      printf("left_shift:%d\n",left_shift);
  //      printf("total_right_shift:%d\n",right_shift+11);
  //  }
  // 2) Multiply the integer multiplier
  // if (left_shift != 0) {
  //   tensor = LeftShift(tensor, MakeConstantScalar(hp_dtype, left_shift));
  // }

  // 3) Perform the multiplication in higher precision.
  // The scalar is a fixed point value of int32 where the decimal point is
  // between bits 31 and 30. After multiplying with input_tensor, the result
  // is in int64 where the decimal point is sitting between bits 31 and 30
  // (from the right, rightmost bit is bit 0). The computation is performed in
  // higher precision to avoid overflow in multiplying two int32 values.
  Expr scalar = MakeConstantScalar(hp_dtype, fixed_point_multiplier);
  tensor = Multiply(tensor, scalar);

  // 4) Find the rounding scalar. This depends on where the final decimal
  // point sits. As we will be right shifting the multiplied_t, we need to
  // first calculate the total_right_shift.
  int total_right_shift = right_shift + 11 - left_shift;
  //int total_right_shift = right_shift + real_shift - left_shift;
  //printf("total_right_shift:%d\n",total_right_shift);
  if(total_right_shift<0){
    total_right_shift = 0;
  }
  int64_t pos_rounding_value = (1ll << (total_right_shift - 1));

  Expr round_scalar;

  auto pos_rounder = MakeConstantScalar(hp_dtype, pos_rounding_value);
  auto neg_rounder = MakeConstantScalar(hp_dtype, pos_rounding_value - 1);
  auto pos_rounder_t = Full(pos_rounder, input_shape, hp_dtype);
  auto neg_rounder_t = Full(neg_rounder, input_shape, hp_dtype);

  auto zero_t = Zeros(input_shape, hp_dtype);
  round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder_t, neg_rounder_t);

  // Add the rounding scalar.
  tensor = Add(tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  tensor = RightShift(tensor, MakeConstantScalar(hp_dtype, total_right_shift));

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return tensor;
}

Expr FixedPointMultiplyToNearest(Expr tensor, double multiplier,
                                 const Array<IndexExpr>& input_shape) {
  // Choose high precision datatype to be int64. This is for avoiding overflow
  // in multiplication of two int32 values.
  DataType hp_dtype = DataType::Int(64);
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
  tensor = Multiply(tensor, scalar);

  // 4) Find the rounding scalar. This depends on where the final decimal
  // point sits. As we will be right shifting the multiplied_t, we need to
  // first calculate the total_right_shift.
  int total_right_shift = right_shift + 31;
  int64_t pos_rounding_value = (1ll << (total_right_shift - 1));

  Expr round_scalar;

  auto pos_rounder = MakeConstantScalar(hp_dtype, pos_rounding_value);
  auto neg_rounder = MakeConstantScalar(hp_dtype, pos_rounding_value - 1);
  auto pos_rounder_t = Full(pos_rounder, input_shape, hp_dtype);
  auto neg_rounder_t = Full(neg_rounder, input_shape, hp_dtype);

  auto zero_t = Zeros(input_shape, hp_dtype);
  round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder_t, neg_rounder_t);

  // Add the rounding scalar.
  tensor = Add(tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  tensor = RightShift(tensor, MakeConstantScalar(hp_dtype, total_right_shift));

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return Cast(tensor, DataType::Int(32));
}

Expr FixedPointMultiplyPerChannel_12bit(Expr tensor, std::vector<double> multipliers,
                                  const Array<IndexExpr>& input_shape, int channel_axis,
                                  const std::string& rounding) {
  // Get the n dim. This will be used to expand the multiplier to match the axis.
  size_t n_dim = input_shape.size();

  // Get the num of channels/axis along which the tensor was quantized.
  int64_t n_channels = (int64_t)multipliers.size();

  // Choose high precision datatype to be int64. This is for avoiding overflow
  // in multiplication of two int32 values.
  DataType hp_dtype = DataType::Int(64);
  tensor = Cast(tensor, hp_dtype);

  // 1) Calculating the integer multiplier and integer shift. These are calculated per axis/per
  // channel.
  std::vector<int32_t> fixed_pt_multipliers, lshifts, rshifts;
  int32_t real_shift;
  bool is_lshift_required = false;
  //printf("one point perchannel:\n");
  for (auto multiplier : multipliers) {
    int32_t fixed_pt_multiplier, shift;
    //printf("multiplier = %lf\n",multiplier);
    std::tie(fixed_pt_multiplier, shift) = GetFixedPointMultiplierShift_12(multiplier);
    // if (-shift + 11 >=8 && -shift + 11 < 15){
    //   real_shift = 8 + shift;
    // }
    // else if (-shift + 11 >= 15 && -shift + 11 < 18){
    //   real_shift = 15 + shift;
    // }
    // else if (-shift + 11 >= 18 && -shift + 11 < 21){
    //   real_shift = 18 + shift;
    // }
    // else if (-shift + 11 >= 21 && -shift + 11 < 25){
    //   real_shift = 21 + shift;
    // }
    // else if (-shift + 11 >= 25 && -shift + 11 < 29){
    //   real_shift = 25 + shift;
    // }
    // else if(-shift + 11 >= 29 &&  -shift + 11 < 31){
    //   real_shift = 29 +shift;
    // }
    // else if(-shift + 11 >= 31 &&  -shift + 11 < 39){
    //   real_shift = 31 + shift;
    // }
    // else{
    //   real_shift = 0;
    // }

    // if (-shift + 11 >=8 && -shift + 11 < 10){
    //   real_shift = 8 + shift;
    // }
    // else if (-shift + 11 >= 10 && -shift + 11 < 12){
    //   real_shift = 10 + shift;
    // }
    // else if (-shift + 11 >= 12 && -shift + 11 < 14){
    //   real_shift = 12 + shift;
    // }
    // else if (-shift + 11 >= 14 && -shift + 11 < 16){
    //   real_shift = 14 + shift;
    // }
    // else if (-shift + 11 >= 16 && -shift + 11 < 18){
    //   real_shift = 16 + shift;
    // }
    // else if(-shift + 11 >= 18 &&  -shift + 11 < 20){
    //   real_shift = 18 +shift;
    // }
    // else if(-shift + 11 >= 20 &&  -shift + 11 < 22){
    //   real_shift = 20 + shift;
    // }
    // else if (-shift + 11 >= 22 && -shift + 11 < 24){
    //   real_shift = 22 + shift;
    // }
    // else if (-shift + 11 >= 24 && -shift + 11 < 26){
    //   real_shift = 24 + shift;
    // }
    // else if(-shift + 11 >= 26 &&  -shift + 11 < 28){
    //   real_shift = 26 +shift;
    // }
    // else if(-shift + 11 >= 28 &&  -shift + 11 < 30){
    //   real_shift = 28 + shift;
    // }
    // else if (-shift + 11 >= 30 && -shift + 11 < 32){
    //   real_shift = 30 + shift;
    // }
    // else if(-shift + 11 >= 32 &&  -shift + 11 < 34){
    //   real_shift = 32 +shift;
    // }
    // else if(-shift + 11 >= 34 &&  -shift + 11 < 36){
    //   real_shift = 34 + shift;
    // }
    // else if (-shift + 11 >= 36 && -shift + 11 < 38){
    //   real_shift = 36 + shift;
    // }
    // else if(-shift + 11 >= 38 &&  -shift + 11 < 40){
    //   real_shift = 38 +shift;
    // }
    // else{
    //   real_shift = 11;
    // }

    // if (-shift + 11 >=8 && -shift + 11 < 10){
    //   real_shift = 8 + shift;
    // }
    // else if (-shift + 11 >= 10 && -shift + 11 < 12){
    //   real_shift = 10 + shift;
    // }
    // else if (-shift + 11 >= 12 && -shift + 11 < 14){
    //   real_shift = 12 + shift;
    // }
    // else if (-shift + 11 >= 14 && -shift + 11 < 16){
    //   real_shift = 14 + shift;
    // }
    // else if(-shift + 11 >= 32 &&  -shift + 11 < 34){
    //   real_shift = 32 +shift;
    // }
    // else if(-shift + 11 >= 34 &&  -shift + 11 < 36){
    //   real_shift = 34 + shift;
    // }
    // else if (-shift + 11 >= 36 && -shift + 11 < 38){
    //   real_shift = 36 + shift;
    // }
    // else if(-shift + 11 >= 38 &&  -shift + 11 < 40){
    //   real_shift = 38 +shift;
    // }
    // else{
    //   real_shift = 11;
    // }
    // printf("12bit_fixed_perchannel:%.10f = %d * 2^%d\n",multiplier, fixed_pt_multiplier, shift-real_shift);
    //printf("%d\n",-shift+11);
    //printf("12bit_fixed_perchannel:%.10f = %d * 2^%d\n",multiplier, fixed_pt_multiplier, shift-11);
    int lshift = shift > 0 ? shift : 0;
    int rshift = shift > 0 ? 0 : -shift;
    fixed_pt_multipliers.push_back(fixed_pt_multiplier);
    lshifts.push_back(lshift);
    rshifts.push_back(rshift);
    is_lshift_required = is_lshift_required | (lshift != 0);
    // if(lshift!=0){
    //   printf("left_shift:%d\n",lshift);
    //   printf("total_right_shift:%d\n",rshift+11);
    // }

  }

  // 2) Multiply the integer multiplier. Convert lefts shifts into expr and multiply.
  // if (is_lshift_required) {
  //   auto lshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, lshifts);
  //   auto exp_lshift_expr = ExpandBiasToMatchAxis(lshift_expr, n_dim, {channel_axis});
  //   tensor = LeftShift(tensor, exp_lshift_expr);
  // }

  // 3) Perform the multiplication in higher precision.
  // The scalar is a fixed point value of int32 where the decimal point is
  // between bits 31 and 30. After multiplying with input_tensor, the result
  // is in int64 where the decimal point is sitting between bits 31 and 30
  // (from the right, rightmost bit is bit 0). The computation is performed in
  // higher precision to avoid overflow in multiplying two int32 values.
  auto fixed_pt_multiplier_expr = MakeConstantTensor(hp_dtype, {n_channels}, fixed_pt_multipliers);
  auto exp_fixed_pt_multiplier_expr =
      ExpandBiasToMatchAxis(fixed_pt_multiplier_expr, n_dim, {channel_axis});
  tensor = Multiply(tensor, exp_fixed_pt_multiplier_expr);

  // 4) Find the rounding scalar. This depends on where the final decimal point sits. As we will be
  // right shifting the multiplied_t, we need to first calculate the total_rshift. Further, we can
  // calculate the pos and neg rounding offset.
  std::vector<int64_t> pos_rounding_values, neg_rounding_values, total_rshifts;
  //for (auto rshift : rshifts) {
  for (size_t i = 0; i < rshifts.size(); i++){
    int total_rshift = rshifts[i] + 11 - lshifts[i];
    //int total_rshift = rshifts[i] + real_shift - lshifts[i];
    if(total_rshift < 0){
      total_rshift = 0;
    }
    total_rshifts.push_back(total_rshift);
    pos_rounding_values.push_back((1ll << (total_rshift - 1)));
    neg_rounding_values.push_back((1ll << (total_rshift - 1)) - 1);
  }
  // Make a Relay expr from positive and negative rounding offset values.
  auto pos_rounding_value_expr = MakeConstantTensor(hp_dtype, {n_channels}, pos_rounding_values);
  auto exp_pos_rounding_value_expr =
      ExpandBiasToMatchAxis(pos_rounding_value_expr, n_dim, {channel_axis});
  auto neg_rounding_value_expr = MakeConstantTensor(hp_dtype, {n_channels}, neg_rounding_values);
  auto exp_neg_rounding_value_expr =
      ExpandBiasToMatchAxis(neg_rounding_value_expr, n_dim, {channel_axis});

  Expr round_scalar;
  if (rounding == "UPWARD") {
    round_scalar = exp_pos_rounding_value_expr;
  } else if (rounding == "TONEAREST") {
    // To satisfy where op shape requirements, the rounding values are broadcasted.
    auto pos_rounder = BroadCastTo(exp_pos_rounding_value_expr, input_shape);
    auto neg_rounder = BroadCastTo(exp_neg_rounding_value_expr, input_shape);

    auto zero_t = Zeros(input_shape, hp_dtype);
    round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder, neg_rounder);
  } else {
    LOG(FATAL) << "Rounding mode " << rounding << " not supported.";
  }
  // Add the rounding scalar.
  tensor = Add(tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  auto total_rshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, total_rshifts);
  auto exp_total_rshift_expr = ExpandBiasToMatchAxis(total_rshift_expr, n_dim, {channel_axis});
  tensor = RightShift(tensor, exp_total_rshift_expr);

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return tensor;
}

Expr FixedPointMultiplyPerChannel_16bit(Expr tensor, std::vector<double> multipliers,
                                  const Array<IndexExpr>& input_shape, int channel_axis,
                                  const std::string& rounding) {
  // Get the n dim. This will be used to expand the multiplier to match the axis.
  size_t n_dim = input_shape.size();

  // Get the num of channels/axis along which the tensor was quantized.
  int64_t n_channels = (int64_t)multipliers.size();

  // Choose high precision datatype to be int64. This is for avoiding overflow
  // in multiplication of two int32 values.
  DataType hp_dtype = DataType::Int(64);
  tensor = Cast(tensor, hp_dtype);

  // 1) Calculating the integer multiplier and integer shift. These are calculated per axis/per
  // channel.
  std::vector<int32_t> fixed_pt_multipliers, lshifts, rshifts;
  bool is_lshift_required = false;
  //printf("one point:\n");
  for (auto multiplier : multipliers) {
    int32_t fixed_pt_multiplier, shift;
    std::tie(fixed_pt_multiplier, shift) = GetFixedPointMultiplierShift_16(multiplier);
    //printf("16bit_fixed_perchannel:%.10f = %d * 2^%d\n",multiplier, fixed_pt_multiplier, shift-15);
    int lshift = shift > 0 ? shift : 0;
    int rshift = shift > 0 ? 0 : -shift;
    fixed_pt_multipliers.push_back(fixed_pt_multiplier);
    lshifts.push_back(lshift);
    rshifts.push_back(rshift);
    is_lshift_required = is_lshift_required | (lshift != 0);
  }

  // 2) Multiply the integer multiplier. Convert lefts shifts into expr and multiply.
  if (is_lshift_required) {
    auto lshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, lshifts);
    auto exp_lshift_expr = ExpandBiasToMatchAxis(lshift_expr, n_dim, {channel_axis});
    tensor = LeftShift(tensor, exp_lshift_expr);
  }

  // 3) Perform the multiplication in higher precision.
  // The scalar is a fixed point value of int32 where the decimal point is
  // between bits 31 and 30. After multiplying with input_tensor, the result
  // is in int64 where the decimal point is sitting between bits 31 and 30
  // (from the right, rightmost bit is bit 0). The computation is performed in
  // higher precision to avoid overflow in multiplying two int32 values.
  auto fixed_pt_multiplier_expr = MakeConstantTensor(hp_dtype, {n_channels}, fixed_pt_multipliers);
  auto exp_fixed_pt_multiplier_expr =
      ExpandBiasToMatchAxis(fixed_pt_multiplier_expr, n_dim, {channel_axis});
  tensor = Multiply(tensor, exp_fixed_pt_multiplier_expr);

  // 4) Find the rounding scalar. This depends on where the final decimal point sits. As we will be
  // right shifting the multiplied_t, we need to first calculate the total_rshift. Further, we can
  // calculate the pos and neg rounding offset.
  std::vector<int64_t> pos_rounding_values, neg_rounding_values, total_rshifts;
  for (auto rshift : rshifts) {
    int total_rshift = rshift + 15;
    total_rshifts.push_back(total_rshift);
    pos_rounding_values.push_back((1ll << (total_rshift - 1)));
    neg_rounding_values.push_back((1ll << (total_rshift - 1)) - 1);
  }
  // Make a Relay expr from positive and negative rounding offset values.
  auto pos_rounding_value_expr = MakeConstantTensor(hp_dtype, {n_channels}, pos_rounding_values);
  auto exp_pos_rounding_value_expr =
      ExpandBiasToMatchAxis(pos_rounding_value_expr, n_dim, {channel_axis});
  auto neg_rounding_value_expr = MakeConstantTensor(hp_dtype, {n_channels}, neg_rounding_values);
  auto exp_neg_rounding_value_expr =
      ExpandBiasToMatchAxis(neg_rounding_value_expr, n_dim, {channel_axis});

  Expr round_scalar;
  if (rounding == "UPWARD") {
    round_scalar = exp_pos_rounding_value_expr;
  } else if (rounding == "TONEAREST") {
    // To satisfy where op shape requirements, the rounding values are broadcasted.
    auto pos_rounder = BroadCastTo(exp_pos_rounding_value_expr, input_shape);
    auto neg_rounder = BroadCastTo(exp_neg_rounding_value_expr, input_shape);

    auto zero_t = Zeros(input_shape, hp_dtype);
    round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder, neg_rounder);
  } else {
    LOG(FATAL) << "Rounding mode " << rounding << " not supported.";
  }
  // Add the rounding scalar.
  tensor = Add(tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  auto total_rshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, total_rshifts);
  auto exp_total_rshift_expr = ExpandBiasToMatchAxis(total_rshift_expr, n_dim, {channel_axis});
  tensor = RightShift(tensor, exp_total_rshift_expr);

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return tensor;
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
  tensor = Cast(tensor, hp_dtype);

  // 1) Calculating the integer multiplier and integer shift. These are calculated per axis/per
  // channel.
  std::vector<int32_t> fixed_pt_multipliers, lshifts, rshifts;
  bool is_lshift_required = false;
  for (auto multiplier : multipliers) {
    int32_t fixed_pt_multiplier, shift;
    std::tie(fixed_pt_multiplier, shift) = GetFixedPointMultiplierShift(multiplier);
    int lshift = shift > 0 ? shift : 0;
    int rshift = shift > 0 ? 0 : -shift;
    fixed_pt_multipliers.push_back(fixed_pt_multiplier);
    lshifts.push_back(lshift);
    rshifts.push_back(rshift);
    is_lshift_required = is_lshift_required | (lshift != 0);
  }

  // 2) Multiply the integer multiplier. Convert lefts shifts into expr and multiply.
  if (is_lshift_required) {
    auto lshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, lshifts);
    auto exp_lshift_expr = ExpandBiasToMatchAxis(lshift_expr, n_dim, {channel_axis});
    tensor = LeftShift(tensor, exp_lshift_expr);
  }

  // 3) Perform the multiplication in higher precision.
  // The scalar is a fixed point value of int32 where the decimal point is
  // between bits 31 and 30. After multiplying with input_tensor, the result
  // is in int64 where the decimal point is sitting between bits 31 and 30
  // (from the right, rightmost bit is bit 0). The computation is performed in
  // higher precision to avoid overflow in multiplying two int32 values.
  auto fixed_pt_multiplier_expr = MakeConstantTensor(hp_dtype, {n_channels}, fixed_pt_multipliers);
  auto exp_fixed_pt_multiplier_expr =
      ExpandBiasToMatchAxis(fixed_pt_multiplier_expr, n_dim, {channel_axis});
  tensor = Multiply(tensor, exp_fixed_pt_multiplier_expr);

  // 4) Find the rounding scalar. This depends on where the final decimal point sits. As we will be
  // right shifting the multiplied_t, we need to first calculate the total_rshift. Further, we can
  // calculate the pos and neg rounding offset.
  std::vector<int64_t> pos_rounding_values, neg_rounding_values, total_rshifts;
  for (auto rshift : rshifts) {
    int total_rshift = rshift + 31;
    total_rshifts.push_back(total_rshift);
    pos_rounding_values.push_back((1ll << (total_rshift - 1)));
    neg_rounding_values.push_back((1ll << (total_rshift - 1)) - 1);
  }
  // Make a Relay expr from positive and negative rounding offset values.
  auto pos_rounding_value_expr = MakeConstantTensor(hp_dtype, {n_channels}, pos_rounding_values);
  auto exp_pos_rounding_value_expr =
      ExpandBiasToMatchAxis(pos_rounding_value_expr, n_dim, {channel_axis});
  auto neg_rounding_value_expr = MakeConstantTensor(hp_dtype, {n_channels}, neg_rounding_values);
  auto exp_neg_rounding_value_expr =
      ExpandBiasToMatchAxis(neg_rounding_value_expr, n_dim, {channel_axis});

  Expr round_scalar;
  if (rounding == "UPWARD") {
    round_scalar = exp_pos_rounding_value_expr;
  } else if (rounding == "TONEAREST") {
    // To satisfy where op shape requirements, the rounding values are broadcasted.
    auto pos_rounder = BroadCastTo(exp_pos_rounding_value_expr, input_shape);
    auto neg_rounder = BroadCastTo(exp_neg_rounding_value_expr, input_shape);

    auto zero_t = Zeros(input_shape, hp_dtype);
    round_scalar = Where(GreaterEqual(tensor, zero_t), pos_rounder, neg_rounder);
  } else {
    LOG(FATAL) << "Rounding mode " << rounding << " not supported.";
  }
  // Add the rounding scalar.
  tensor = Add(tensor, round_scalar);

  // 5) Simply right shift the result to get the final output.
  auto total_rshift_expr = MakeConstantTensor(hp_dtype, {n_channels}, total_rshifts);
  auto exp_total_rshift_expr = ExpandBiasToMatchAxis(total_rshift_expr, n_dim, {channel_axis});
  tensor = RightShift(tensor, exp_total_rshift_expr);

  // 6) The fixed point multiplication keeps the value in int32 range. Casting back to int32.
  return Cast(tensor, DataType::Int(32));
}

std::string SelectRequntizeParameter(const std::string& arg_value, const std::string& cfg_value,
                                     const bool is_cfg_default, const std::string& name) {
  if (arg_value == "None") {
    return cfg_value;
  } else {
    if (!is_cfg_default && arg_value != cfg_value) {
      DLOG(INFO) << "The value of parameter \"" << name
                 << "\" from the non-default requantize config will not be used. The value "
                    "provided from "
                    "requantize function argument will be used instead. The value used is \""
                 << arg_value << "\".";
    }
    return arg_value;
  }
}

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
