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
 * \file dtype_conversion.cc
 * \brief Header file of data type conversion routines.
 */
#include "dtype_conversion.h"

namespace tvm {
namespace tir {

PrimExpr ReinterpretAsUInt(PrimExpr value) {
  return reinterpret(GetStorageUIntDType(value.dtype()), value);
}

DataType GetStorageUIntDType(DataType dtype) { return DataType::UInt(dtype.bits(), dtype.lanes()); }

PrimExpr FpToFp(PrimExpr src_value, DataType tgt_dtype, RoundingMode round_mode) {
  FloatConfig src_fp = FloatConfig::FromDataType(src_value.dtype()),
              tgt_fp = FloatConfig::FromDataType(tgt_dtype);
  int exponent_delta = tgt_fp.exponent - src_fp.exponent;
  int bias_delta = tgt_fp.bias - src_fp.bias;
  int mantissa_delta = tgt_fp.mantissa - src_fp.mantissa;
  DataType src_uint = GetStorageUIntDType(src_value.dtype()),
           tgt_uint = GetStorageUIntDType(tgt_dtype);
  PrimExpr src_uint_value = ReinterpretAsUInt(src_value);
  if (mantissa_delta < 0) {
    // use rounding
    CHECK_EQ(round_mode, RoundingMode::kHalfToEven)
        << "Currently we only support HalfToEven rounding mode.";
    PrimExpr rounding_bias = ((src_uint_value >> (-mantissa_delta)) & 1) +
                             make_const(src_uint, (int64_t(1) << (-mantissa_delta - 1)) - 1);
    src_uint_value = src_uint_value + rounding_bias;
  }
  if (exponent_delta == 0) {
    // number of exponent bits exactly matches
    PrimExpr ret = src_uint_value;
    if (mantissa_delta >= 0) {
      ret = cast(tgt_uint, ret) << mantissa_delta;
      if (bias_delta != 0) {
        ret = ret + (make_const(tgt_uint, bias_delta) << tgt_fp.mantissa);
      }
    } else {  // mantissa_delta < 0
      ret = cast(tgt_uint, ret >> mantissa_delta);
      if (bias_delta != 0) {
        ret = ret + (make_const(tgt_uint, bias_delta) << tgt_fp.mantissa);
      }
    }
    return reinterpret(tgt_dtype, ret);
  } else {
    // number of exponent bits mismatch.
    PrimExpr ret_mantissa =
        cast(tgt_uint, ((mantissa_delta >= 0) ? (src_uint_value >> mantissa_delta)
                                              : (src_uint_value << (-mantissa_delta)))) &
        make_const(tgt_uint, (int64_t(1) << (tgt_fp.mantissa)) - 1);
    PrimExpr ret_exponent =
        cast(tgt_uint, (((src_uint_value << 1) >> (src_fp.mantissa + 1)) + bias_delta))
        << tgt_fp.mantissa;
    PrimExpr ret_sign = make_const(tgt_uint, int64_t(1) << (tgt_fp.mantissa + tgt_fp.exponent));
    return reinterpret(tgt_dtype, ret_mantissa | ret_exponent | ret_sign);
  }
}

PrimExpr IntToFp(PrimExpr src_value, FloatConfig tgt_fp) {
  LOG(FATAL) << "Not implemented yet";
}

PrimExpr FpToInt(PrimExpr src_value, FloatConfig src_fp, DataType tgt_dtype) {
  LOG(FATAL) << "Not implemented yet";
}

PrimExpr DTypeConversion(PrimExpr src_value, DataType tgt_dtype, RoundingMode round_mode) {
  DataType src_dtype = src_value.dtype();
  CHECK_EQ(src_dtype.lanes(), tgt_dtype.lanes())
      << "The lanes for data type for source value must matches the target datatype.";
  auto is_floating_point = [](DataType dtype) {
    return dtype.is_float() || dtype.is_float8() || dtype.is_bfloat16();
  };
  auto is_integer = [](DataType dtype) { return dtype.is_int() || dtype.is_uint(); };
  if (is_floating_point(src_dtype) && is_floating_point(tgt_dtype)) {
    return FpToFp(src_value, tgt_dtype, round_mode);
  } else {
    if (is_integer(src_dtype) && is_floating_point(tgt_dtype)) {
      return IntToFp(src_value, FloatConfig::FromDataType(tgt_dtype));
    } else if (is_floating_point(src_dtype) && is_integer(src_dtype)) {
      return FpToInt(src_value, FloatConfig::FromDataType(src_dtype), tgt_dtype);
    } else {
      LOG(FATAL) << "Not Implemented yet";
    }
  }
}

}  // namespace tir
}  // namespace tvm
