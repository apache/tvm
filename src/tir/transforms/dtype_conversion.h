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
 * \file dtype_conversion.h
 * \brief Header file of data type conversion routines.
 */
#ifndef TVM_TIR_TRANSFORMS_DTYPE_CONVERSION_H_
#define TVM_TIR_TRANSFORMS_DTYPE_CONVERSION_H_

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

namespace tvm {

namespace tir {

/*!
 * \brief Rounding mode: https://en.wikipedia.org/wiki/Rounding
 */
enum class RoundingMode {
  // Round half to nearest even
  kHalfToEven = 0U,
  // Round down
  kDown = 1U,
  // Round up
  kUp = 2U,
  // Round towards zero
  kTowardsZero = 3U,
};

/*!
 * \brief Floating point internal representation.
 */
class FloatConfig {
 public:
  /*!
   * \brief Style of infinite number representation.
   */
  enum class InftyStyle {
    // Exponent all ones, mantissa all zeros
    kIEEE = 0U,
    // No representation of infinity
    kNone = 1U
  };
  /*!
   * \brief Style of NaN (not-a-number) representation.
   */
  enum class NaNStyle {
    // Exponent all ones, mantissa non zeros
    // - quiet NaN : 1XXXXX...
    // - signaling NaN : 0XXXXX...
    kIEEE = 0U,
    // No representation of infinity
    kNone = 1U,
    // Both exponent bits and mantissa bits are all ones.
    kAllOnes = 2U,
  };
  // The number of exponent bits.
  int exponent;
  // The number of mantissa (also know as fraction in IEEE format) bits.
  int mantissa;
  // The exponent bias in IEEE format.
  int bias;
  // The representation of infinity.
  InftyStyle infty_style;
  // The representation of NaN (Not a Number).
  NaNStyle nan_style;

  FloatConfig(int exponent, int mantissa, int bias, InftyStyle infty_style, NaNStyle nan_style)
      : exponent(exponent),
        mantissa(mantissa),
        bias(bias),
        infty_style(infty_style),
        nan_style(nan_style) {}

  inline int bits() const { return mantissa + exponent + 1; }

  /*!
   * \brief Create float config from data type.
   * \param dtype The data type, must be a floating point.
   * \return The FloatConfig class containing internal floating point representation.
   */
  static FloatConfig FromDataType(DataType dtype) {
    CHECK(dtype.is_float() || dtype.is_bfloat16() || dtype.is_float8())
        << "FloatConfig is only applicable to floating point data types, got " << dtype
        << " instead.";
    if (dtype.is_float()) {
      // IEEE 754 binary formats
      // Reference: https://en.wikipedia.org/wiki/Floating-point_arithmetic
      switch (dtype.bits()) {
        case 16:
          return FloatConfig(5, 10, 15, InftyStyle::kIEEE, NaNStyle::kIEEE);
        case 32:
          return FloatConfig(8, 23, 127, InftyStyle::kIEEE, NaNStyle::kIEEE);
        default:
          // float64
          return FloatConfig(11, 52, 1023, InftyStyle::kIEEE, NaNStyle::kIEEE);
      }
    } else if (dtype.is_bfloat16()) {
      // bfloat16,
      return FloatConfig(8, 7, 127, InftyStyle::kIEEE, NaNStyle::kIEEE);
    } else {  // float8
      // NVIDIA/Arm/Intel's FP8 formats for Deep Learning
      // Reference: https://arxiv.org/abs/2209.05433
      switch (dtype.code()) {
        case DataType::kE4M3Float:
          // E4M3 format, not consistent with IEEE-754
          return FloatConfig(4, 3, 7, InftyStyle::kNone, NaNStyle::kAllOnes);
        default:
          // E5M2 format, consistent with IEEE-754
          return FloatConfig(5, 2, 15, InftyStyle::kIEEE, NaNStyle::kIEEE);
      }
    }
  }
};

/*!
 * \brief Reinterpret value as unsigned integer with equal number of bits.
 * \param value The value to interpret.
 * \return The reinterpreted uint value.
 */
PrimExpr ReinterpretAsUInt(PrimExpr value);

/*!
 * \brief Get the unsigned integer data type used as storage when the specified dtype is not
 *   supported natively.
 * \param dtype The data type.
 * \return The uint data type, the number of bits is
 *   the same as input dtype.
 */
DataType GetStorageUIntDType(DataType dtype);

/*!
 * \brief Conversion routine from value stored in one floating point data type to another floating
 *   point data type.
 * \param src_value The floating point value to be converted.
 * \param tgt_dtype The target floating point data type.
 * \param round_mode The rounding mode to use, defaults to kHalfToEven.
 * \return The converted value in target floating point data type.
 * \note Used when there is no native data type conversion implementation.
 */
PrimExpr DTypeConversion(PrimExpr src_value, DataType tgt_dtype,
                         RoundingMode round_mode = RoundingMode::kHalfToEven);

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORMS_DTYPE_CONVERSION_H_
