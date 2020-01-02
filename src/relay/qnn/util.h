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
 * \file src/relay/qnn/util.h
 * \brief Utility methods needs for quantized ops that can be shared
 */

#ifndef TVM_RELAY_QNN_UTIL_H_
#define TVM_RELAY_QNN_UTIL_H_

#include <tvm/expr.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/attrs.h>
#include <limits>
#include <string>
#include <utility>

namespace tvm {
namespace relay {
namespace qnn {

static inline Array<IndexExpr> get_shape(const Type& type) {
  auto input_tt = type.as<TensorTypeNode>();
  CHECK(input_tt != nullptr) << "Type information missing."
                             << " Please run infer_type pass.";
  return input_tt->shape;
}

static inline const int32_t GetQmin(const DataType& dtype) {
  CHECK_LE(dtype.bits(), 32)
      << "QNN ops support int32 or lower precision";
  if (dtype.is_int()) {
    auto* min_value = as_const_int(tvm::min_value(dtype));
    CHECK(min_value != nullptr);
    return static_cast<int32_t>(min_value[0]);
  } else if (dtype.is_uint()) {
    auto* min_value = as_const_uint(tvm::min_value(dtype));
    CHECK(min_value != nullptr);
    return static_cast<int32_t>(min_value[0]);
  } else {
    LOG(FATAL) << "Type not supported " << dtype;
    return -1;  // To hide the warning
  }
}

static inline const int32_t GetQmax(const DataType& dtype) {
  CHECK_LE(dtype.bits(), 32)
      << "QNN ops support int32 or lower precision";
  if (dtype.is_int()) {
    auto* max_value = as_const_int(tvm::max_value(dtype));
    CHECK(max_value != nullptr);
    return static_cast<int32_t>(max_value[0]);
  } else if (dtype.is_uint()) {
    auto* max_value = as_const_uint(tvm::max_value(dtype));
    CHECK(max_value != nullptr);
    return static_cast<int32_t>(max_value[0]);
  } else {
    LOG(FATAL) << "Type not supported " << dtype;
    return -1;  // To hide the warning
  }
}

Expr RequantizeLower(const Expr& input_tensor, const Expr& input_scale,
                     const Expr& input_zero_point, const Expr& output_scale,
                     const Expr& output_zero_point, const RequantizeAttrs* param,
                     const Array<IndexExpr>& input_shape, const DataType& out_dtype);

static inline Expr Requantize(const Expr& data, const Array<IndexExpr>& input_shape,
                              const Expr& input_scale, const Expr& input_zero_point,
                              const Expr& output_scale, const Expr& output_zero_point,
                              const DataType& out_dtype, const std::string& rounding = "UPWARD") {
  auto attrs = make_object<RequantizeAttrs>();
  attrs->rounding = std::move(rounding);
  attrs->out_dtype = std::move(out_dtype);
  return RequantizeLower(data, input_scale, input_zero_point, output_scale, output_zero_point,
                         attrs.operator->(), input_shape, out_dtype);
}

static inline int64_t get_const_int(const tvm::Expr& x) {
  auto* value_ptr = as_const_int(x);
  CHECK(value_ptr) << "Expr is not a constant int";
  return value_ptr[0];
}

/*
 * \brief Fixed point multiplication between integer tensor with floating point
 scalar.
 * \param tensor The quantized input tensor of dtype int64.
 * \param multiplier The scalar multiplier.
 * \param input_shape Shape of the input tensor.
 * \param rounding "UPWARD" or "TONEAREST". The rounding direction when the value
 is midway between" "two representable values.
 * \return The sequence of Relay ops for fixed point multiplication.

 * \note Original compuation is scale_fp32 * quantized_tensor.  To convert into
 *       integer computation, the multiplication with fp32 scalar can be
 *       replaced by multiplication with an int value and then right shifting
 *       the result. This approximates the floating point computation with a
 *       fixed point computation.
 *
 *       Computation of fixed point multiplication is consist of following
 steps:
 *       1) Multiply the fixed point multiplier with quantized tensor.
 *       2) Round the result.
 *       3) Right shift the result
 */
Expr FixedPointMultiply(Expr tensor, double multiplier, const Array<IndexExpr>& input_shape,
                        const std::string& rounding);

/*
 * \brief Checks whether an expr type is scalar of a given data type.
 * \param expr_type The type of expr to be checked.
 * \param dtype The expected dtype.
 * \return True if the type is a scalar of given dtype
 */
static inline bool IsScalarType(const Type& expr_type, const DataType& dtype) {
  const auto* scale = expr_type.as<TensorTypeNode>();
  CHECK_EQ(scale->shape.size(), 0);
  CHECK(scale->dtype == dtype) << "Expected " << dtype << " but got " << scale->dtype;
  return true;
}

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_UTIL_H_
