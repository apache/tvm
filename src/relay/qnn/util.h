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

static inline const int32_t GetQmin(const DataType& dtype) {
  CHECK_LE(dtype.bits(), 32)
      << "QNN ops support int32 or lower precision";
  if (dtype.is_int()) {
    auto* min_value = as_const_int(dtype.min());
    CHECK(min_value != nullptr);
    return static_cast<int32_t>(min_value[0]);
  } else if (dtype.is_uint()) {
    auto* min_value = as_const_uint(dtype.min());
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
    auto* max_value = as_const_int(dtype.max());
    CHECK(max_value != nullptr);
    return static_cast<int32_t>(max_value[0]);
  } else if (dtype.is_uint()) {
    auto* max_value = as_const_uint(dtype.max());
    CHECK(max_value != nullptr);
    return static_cast<int32_t>(max_value[0]);
  } else {
    LOG(FATAL) << "Type not supported " << dtype;
    return -1;  // To hide the warning
  }
}

Expr RequantizeLower(const Expr& input_tensor, const RequantizeAttrs* param,
                     const Array<IndexExpr>& input_shape, const DataType& out_dtype);

static inline Expr Requantize(const Expr& data, const Array<IndexExpr>& input_shape,
                              double input_scale, int32_t input_zero_point, double output_scale,
                              int32_t output_zero_point, const DataType& out_dtype,
                              const std::string& rounding = "TONEAREST") {
  auto attrs = make_node<RequantizeAttrs>();
  attrs->input_scale = std::move(input_scale);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->output_scale = std::move(output_scale);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->rounding = std::move(rounding);
  attrs->out_dtype = std::move(out_dtype);
  return RequantizeLower(data, attrs.operator->(), input_shape, out_dtype);
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
Expr FixedPointMuliply(Expr tensor, double multiplier,
                       const Array<IndexExpr>& input_shape,
                       const std::string& rounding);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_UTIL_H_
