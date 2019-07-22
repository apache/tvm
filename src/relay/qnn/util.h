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
#include <limits>

namespace tvm {
namespace relay {
namespace qnn {

static inline bool IsQNNDataType(const DataType& dtype) {
  return dtype == Int(8) || dtype == UInt(8)
      || dtype == Int(16) || dtype == UInt(16);
}

enum class QuantizeOpType {
  Quantize,
  Dequantize,
  Requantize
};

static inline bool IsValidOpInputType(const QuantizeOpType& op_type,
        const DataType& in_dtype) {
  switch (op_type) {
    case QuantizeOpType::Quantize:
      return in_dtype == Float(32) || IsQNNDataType(in_dtype);
    case QuantizeOpType::Dequantize:
      return IsQNNDataType(in_dtype);
    case QuantizeOpType::Requantize:
      return in_dtype.is_int() || in_dtype.is_uint();
    default:
      return false;
  }
}

static inline bool IsValidOpOutputType(const QuantizeOpType& op_type,
        const DataType& in_dtype) {
  switch (op_type) {
    case QuantizeOpType::Quantize:
      return IsQNNDataType(in_dtype);
    case QuantizeOpType::Dequantize:
      return in_dtype == Float(32);
    default:
      return false;
  }
}

static inline const int32_t GetQmin(const DataType& dtype) {
  CHECK_LE(dtype.bits(), 32)
      << "QNN ops support uint32/int32 or lower precision";
  if (dtype.is_int()) {
    auto* min_value = as_const_int(dtype.min());
    CHECK(min_value != nullptr);
    return static_cast<int32_t>(min_value[0]);
  } else if (dtype.is_uint()) {
    auto* min_value = as_const_uint(dtype.min());
    CHECK(min_value != nullptr);
    return static_cast<int32_t>(min_value[0]);
  }
  LOG(FATAL) << "Type not supported " << dtype;
  return -1;
}

static inline const int32_t GetQmax(const DataType& dtype) {
  CHECK_LE(dtype.bits(), 32)
      << "QNN ops support uint32/int32 or lower precision";
  if (dtype.is_int()) {
    auto* max_value = as_const_int(dtype.max());
    CHECK(max_value != nullptr);
    return static_cast<int32_t>(max_value[0]);
  } else if (dtype.is_uint()) {
    auto* max_value = as_const_uint(dtype.max());
    CHECK(max_value != nullptr);
    return static_cast<int32_t>(max_value[0]);
  }
  LOG(FATAL) << "Type not supported " << dtype;
  return -1;
}

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_UTIL_H_
