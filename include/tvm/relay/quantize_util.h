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
 * \file nnvm/compiler/quantize_util.h
 * \brief Utility methods needs for quantized ops that can be shared
 */

#ifndef TVM_QUANTIZE_UTIL_H
#define TVM_QUANTIZE_UTIL_H

#include <tvm/expr.h>
#include "./base.h"

namespace tvm {
namespace relay {

inline bool is_Int8(const DataType& dtype) {
  return dtype == Int(8);
}

inline bool is_UInt8(const DataType& dtype) {
  return dtype == UInt(8);
}


inline bool is_Int16(const DataType& dtype) {
  return dtype == Int(16);
}

inline bool is_UInt16(const DataType& dtype) {
  return dtype == UInt(16);
}

inline bool is_Int32(const DataType& dtype) {
  return dtype == Int(32);
}

inline bool is_UInt32(const DataType& dtype) {
  return dtype == UInt(32);
}



inline bool is_Float32(const DataType& dtype) {
  return dtype == Float(32);
}

inline bool is_quantized_type(const DataType& dtype) {
  return is_Int8(dtype) || is_UInt8(dtype)
      || is_Int16(dtype) || is_UInt16(dtype);
}

enum class QuantizeOpType : uint8_t {
  Quantize_Requantize,
  Dequantize,
  Requantize
};

inline bool is_valid_quantized_op_input_type(const QuantizeOpType &op_type, const DataType &in_dtype) {
  switch(op_type) {
    case QuantizeOpType::Quantize_Requantize:
      return is_Float32(in_dtype) || is_quantized_type(in_dtype);
    case QuantizeOpType ::Dequantize:
      return is_quantized_type(in_dtype);
    case QuantizeOpType ::Requantize:
      return is_Int16(in_dtype) || is_Int32(in_dtype);
    default:
      return false;
  }
}

inline bool is_valid_quantized_op_output_type(const QuantizeOpType &op_type, const DataType &in_dtype) {
  switch(op_type) {
    case QuantizeOpType::Quantize_Requantize:
      return is_quantized_type(in_dtype);
    case QuantizeOpType::Dequantize:
      return is_Float32(in_dtype);
    default:
      return false;
  }
}

inline const int32_t get_qmin(const DataType&  dtype) {
  if (is_Int8(dtype)) {
    return std::numeric_limits<int8_t>::min();
  } else if (is_UInt8(dtype)) {
    return std::numeric_limits<uint8_t>::min();
  } else if (is_Int16(dtype)) {
    return std::numeric_limits<int16_t>::min();
  } else if (is_UInt16(dtype)) {
    return std::numeric_limits<uint16_t>::min();
  } else if (is_Int32(dtype)) {
    return std::numeric_limits<int32_t>::min();
  } else if (is_UInt32(dtype)) {
    return std::numeric_limits<uint32_t>::min();
  }
  LOG(FATAL) << "Type not supported\n";
  return -1;
}


inline const int32_t get_qmax(const DataType&  dtype) {
  if (is_Int8(dtype)) {
    return std::numeric_limits<int8_t>::max();
  } else if (is_UInt8(dtype)) {
    return std::numeric_limits<uint8_t>::max();
  } else if (is_Int16(dtype)) {
    return std::numeric_limits<int16_t>::max();
  } else if (is_UInt16(dtype)) {
    return std::numeric_limits<uint16_t>::max();
  } else if (is_Int32(dtype)) {
    return std::numeric_limits<int32_t>::max();
  } else if (is_UInt32(dtype)) {
    return std::numeric_limits<uint32_t>::max();
  }
  LOG(FATAL) << "Type not supported\n";
  return -1;
}

} // namespace relay
} // namespace tvm
#endif //TVM_QUANTIZE_UTIL_H
