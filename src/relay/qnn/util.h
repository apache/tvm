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

inline bool IsInt8(const DataType& dtype) {
  return dtype == Int(8);
}

inline bool IsUint8(const DataType& dtype) {
  return dtype == UInt(8);
}

inline bool IsInt16(const DataType& dtype) {
  return dtype == Int(16);
}

inline bool IsUint16(const DataType& dtype) {
  return dtype == UInt(16);
}

inline bool IsInt32(const DataType& dtype) {
  return dtype == Int(32);
}

inline bool IsUint32(const DataType& dtype) {
  return dtype == UInt(32);
}

inline bool IsFloat32(const DataType& dtype) {
  return dtype == Float(32);
}

inline bool IsQuantizedType(const DataType& dtype) {
  return IsInt8(dtype) || IsUint8(dtype)
      || IsInt16(dtype) || IsUint16(dtype);
}

enum class QuantizeOpType : uint8_t {
  Quantize,
  Dequantize,
  Requantize,
  QuantizedDense
};
  inline bool IsQNNDataType(const DataType& dtype) {
  return dtype == Int(8) || dtype == UInt(8)
      || dtype == Int(16) || dtype == UInt(16);
}


inline bool IsValidOpInputType(const QuantizeOpType& op_type,
        const DataType& in_dtype) {
  switch (op_type) {
    case QuantizeOpType::Quantize:
      return IsFloat32(in_dtype);
    case QuantizeOpType::Dequantize:
    case QuantizeOpType::QuantizedDense:
      return IsQuantizedType(in_dtype);
    case QuantizeOpType ::Requantize:
      return IsInt16(in_dtype) || IsInt32(in_dtype);
    default:
      return false;
  }
}

inline bool IsValidOpOutputType(const QuantizeOpType& op_type,
        const DataType& out_dtype) {
  switch (op_type) {
    case QuantizeOpType::Quantize:
      return IsQNNDataType(out_dtype);
    case QuantizeOpType::Dequantize:
      return IsFloat32(out_dtype);
    case QuantizeOpType::QuantizedDense:
      return IsInt32(out_dtype) || IsInt16(out_dtype);
    default:
      return false;
  }
}

inline const int32_t GetQmin(const DataType& dtype) {
  if (dtype == Int(8)) {
    return std::numeric_limits<int8_t>::min();
  } else if (dtype == UInt(8)) {
    return std::numeric_limits<uint8_t>::min();
  } else if (dtype == Int(16)) {
    return std::numeric_limits<int16_t>::min();
  } else if (dtype == UInt(16)) {
    return std::numeric_limits<uint16_t>::min();
  } else if (dtype == Int(32)) {
    return std::numeric_limits<int32_t>::min();
  } else if (dtype == UInt(32)) {
    return std::numeric_limits<uint32_t>::min();
  }
  LOG(FATAL) << "Type not supported " << dtype;
  return -1;
}


inline const int32_t GetQmax(const DataType& dtype) {

  if (dtype == Int(8)) {
    return std::numeric_limits<int8_t>::max();
  } else if (dtype == UInt(8)) {
    return std::numeric_limits<uint8_t>::max();
  } else if (dtype == Int(16)) {
    return std::numeric_limits<int16_t>::max();
  } else if (dtype == UInt(16)) {
    return std::numeric_limits<uint16_t>::max();
  } else if (dtype == Int(32)) {
    return std::numeric_limits<int32_t>::max();
  } else if (dtype == UInt(32)) {
    return std::numeric_limits<uint32_t>::max();
  }
  LOG(FATAL) << "Type not supported " << dtype;
  return -1;
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_QNN_UTIL_H_
