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

inline bool is_Float32(const DataType& dtype) {
  return dtype == Float(32);
}

inline bool is_qauntized_type(const DataType& dtype) {
  return is_Int8(dtype) || is_UInt8(dtype);
}

enum class QuantizeOpType : uint8_t {
  Quantize_Requantize,
  Dequantize
};

inline bool is_valid_quantized_op_input_type(const QuantizeOpType &op_type, const DataType &in_dtype) {
  switch(op_type) {
    case QuantizeOpType::Quantize_Requantize:
      return is_Float32(in_dtype) || is_qauntized_type(in_dtype);
    case QuantizeOpType ::Dequantize:
      return is_qauntized_type(in_dtype);
    default:
      return false;
  }
}

inline bool is_valid_quantized_op_output_type(const QuantizeOpType &op_type, const DataType &in_dtype) {
  switch(op_type) {
    case QuantizeOpType::Quantize_Requantize:
      return is_qauntized_type(in_dtype);
    case QuantizeOpType::Dequantize:
      return is_Float32(in_dtype);
    default:
      return false;
  }
}

inline const int32_t get_qmin(const DataType&  dtype) {
  CHECK(is_qauntized_type(dtype)) << "Expected quantized data type [int8, uint8] but was " << dtype;
  if(is_Int8(dtype)) {
    return std::numeric_limits<int8_t>::min();
  } else {
    return std::numeric_limits<uint8_t>::min();
  }
}


inline const int32_t get_qmax(const DataType&  dtype) {
  CHECK(is_qauntized_type(dtype)) << "Expected quantized data type [int8, uint8] but was " << dtype;
  if(dtype == Int(8)) {
    return std::numeric_limits<int8_t>::max();
  } else {
    return std::numeric_limits<uint8_t>::max();
  }
}

} // namespace relay
} // namespace tvm
#endif //TVM_QUANTIZE_UTIL_H