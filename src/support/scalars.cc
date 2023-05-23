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
 * \file src/support/scalars.cc
 * \brief Helpers for converting between scalars in native, text, TIR immediate and NDArray forms.
 */

#include "./scalars.h"

#include "tvm/relay/expr.h"
#include "tvm/runtime/builtin_fp16.h"

namespace tvm {
namespace support {

/*! \brief The standard scalar dtypes. */
static const DataType kInt16 = DataType::Int(16);
static const DataType kInt32 = DataType::Int(32);
static const DataType kInt64 = DataType::Int(64);
static const DataType kFloat16 = DataType::Float(16);
static const DataType kFloat32 = DataType::Float(32);
static const DataType kFloat64 = DataType::Float(64);
static const DataType kBool = DataType::Bool();

bool IsSimpleScalarDtype(DataType dtype) {
  return dtype == kInt16 || dtype == kInt32 || dtype == kInt64 || dtype == kFloat16 ||
         dtype == kFloat32 || dtype == kFloat64 || dtype == kBool;
}

bool IsSimpleScalar(const relay::ConstantNode* constant_node) {
  return constant_node->is_scalar() && IsSimpleScalarDtype(DataType(constant_node->data->dtype));
}

runtime::NDArray IntImmToNDArray(const IntImm& int_imm) {
  DLDevice dev = {DLDeviceType::kDLCPU, 0};
  auto data = runtime::NDArray::Empty({}, int_imm->dtype, dev);
  if (int_imm.dtype() == kInt16) {
    auto* array = reinterpret_cast<int16_t*>(data->data);
    array[0] = static_cast<int16_t>(int_imm->value);
  } else if (int_imm.dtype() == kInt32) {
    auto* array = reinterpret_cast<int32_t*>(data->data);
    array[0] = static_cast<int32_t>(int_imm->value);
  } else if (int_imm.dtype() == kInt64) {
    auto* array = reinterpret_cast<int64_t*>(data->data);
    array[0] = int_imm->value;
  } else {
    LOG(FATAL) << "Unrecognized numeric literal dtype: " << DLDataType2String(int_imm.dtype());
  }
  return data;
}

runtime::NDArray FloatImmToNDArray(const FloatImm& float_imm) {
  DLDevice dev = {DLDeviceType::kDLCPU, 0};
  auto data = runtime::NDArray::Empty({}, float_imm->dtype, dev);
  if (float_imm.dtype() == kFloat16) {
    auto* array = reinterpret_cast<uint16_t*>(data->data);
    array[0] = __gnu_f2h_ieee(static_cast<float>(float_imm->value));
  } else if (float_imm.dtype() == kFloat32) {
    auto* array = reinterpret_cast<float*>(data->data);
    array[0] = static_cast<float>(float_imm->value);
  } else if (float_imm.dtype() == kFloat64) {
    auto* array = reinterpret_cast<double*>(data->data);
    array[0] = float_imm->value;
  } else {
    LOG(FATAL) << "Unrecognized numeric literal dtype: " << DLDataType2String(float_imm.dtype());
  }
  return data;
}

runtime::NDArray BoolToNDArray(bool value) {
  DLDevice dev = {DLDeviceType::kDLCPU, 0};
  auto data = runtime::NDArray::Empty({}, kBool, dev);
  auto array = reinterpret_cast<bool*>(data->data);
  array[0] = value;
  return data;
}

std::string NDArrayScalarToString(const runtime::NDArray& data) {
  std::ostringstream os;
  DataType dtype(data->dtype);
  ICHECK_EQ(data->device.device_type, kDLCPU) << "Scalars must reside on the CPU to be printed";
  if (dtype == kInt16) {
    auto value = static_cast<const int16_t*>(data->data)[0];
    os << value << "i16";
  } else if (dtype == kInt32) {
    auto value = static_cast<const int32_t*>(data->data)[0];
    os << value;
  } else if (dtype == kInt64) {
    auto value = static_cast<const int64_t*>(data->data)[0];
    os << value << "i64";
  } else if (dtype == kFloat16) {
    auto value = __gnu_h2f_ieee(static_cast<const uint16_t*>(data->data)[0]);
    os << value << "f16";
  } else if (dtype == kFloat32) {
    auto value = static_cast<const float*>(data->data)[0];
    os << value << "f";
  } else if (dtype == kFloat64) {
    auto value = static_cast<const double*>(data->data)[0];
    os << value << "f64";
  } else if (dtype == kBool) {
    auto value = static_cast<const uint8_t*>(data->data)[0];
    os << (value ? "True" : "False");
  } else {
    LOG(FATAL) << "Unrecognized NDArray scalar dtype: " << DLDataType2String(dtype);
  }
  return os.str();
}

std::string IntImmToString(const IntImm& int_imm) {
  std::ostringstream os;
  if (int_imm->dtype == kInt16) {
    os << int_imm->value << "i16";
  } else if (int_imm->dtype == kInt32) {
    os << int_imm->value;
  } else if (int_imm->dtype == kInt64) {
    os << int_imm->value << "i64";
  } else if (int_imm->dtype == kBool) {
    os << (int_imm->value ? "True" : "False");
  } else {
    LOG(FATAL) << "Unrecognised IntImm dtype: " << DLDataType2String(int_imm->dtype);
  }
  return os.str();
}

std::string FloatImmToString(const FloatImm& float_imm) {
  std::ostringstream os;
  if (float_imm->dtype == kFloat16) {
    os << float_imm->value << "f16";
  } else if (float_imm->dtype == kFloat32) {
    os << float_imm->value << "f";
  } else if (float_imm->dtype == kFloat64) {
    os << float_imm->value << "f64";
  } else {
    LOG(FATAL) << "Unrecognised FloatImm dtype: " << DLDataType2String(float_imm->dtype);
  }
  return os.str();
}

IntImm ValueToIntImm(int64_t value, int width) {
  if (width == 16) {
    if (value < std::numeric_limits<int16_t>::min() ||
        value > std::numeric_limits<int16_t>::max()) {
      return {};
    }
    return IntImm(kInt16, value);
  } else if (width == 32) {
    if (value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
      return {};
    }
    return IntImm(kInt32, value);
  } else if (width == 64) {
    return IntImm(kInt64, value);
  } else {
    LOG(FATAL) << "Unrecognized int scalar width: " << width;
  }
}

FloatImm ValueToFloatImm(double value, int width) {
  if (width == 16) {
    if (!std::isinf(value) && (value < -kMaxFloat16 || value > kMaxFloat16)) {
      return {};
    }
    return FloatImm(kFloat16, value);
  } else if (width == 32) {
    if (!std::isinf(value) &&
        (value < -std::numeric_limits<float>::max() || value > std::numeric_limits<float>::max())) {
      return {};
    }
    return FloatImm(kFloat32, value);
  } else if (width == 64) {
    return FloatImm(kFloat64, value);
  } else {
    LOG(FATAL) << "Unrecognized float scalar width: " << width;
  }
}

}  // namespace support
}  // namespace tvm
