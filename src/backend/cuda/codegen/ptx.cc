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
 * \file ptx.cc
 */

#include <tvm/ffi/error.h>
#include <tvm/ffi/reflection/registry.h>

#include <string>

namespace tvm {
namespace codegen {

// PTX related data structures and functions.
namespace ptx {

/*!
 * \brief PTX data type.
 * \note
 * PTX fundamental data types:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#fundamental-types
 * PTX matrix data types:
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-data-types
 */
enum class DataType : int {
  kInt4 = 0,
  kUInt4 = 1,
  kInt8 = 2,
  kUInt8 = 3,
  kInt16 = 4,
  kUInt16 = 5,
  kInt32 = 6,
  kUInt32 = 7,
  kInt64 = 8,
  kUInt64 = 9,
  kFloat4_e2m1fn = 10,
  kFloat6_e2m3fn = 11,
  kFloat6_e3m2fn = 12,
  kFloat8_e4m3fn = 13,
  kFloat8_e4m3fnuz = 14,
  kFloat8_e5m2 = 15,
  kFloat8_e8m0fnu = 16,
  kFloat16 = 17,
  kBFloat16 = 18,
  kFloat16x2 = 19,
  kFloat32 = 20,
  kTensorFloat32 = 21,
  kFloat64 = 22,
  kBit1 = 23,
  kBit8 = 24,
  kBit16 = 25,
  kBit32 = 26,
  kBit64 = 27,
};

static const char* dtype_str[] = {".s4",    ".u4",   ".s8",    ".u8",   ".s16",  ".u16",   ".s32",
                                  ".u32",   ".s64",  ".u64",   ".e2m1", ".e2m3", ".e3m2",  ".e4m3",
                                  ".ue4m3", ".e5m2", ".ue8m0", ".f16",  ".bf16", ".f16x2", ".f32",
                                  ".tf32",  ".f64",  ".b1",    ".b8",   ".b16",  ".b32",   ".b64"};

/*!
 * \brief Create PTX data type from string.
 */
inline DataType DTypeFromString(const std::string str) {
  if (str == "int4" || str == ".s4") {
    return DataType::kInt4;
  } else if (str == "uint4" || str == ".u4") {
    return DataType::kUInt4;
  } else if (str == "int8" || str == ".s8") {
    return DataType::kInt8;
  } else if (str == "uint8" || str == ".u8") {
    return DataType::kUInt8;
  } else if (str == "int16" || str == ".s16") {
    return DataType::kInt16;
  } else if (str == "uint16" || str == ".u16") {
    return DataType::kUInt16;
  } else if (str == "int32" || str == ".s32") {
    return DataType::kInt32;
  } else if (str == "uint32" || str == ".u32") {
    return DataType::kUInt32;
  } else if (str == "int64" || str == ".s64") {
    return DataType::kInt64;
  } else if (str == "uint64" || str == ".u64") {
    return DataType::kUInt64;
  } else if (str == "e2m1" || str == ".e2m1" || str == "float4_e2m1fn") {
    return DataType::kFloat4_e2m1fn;
  } else if (str == "e2m3" || str == ".e2m3" || str == "float6_e2m3fn") {
    return DataType::kFloat6_e2m3fn;
  } else if (str == "e3m2" || str == ".e3m2" || str == "float6_e3m2fn") {
    return DataType::kFloat6_e3m2fn;
  } else if (str == "e4m3" || str == ".e4m3" || str == "float8_e4m3fn") {
    return DataType::kFloat8_e4m3fn;
  } else if (str == "float8_e4m3fnuz" || str == "float8_e4m3b11fnuz") {
    return DataType::kFloat8_e4m3fnuz;
  } else if (str == "e5m2" || str == ".e5m2" || str == "float8_e5m2" || str == "float8_e5m2fn" ||
             str == "float8_e5m2fnuz") {
    return DataType::kFloat8_e5m2;
  } else if (str == "ue8m0" || str == ".ue8m0" || str == "float8_e8m0fnu") {
    return DataType::kFloat8_e8m0fnu;
  } else if (str == "float16" || str == "fp16" || str == ".f16") {
    return DataType::kFloat16;
  } else if (str == "bfloat16" || str == "bf16") {
    return DataType::kBFloat16;
  } else if (str == ".f16x2") {
    return DataType::kFloat16x2;
  } else if (str == "float32" || str == "fp32" || str == ".f32") {
    return DataType::kFloat32;
  } else if (str == "tf32") {
    return DataType::kTensorFloat32;
  } else if (str == "float64" || str == "fp64" || str == ".f64") {
    return DataType::kFloat64;
  } else if (str == "int1" || str == ".b1") {
    return DataType::kBit1;
  } else if (str == ".b8") {
    return DataType::kBit8;
  } else if (str == ".b16") {
    return DataType::kBit16;
  } else if (str == ".b32") {
    return DataType::kBit32;
  } else if (str == ".b64") {
    return DataType::kBit64;
  } else {
    TVM_FFI_THROW(InternalError) << "Unrecognized PTX data type " << str;
  }
}

/*!
 * \brief Get the string representation of given PTX data type.
 */
inline std::string DTypeToString(DataType dtype) { return dtype_str[static_cast<int>(dtype)]; }

void RegisterCudaPTXHelpers() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.intrinsics.cuda.PTXDTypeFromString",
           [](const std::string& str) -> int { return static_cast<int>(DTypeFromString(str)); })
      .def("tirx.intrinsics.cuda.PTXDTypeToString", [](const int dtype) -> std::string {
        return DTypeToString(static_cast<DataType>(dtype));
      });
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaPTXHelpers(); }

}  // namespace ptx
}  // namespace codegen
}  // namespace tvm
