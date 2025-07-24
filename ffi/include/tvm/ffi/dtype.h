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
 * \file tvm/ffi/dtype.h
 * \brief Data type handling.
 */
#ifndef TVM_FFI_DTYPE_H_
#define TVM_FFI_DTYPE_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/type_traits.h>

#include <string>

namespace tvm {
namespace ffi {
/*!
 * \brief Extension code beyond the DLDataType.
 *
 * This class is always consistent with the DLPack.
 *
 * TOTO(tvm-team): update to latest DLPack types.
 */
enum DLExtDataTypeCode { kDLExtCustomBegin = 129 };

namespace details {

/*
 * \brief Convert a DLDataTypeCode to a string.
 * \param os The output stream.
 * \param type_code The DLDataTypeCode to convert.
 */
inline const char* DLDataTypeCodeAsCStr(DLDataTypeCode type_code) {  // NOLINT(*)
  switch (static_cast<int>(type_code)) {
    case kDLInt: {
      return "int";
    }
    case kDLUInt: {
      return "uint";
    }
    case kDLFloat: {
      return "float";
    }
    case kDLOpaqueHandle: {
      return "handle";
    }
    case kDLBfloat: {
      return "bfloat";
    }
    case kDLFloat8_e3m4: {
      return "float8_e3m4";
    }
    case kDLFloat8_e4m3: {
      return "float8_e4m3";
    }
    case kDLFloat8_e4m3b11fnuz: {
      return "float8_e4m3b11fnuz";
    }
    case kDLFloat8_e4m3fn: {
      return "float8_e4m3fn";
    }
    case kDLFloat8_e4m3fnuz: {
      return "float8_e4m3fnuz";
    }
    case kDLFloat8_e5m2: {
      return "float8_e5m2";
    }
    case kDLFloat8_e5m2fnuz: {
      return "float8_e5m2fnuz";
    }
    case kDLFloat8_e8m0fnu: {
      return "float8_e8m0fnu";
    }
    case kDLFloat6_e2m3fn: {
      return "float6_e2m3fn";
    }
    case kDLFloat6_e3m2fn: {
      return "float6_e3m2fn";
    }
    case kDLFloat4_e2m1fn: {
      return "float4_e2m1fn";
    }
    default: {
      if (static_cast<int>(type_code) >= static_cast<int>(DLExtDataTypeCode::kDLExtCustomBegin)) {
        return "custom";
      } else {
        TVM_FFI_THROW(ValueError) << "DLDataType contains unknown type_code="
                                  << static_cast<int>(type_code);
      }
      TVM_FFI_UNREACHABLE();
    }
  }
}
}  // namespace details

inline DLDataType StringToDLDataType(const String& str) {
  DLDataType out;
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIDataTypeFromString(str.get(), &out));
  return out;
}

inline String DLDataTypeToString(DLDataType dtype) {
  TVMFFIObjectHandle out;
  TVM_FFI_CHECK_SAFE_CALL(TVMFFIDataTypeToString(&dtype, &out));
  return String(details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<TVMFFIObject*>(out)));
}

// DLDataType
template <>
struct TypeTraits<DLDataType> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDataType;

  TVM_FFI_INLINE static void CopyToAnyView(const DLDataType& src, TVMFFIAny* result) {
    // clear padding part to ensure the equality check can always check the v_uint64 part
    result->v_uint64 = 0;
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  TVM_FFI_INLINE static void MoveToAny(DLDataType src, TVMFFIAny* result) {
    // clear padding part to ensure the equality check can always check the v_uint64 part
    result->v_uint64 = 0;
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDataType;
  }

  TVM_FFI_INLINE static DLDataType CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return src->v_dtype;
  }

  TVM_FFI_INLINE static std::optional<DLDataType> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDataType) {
      return src->v_dtype;
    }
    // enable string to dtype auto conversion
    if (auto opt_str = TypeTraits<std::string>::TryCastFromAnyView(src)) {
      return StringToDLDataType(*opt_str);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return ffi::StaticTypeKey::kTVMFFIDataType; }
};
}  // namespace ffi
}  // namespace tvm

// define DLDataType comparison and printing in root namespace
inline std::ostream& operator<<(std::ostream& os, DLDataType dtype) {  // NOLINT(*)
  return os << tvm::ffi::DLDataTypeToString(dtype);
}

inline bool operator==(const DLDataType& lhs, const DLDataType& rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

inline bool operator!=(const DLDataType& lhs, const DLDataType& rhs) { return !(lhs == rhs); }
#endif  // TVM_FFI_DTYPE_H_
