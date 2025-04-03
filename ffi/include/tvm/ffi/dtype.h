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
 *
 * This file contains convenient methods for holding
 */
#ifndef TVM_FFI_DTYPE_H_
#define TVM_FFI_DTYPE_H_

#include <dlpack/dlpack.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/type_traits.h>

namespace tvm {
namespace ffi {
/*!
 * \brief Extension code beyond the DLDataType.
 *
 * This class is always consistent with the DLPack.
 *
 * TOTO(tvm-team): update to latest DLPack types.
 */
enum DLExtDataTypeCode {
  kDLExtFloat8_e4m3fn = 6,
  kDLExtFloat8_e5m2 = 7,
  kDLExtFloat4_e2m1fn = 8,
  kDLExtCustomBegin = 129
};

namespace details {
/*!
 * \brief Get the custom type name for a given type code.
 */
inline String DLDataTypeCodeGetCustomTypeName(DLDataTypeCode type_code) {
  static Function fget_custom_type_name = Function::GetGlobalRequired("dtype.get_custom_type_name");
  return fget_custom_type_name(static_cast<int>(type_code)).operator String();
}

/*!
 * \brief Get the custom type name for a given type code.
 * \param str The string to parse.
 * \param scan The scan pointer.
 * \return The custom type name.
 */
inline int ParseCustomDataTypeCode(const std::string& str, const char** scan) {
  TVM_FFI_ICHECK(str.substr(0, 6) == "custom") << "Not a valid custom datatype string";
  auto tmp = str.c_str();
  TVM_FFI_ICHECK(str.c_str() == tmp);
  *scan = str.c_str() + 6;
  TVM_FFI_ICHECK(str.c_str() == tmp);
  if (**scan != '[')
    TVM_FFI_THROW(ValueError) << "expected opening brace after 'custom' type in" << str;
  TVM_FFI_ICHECK(str.c_str() == tmp);
  *scan += 1;
  TVM_FFI_ICHECK(str.c_str() == tmp);
  size_t custom_name_len = 0;
  TVM_FFI_ICHECK(str.c_str() == tmp);
  while (*scan + custom_name_len <= str.c_str() + str.length() &&
         *(*scan + custom_name_len) != ']') {
    ++custom_name_len;
  }
  TVM_FFI_ICHECK(str.c_str() == tmp);
  if (*(*scan + custom_name_len) != ']') {
    TVM_FFI_THROW(ValueError) << "expected closing brace after 'custom' type in" << str;
  }
  TVM_FFI_ICHECK(str.c_str() == tmp);
  *scan += custom_name_len + 1;
  TVM_FFI_ICHECK(str.c_str() == tmp);
  auto type_name = str.substr(7, custom_name_len);
  TVM_FFI_ICHECK(str.c_str() == tmp);
  static Function fget_custom_type_code = Function::GetGlobalRequired("dtype.get_custom_type_code");
  return fget_custom_type_code(type_name);
}
/*
 * \brief Convert a DLDataTypeCode to a string.
 * \param os The output stream.
 * \param type_code The DLDataTypeCode to convert.
 */
inline void PrintDLDataTypeCodeAsStr(std::ostream& os, DLDataTypeCode type_code) {  // NOLINT(*)
  switch (static_cast<int>(type_code)) {
    case kDLInt: {
      os << "int";
      break;
    }
    case kDLUInt: {
      os << "uint";
      break;
    }
    case kDLFloat: {
      os << "float";
      break;
    }
    case kDLOpaqueHandle: {
      os << "handle";
      break;
    }
    case kDLBfloat: {
      os << "bfloat";
      break;
    }
    case kDLExtFloat8_e4m3fn: {
      os << "float8_e4m3fn";
      break;
    }
    case kDLExtFloat8_e5m2: {
      os << "float8_e5m2";
      break;
    }
    case kDLExtFloat4_e2m1fn: {
      os << "float4_e2m1fn";
      break;
    }
    default: {
      if (type_code >= static_cast<int>(DLExtDataTypeCode::kDLExtCustomBegin)) {
        os << "custom[" << details::DLDataTypeCodeGetCustomTypeName(type_code) << "]";
      } else {
        TVM_FFI_THROW(ValueError) << "DLDataType contains unknown type_code="
                                  << static_cast<int>(type_code);
      }
      TVM_FFI_UNREACHABLE();
    }
  }
}
}  // namespace details

/*!
 *  \brief Printer function for DLDataType.
 *  \param os The output stream.
 *  \param dtype The DLDataType to print.
 *  \return The output stream.
 */
inline std::ostream& operator<<(std::ostream& os, DLDataType dtype) {  // NOLINT(*)
  if (dtype.bits == 1 && dtype.lanes == 1 && dtype.code == kDLUInt) {
    os << "bool";
    return os;
  }
  // specially handle void
  if (dtype.code == kDLOpaqueHandle && dtype.lanes == 0 && dtype.bits == 0) {
    return os << "void";
  }
  details::PrintDLDataTypeCodeAsStr(os, static_cast<DLDataTypeCode>(dtype.code));
  if (dtype.code == kDLOpaqueHandle) return os;
  int16_t lanes = static_cast<int16_t>(dtype.lanes);
  if (dtype.code != kDLExtFloat8_e4m3fn && dtype.code != kDLExtFloat8_e5m2 &&
      dtype.code != kDLExtFloat4_e2m1fn) {
    os << static_cast<int>(dtype.bits);
  }
  if (lanes > 1) {
    os << 'x' << lanes;
  } else if (lanes < -1) {
    os << "xvscalex" << -lanes;
  }
  return os;
}

/*!
 * \brief convert a DLDataType to string.
 * \param dtype The DLDataType to convert.
 * \return The corresponding DLDataType in string.
 */
inline std::string DLDataTypeToString(DLDataType dtype) {
  std::ostringstream oss;
  oss << dtype;
  return oss.str();
}

/*!
 * \brief Parse a string to a DLDataType.
 * \param str The string to convert.
 * \return The corresponding DLDataType.
 */
inline DLDataType StringToDLDataType(const std::string& str) {
  DLDataType dtype;
  // handle void type
  if (str.length() == 0 || str == "void") {
    dtype.code = kDLOpaqueHandle;
    dtype.bits = 0;
    dtype.lanes = 0;
    return dtype;
  }
  // set the default values;
  dtype.bits = 32;
  dtype.lanes = 1;
  const char* scan;

  auto parse_float = [&](const std::string& str, int offset, int code, int bits) {
    dtype.code = code;
    dtype.bits = bits;
    scan = str.c_str() + offset;
    char* endpt = nullptr;
    if (*scan == 'x') {
      dtype.lanes = static_cast<uint16_t>(strtoul(scan + 1, &endpt, 10));
      scan = endpt;
    }
    if (scan != str.c_str() + str.length()) {
      TVM_FFI_THROW(ValueError) << "unknown dtype " << str;
    }
    return dtype;
  };

  if (str.substr(0, 3) == "int") {
    dtype.code = kDLInt;
    scan = str.c_str() + 3;
  } else if (str.substr(0, 4) == "uint") {
    dtype.code = kDLUInt;
    scan = str.c_str() + 4;
  } else if (str.substr(0, 13) == "float4_e2m1fn") {
    return parse_float(str, 13, DLExtDataTypeCode::kDLExtFloat4_e2m1fn, 4);
  } else if (str.substr(0, 13) == "float8_e4m3fn") {
    return parse_float(str, 13, DLExtDataTypeCode::kDLExtFloat8_e4m3fn, 8);
  } else if (str.substr(0, 11) == "float8_e5m2") {
    return parse_float(str, 11, DLExtDataTypeCode::kDLExtFloat8_e5m2, 8);
  } else if (str.substr(0, 5) == "float") {
    dtype.code = kDLFloat;
    scan = str.c_str() + 5;
  } else if (str.substr(0, 6) == "handle") {
    dtype.code = kDLOpaqueHandle;
    dtype.bits = 64;  // handle uses 64 bit by default.
    scan = str.c_str() + 6;
  } else if (str == "bool") {
    dtype.code = kDLUInt;
    dtype.bits = 1;
    dtype.lanes = 1;
    return dtype;
  } else if (str.substr(0, 6) == "bfloat") {
    dtype.code = kDLBfloat;
    dtype.bits = 16;
    scan = str.c_str() + 6;
  } else if (str.substr(0, 6) == "custom") {
    dtype.code = details::ParseCustomDataTypeCode(str, &scan);
  } else {
    scan = str.c_str();
    TVM_FFI_THROW(ValueError) << "unknown dtype " << str;
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0) dtype.bits = bits;
  int scalable_multiplier = 1;
  if (strncmp(xdelim, "xvscale", 7) == 0) {
    scalable_multiplier = -1;
    xdelim += 7;
  }
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    dtype.lanes = static_cast<uint16_t>(scalable_multiplier * strtoul(xdelim + 1, &endpt, 10));
  }
  if (endpt != str.c_str() + str.length()) {
    TVM_FFI_THROW(ValueError) << "unknown dtype " << str;
  }
  return dtype;
}
// DLDataType
template <>
struct TypeTraits<DLDataType> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDataType;

  static TVM_FFI_INLINE void CopyToAnyView(const DLDataType& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  static TVM_FFI_INLINE void MoveToAny(DLDataType src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  static TVM_FFI_INLINE bool CheckAnyStorage(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDataType;
  }

  static TVM_FFI_INLINE DLDataType CopyFromAnyStorageAfterCheck(const TVMFFIAny* src) {
    return src->v_dtype;
  }

  static TVM_FFI_INLINE std::optional<DLDataType> TryConvertFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDataType) {
      return src->v_dtype;
    }
    // enable string to dtype auto conversion
    if (auto opt_str = TypeTraits<std::string>::TryConvertFromAnyView(src)) {
      return StringToDLDataType(*opt_str);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return ffi::StaticTypeKey::kTVMFFIDataType; }
};
}  // namespace ffi
}  // namespace tvm

// define DLDataType comparison and printing in root namespace
inline std::ostream& operator<<(std::ostream& os, DLDataType dtype) {  // NOLINT(*)
  return tvm::ffi::operator<<(os, dtype);
}

inline bool operator==(const DLDataType& lhs, const DLDataType& rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

inline bool operator!=(const DLDataType& lhs, const DLDataType& rhs) { return !(lhs == rhs); }
#endif  // TVM_FFI_DTYPE_H_
