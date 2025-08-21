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
/*
 * \file src/ffi/json/writer.cc
 *
 * \brief A minimalistic JSON writer based on ffi values.
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#ifdef _MSC_VER
#define TVM_FFI_SNPRINTF _snprintf_s
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4127)
#pragma warning(disable : 4702)
#else
#define TVM_FFI_SNPRINTF snprintf
#endif

namespace tvm {
namespace ffi {
namespace json {

class JSONWriter {
 public:
  static String Stringify(const json::Value& value, Optional<int> indent) {
    JSONWriter writer(indent.value_or(0));
    writer.WriteValue(value);
    return String(std::move(writer.result_));
  }

 private:
  explicit JSONWriter(int indent) : indent_(indent), out_iter_(result_) {}

  static bool FastMathSafeIsNaN(double x) {
#ifdef __FAST_MATH__
    // Bit-level NaN detection (IEEE 754 double)
    // IEEE 754 standard: https://en.wikipedia.org/wiki/IEEE_754
    // NaN is encoded as all 1s in the exponent and non-zero in the mantissa
    static_assert(sizeof(double) == sizeof(uint64_t), "Unexpected double size");
    union {
      double from;
      uint64_t to;
    } u;
    u.from = x;  // write "from", read "to"
    uint64_t bits = u.to;
    uint64_t exponent = (bits >> 52) & 0x7FF;
    uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;
    return (exponent == 0x7FF) && (mantissa != 0);
#else
    // Safe to use std::isnan when fast-math is off
    return std::isnan(x);
#endif
  }

  static bool FastMathSafeIsInf(double x) {
#ifdef __FAST_MATH__
    // IEEE 754 standard: https://en.wikipedia.org/wiki/IEEE_754
    // Inf is encoded as all 1s in the exponent and zero in the mantissa
    static_assert(sizeof(double) == sizeof(uint64_t), "Unexpected double size");
    union {
      double from;
      uint64_t to;
    } u;
    u.from = x;  // write "from", read "to"
    uint64_t bits = u.to;
    uint64_t exponent = (bits >> 52) & 0x7FF;
    uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;
    // inf is encoded as all 1s in the exponent and zero in the mantissa
    return (exponent == 0x7FF) && (mantissa == 0);
#else
    return std::isinf(x);
#endif
  }

  void WriteValue(const json::Value& value) {
    switch (value.type_index()) {
      case TypeIndex::kTVMFFINone: {
        WriteLiteral("null", 4);
        break;
      }
      case TypeIndex::kTVMFFIBool: {
        bool bool_value = details::AnyUnsafe::CopyFromAnyViewAfterCheck<bool>(value);
        if (bool_value) {
          WriteLiteral("true", 4);
        } else {
          WriteLiteral("false", 5);
        }
        break;
      }
      case TypeIndex::kTVMFFIInt: {
        WriteInt(details::AnyUnsafe::CopyFromAnyViewAfterCheck<int64_t>(value));
        break;
      }
      case TypeIndex::kTVMFFIFloat: {
        WriteFloat(details::AnyUnsafe::CopyFromAnyViewAfterCheck<double>(value));
        break;
      }
      case TypeIndex::kTVMFFISmallStr:
      case TypeIndex::kTVMFFIStr: {
        WriteString(details::AnyUnsafe::CopyFromAnyViewAfterCheck<String>(value));
        break;
      }
      case TypeIndex::kTVMFFIArray: {
        WriteArray(details::AnyUnsafe::CopyFromAnyViewAfterCheck<json::Array>(value));
        break;
      }
      case TypeIndex::kTVMFFIMap: {
        WriteObject(details::AnyUnsafe::CopyFromAnyViewAfterCheck<json::Object>(value));
        break;
      }
      default: {
        TVM_FFI_THROW(ValueError) << "Unsupported type: `" << value.GetTypeKey() << "`";
        TVM_FFI_UNREACHABLE();
      }
    }
  }

  void WriteLiteral(const char* literal, int size) {
    for (int i = 0; i < size; ++i) {
      *out_iter_++ = literal[i];
    }
  }

  void WriteInt(int64_t value) {
    // the biggest possible string representation of -INT64_MIN
    char buffer[sizeof("-9223372036854775808") + 1];
    int size = TVM_FFI_SNPRINTF(buffer, sizeof(buffer), "%" PRId64, value);
    WriteLiteral(buffer, size);
  }

  void WriteFloat(double value) {
    // largest possible string representation of a double is around 24 chars plus
    // one null terminator keep 32 to be safe
    char buffer[32];
    if (FastMathSafeIsNaN(value)) {
      WriteLiteral("NaN", 3);
    } else if (FastMathSafeIsInf(value)) {
      if (value < 0) {
        WriteLiteral("-Infinity", 9);
      } else {
        WriteLiteral("Infinity", 8);
      }
    } else {
      double int_part;
      // if the value can be represented as integer
      if (std::fabs(value) < (1ULL << 53) && std::modf(value, &int_part) == 0) {
        // always print an extra .0 for integer so integer numbers are printed as floats
        // this helps us to distinguish between integer and float, which is not necessary
        // but helps to ensure roundtrip property of the parser/printer in terms of int/float types
        int size = TVM_FFI_SNPRINTF(buffer, sizeof(buffer), "%.1f", int_part);
        WriteLiteral(buffer, size);
      } else {
        // Save 17 decimal digits to avoid loss during loading JSON
        // this is the maximum precision that can be represented in a double
        int size = TVM_FFI_SNPRINTF(buffer, sizeof(buffer), "%.17g", value);
        WriteLiteral(buffer, size);
      }
    }
  }

  void WriteString(const String& value) {
    *out_iter_++ = '"';
    const char* data = value.data();
    const size_t size = value.size();
    for (size_t i = 0; i < size; ++i) {
      switch (data[i]) {
// handle escape characters per JSON spec(RFC 8259)
#define HANDLE_ESCAPE_CHAR(pattern, val)                    \
  case pattern:                                             \
    WriteLiteral(val, std::char_traits<char>::length(val)); \
    break
        HANDLE_ESCAPE_CHAR('\"', "\\\"");
        HANDLE_ESCAPE_CHAR('\\', "\\\\");
        HANDLE_ESCAPE_CHAR('/', "\\/");
        HANDLE_ESCAPE_CHAR('\b', "\\b");
        HANDLE_ESCAPE_CHAR('\f', "\\f");
        HANDLE_ESCAPE_CHAR('\n', "\\n");
        HANDLE_ESCAPE_CHAR('\r', "\\r");
        HANDLE_ESCAPE_CHAR('\t', "\\t");
#undef HANDLE_ESCAPE_CHAR
        default: {
          uint8_t u8_val = static_cast<uint8_t>(data[i]);
          // this is a control character, print as \uXXXX
          if (u8_val < 0x20 || u8_val == 0x7f) {
            char buffer[8];
            int size = TVM_FFI_SNPRINTF(buffer, sizeof(buffer), "\\u%04x",
                                        static_cast<int32_t>(data[i]) & 0xff);
            WriteLiteral(buffer, size);
          } else {
            *out_iter_++ = data[i];
          }
          break;
        }
      }
    }
    *out_iter_++ = '"';
  }

  void WriteArray(const json::Array& value) {
    *out_iter_++ = '[';
    if (indent_ != 0) {
      total_indent_ += indent_;
    }
    for (size_t i = 0; i < value.size(); ++i) {
      if (i != 0) {
        *out_iter_++ = ',';
      }
      if (indent_ != 0) {
        WriteIndent();
      }
      WriteValue(value[i]);
    }
    if (indent_ != 0) {
      total_indent_ -= indent_;
      WriteIndent();
    }
    *out_iter_++ = ']';
  }

  void WriteObject(const json::Object& value) {
    *out_iter_++ = '{';
    if (indent_ != 0) {
      total_indent_ += indent_;
    }
    int counter = 0;
    for (const auto& [key, value] : value) {
      if (counter++ != 0) {
        *out_iter_++ = ',';
      }
      if (indent_ != 0) {
        WriteIndent();
      }
      auto opt_key = key.as<String>();
      if (!opt_key.has_value()) {
        TVM_FFI_THROW(ValueError) << "Expect key to be string, got `" << key.GetTypeKey() << "`";
      }
      WriteString(*opt_key);
      *out_iter_++ = ':';
      if (indent_ != 0) {
        *out_iter_++ = ' ';
      }
      WriteValue(value);
    }
    if (indent_ != 0) {
      total_indent_ -= indent_;
      WriteIndent();
    }
    *out_iter_++ = '}';
  }

  // Write a newline and indent the current level
  void WriteIndent() {
    *out_iter_++ = '\n';
    for (int i = 0; i < total_indent_; ++i) {
      *out_iter_++ = ' ';
    }
  }

  int indent_ = 0;
  int total_indent_ = 0;
  std::string result_;
  std::back_insert_iterator<std::string> out_iter_;
};

String Stringify(const json::Value& value, Optional<int> indent) {
  return JSONWriter::Stringify(value, indent);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.json.Stringify", Stringify);
});

}  // namespace json
}  // namespace ffi
}  // namespace tvm

#undef TVM_FFI_SNPRINTF
