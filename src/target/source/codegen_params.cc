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
 * \file codegen_params.cc
 */

#include "codegen_params.h"

#include <dlpack/dlpack.h>

#include <cmath>
#include <iomanip>
#include <memory>
#include <string>

namespace tvm {
namespace codegen {

/*! \brief maximum line length of generated parameters, including indent. */
static constexpr const int kMaxLineLength = 80;

static int ComputeNumElementsPerRow(int one_element_size_bytes, int indent_chars) {
  if (one_element_size_bytes > kMaxLineLength - indent_chars) {
    return 1;
  }
  // When multiple elements fit per line, divide the available space by the size of one element,
  // and return the largest power of 2 less than the result. Using power-of-2-sized elements allows
  // for easily traversing the generated code.
  int elements_per_row = (kMaxLineLength - indent_chars) / one_element_size_bytes;

  // Implementation of fls. Iteratively clear the LSB until one bit remains.
  while ((elements_per_row & (elements_per_row - 1)) > 0) {
    elements_per_row &= elements_per_row - 1;
  }
  return elements_per_row;
}

template <typename T, typename Enable = std::enable_if<std::is_integral<T>::value>>
void PrintIntegralArray(void* data, size_t num_elements, int indent_chars, std::ostream& os,
                        const std::string& eol) {
  int one_element_size_bytes = (sizeof(T) / 4) + (2 /* "0x" */) + (2 /* ", " */);
  if (std::is_signed<T>::value) {
    one_element_size_bytes += 1;  // sign character
    if (sizeof(T) == 64 / 8) {
      one_element_size_bytes += 2;  // "LL"
    }
  } else {
    if (sizeof(T) == 64 / 8) {
      one_element_size_bytes += 3;  // "ULL"
    }
  }

  size_t elements_per_row = ComputeNumElementsPerRow(one_element_size_bytes, indent_chars);
  std::string indent_str(indent_chars, ' ');

  for (size_t i = 0; i < num_elements; i++) {
    if ((i % elements_per_row) == 0) {
      os << indent_str;
    }
    int64_t elem = static_cast<T*>(data)[i];
    if (std::is_signed<T>::value) {
      uint64_t to_print;
      if (elem < 0) {
        os << "-";
        to_print = -elem;
      } else {
        os << "+";
        to_print = elem;
      }
      os << "0x" << std::setw(sizeof(T) * 8 / 4) << static_cast<std::uint64_t>(to_print);
      if (sizeof(T) == 64 / 8) {
        os << "LL";
      }
    } else {
      os << "0x" << std::setw(sizeof(T) * 8 / 4) << static_cast<std::uint64_t>(elem);
      if (sizeof(T) == 64 / 8) {
        os << "ULL";
      }
    }
    if (i < num_elements - 1) {
      os << ", ";
    }
    if ((i % elements_per_row) == elements_per_row - 1) {
      os << eol;
    }
  }

  if ((num_elements % elements_per_row) != 0) {
    os << eol;
  }
}

template <typename T, typename Enable = std::enable_if<std::is_floating_point<T>::value>>
void PrintFloatingPointArray(void* data, size_t num_elements, int indent_chars, std::ostream& os,
                             const std::string& eol) {
  // Floats and doubles are printed as hex but casted.
  int one_element_size_bytes = (sizeof(T) / 4) + (2 /* "0x" */) + (2 /* ", " */) + 1 /* sign */ +
                               1 /* decimal point */ + 1 /* exponent sign */;
  if (sizeof(T) == 64 / 8) {
    one_element_size_bytes += 2; /* 4 decimal digits in exponent, relative to bits / 4 */
  } else if (sizeof(T) == 32 / 8) {
    one_element_size_bytes += 1; /* extra decimal digit in exponent, relative to bits / 4 */
  }

  size_t elements_per_row = ComputeNumElementsPerRow(one_element_size_bytes, indent_chars);
  std::string indent_str(indent_chars, ' ');

  std::stringstream ss;
  if (std::is_signed<T>::value) {
    ss.setf(std::ios::hex | std::ios::showbase | std::ios::fixed | std::ios::scientific,
            std::ios::basefield | std::ios::showbase | std::ios::floatfield);
  } else {
    ss.setf(std::ios::hex | std::ios::fixed | std::ios::scientific,
            std::ios::basefield | std::ios::showbase | std::ios::floatfield);
  }
  for (size_t i = 0; i < num_elements; i++) {
    if ((i % elements_per_row) == 0) {
      os << indent_str;
    }

    T elem = static_cast<T*>(data)[i];
    if (std::isinf(elem)) {
      // C99 standard.
      os << (elem < 0 ? "-" : " ") << std::setw(one_element_size_bytes - 1) << "INFINITY";
    } else if (std::isnan(elem)) {
      // GNU extension, implemenatation-dependent.
      os << std::setw(one_element_size_bytes) << "NAN";
    } else {
      ss << elem;
      os << std::setw(one_element_size_bytes) << ss.str();
      ss.str("");
    }
    if (i < num_elements - 1) {
      os << ", ";
    }
    if ((i % elements_per_row) == elements_per_row - 1) {
      os << eol;
    }
  }

  if ((num_elements % elements_per_row) != 0) {
    os << eol;
  }
}

void NDArrayDataToC(::tvm::runtime::NDArray arr, int indent_chars, std::ostream& os,
                    const std::string& eol) {
  auto arr_type = arr.DataType();
  CHECK_EQ(arr_type.lanes(), 1) << "CodegenParams: only support generating 1-lane parameters; saw "
                                << arr_type.lanes();

  auto shape = arr.Shape();
  int num_elements = 1;
  for (auto shape_elem : shape) {
    num_elements *= shape_elem;
  }

  auto old_fmtflags = os.flags();
  os.setf(std::ios::internal | std::ios::hex,
          std::ios::adjustfield | std::ios::basefield | std::ios::showbase);
  os.fill('0');
  switch (arr_type.code()) {
    case runtime::DataType::kInt:
      CHECK(arr_type.bits() == 8 || arr_type.bits() == 16 || arr_type.bits() == 32 ||
            arr_type.bits() == 64)
          << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
          << arr_type.bits() << "-bit array";
      if (arr_type.bits() == 8) {
        PrintIntegralArray<int8_t>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 16) {
        PrintIntegralArray<int16_t>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 32) {
        PrintIntegralArray<int32_t>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 64) {
        PrintIntegralArray<int64_t>(arr->data, num_elements, indent_chars, os, eol);
      } else {
        CHECK(false) << "should not get here";
      }
      break;

    case runtime::DataType::TypeCode::kUInt:
      CHECK(arr_type.bits() == 8 || arr_type.bits() == 16 || arr_type.bits() == 32 ||
            arr_type.bits() == 64)
          << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
          << arr_type.bits() << "-bit array";

      if (arr_type.bits() == 8) {
        PrintIntegralArray<uint8_t>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 16) {
        PrintIntegralArray<uint16_t>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 32) {
        PrintIntegralArray<uint32_t>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 64) {
        PrintIntegralArray<uint64_t>(arr->data, num_elements, indent_chars, os, eol);
      } else {
        CHECK(false) << "should not get here";
      }
      break;

    case runtime::DataType::TypeCode::kFloat: {
      os.fill(' ');
      os.setf(std::ios::left, std::ios::adjustfield);
      if (arr_type.bits() == 16) {
        // NOTE: print types not widely supported by C as uint16_t.
        PrintIntegralArray<uint16_t>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 32) {
        PrintFloatingPointArray<float>(arr->data, num_elements, indent_chars, os, eol);
      } else if (arr_type.bits() == 64) {
        PrintFloatingPointArray<double>(arr->data, num_elements, indent_chars, os, eol);
      } else {
        CHECK(false) << "CodegenParams: only support 32- or 64-bit floating point; saw "
                     << arr_type.bits() << "-bit array";
      }
      break;
    }

    case runtime::DataType::TypeCode::kBFloat: {
      // NOTE: print types not widely supported by C as uint16_t.
      CHECK(arr_type.bits() == 16)
          << "CodegenParams: only support generating 16-bit bfloat params; saw " << arr_type.bits()
          << "-bit array";
      PrintIntegralArray<uint16_t>(arr->data, num_elements, indent_chars, os, eol);
      break;
    }

    default:
      CHECK(false) << "Data type '" << arr_type << "' not supported";
  }

  os.flags(old_fmtflags);
}

}  // namespace codegen
}  // namespace tvm
