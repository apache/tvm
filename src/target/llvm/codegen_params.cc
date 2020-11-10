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
 * \file codegen_blob.cc
 */
#ifdef TVM_LLVM_VERSION

#include "codegen_params.h"

#include <iomanip>

namespace tvm {
namespace codegen {

class DLManagedTensorDeleter {
 public:
  void operator()(DLManagedTensor* ptr) {
    ptr->deleter(ptr);
  }
};

llvm::ConstantArray* NDArrayToLLVMArray(llvm::LLVMContext* ctx, ::tvm::runtime::NDArray arr) {
  llvm::Type* element_type = nullptr;

  auto arr_type = arr.DataType();
  CHECK_EQ(arr_type.lanes(), 1)
      << "CodegenParams: only support generating 1-lane parameters; saw " << arr_type.lanes();

  auto shape = arr.Shape();
  int num_elements = 1;
  for (auto shape_elem : shape) {
    num_elements *= shape_elem;
  }

  std::unique_ptr<DLManagedTensor, DLManagedTensorDeleter> tensor(arr.ToDLPack());
  std::vector<llvm::Constant*> elements;

  switch (arr_type.code()) {
  case runtime::DataType::kInt:
    CHECK(arr_type.bits() == 8 ||
          arr_type.bits() == 16 ||
          arr_type.bits() == 32 ||
          arr_type.bits() == 64)
      << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
      << arr_type.bits() << "-bit array";
    element_type = llvm::Type::getIntNTy(*ctx, arr_type.bits());

    if (arr_type.bits() == 8) {
      int8_t* data_buf = static_cast<int8_t*>(tensor->dl_tensor.data);
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(llvm::ConstantInt::getSigned(element_type, data_buf[i]));
      }
    } else if (arr_type.bits() == 16) {
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantInt::getSigned(element_type, ((int16_t*) tensor->dl_tensor.data)[i]));
      }
    } else if (arr_type.bits() == 32) {
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantInt::getSigned(element_type, ((int32_t*) tensor->dl_tensor.data)[i]));
      }
    } else if (arr_type.bits() == 64) {
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantInt::getSigned(element_type, ((int64_t*) tensor->dl_tensor.data)[i]));
      }
    } else {
      CHECK(false) << "should not get here";
    }
    break;

  case runtime::DataType::TypeCode::kUInt:
    CHECK(arr_type.bits() == 8 ||
          arr_type.bits() == 16 ||
          arr_type.bits() == 32 ||
          arr_type.bits() == 64)
      << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
      << arr_type.bits() << "-bit array";
    element_type = llvm::Type::getIntNTy(*ctx, arr_type.bits());

    if (arr_type.bits() == 8) {
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantInt::get(element_type, ((int8_t*) tensor->dl_tensor.data)[i]));
      }
    } else if (arr_type.bits() == 16) {
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantInt::get(element_type, ((int16_t*) tensor->dl_tensor.data)[i]));
      }
    } else if (arr_type.bits() == 32) {
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantInt::get(element_type, ((int32_t*) tensor->dl_tensor.data)[i]));
      }
    } else if (arr_type.bits() == 64) {
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantInt::get(element_type, ((int64_t*) tensor->dl_tensor.data)[i]));
      }
    } else {
      CHECK(false) << "should not get here";
    }
    break;

  case runtime::DataType::TypeCode::kFloat:
    if (arr_type.bits() == 32) {
      element_type = llvm::Type::getFloatTy(*ctx);
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantFP::get(element_type, ((float*) tensor->dl_tensor.data)[i]));
      }
    } else if (arr_type.bits() == 64) {
      element_type = llvm::Type::getDoubleTy(*ctx);
      for (int i = 0; i < num_elements; i++) {
        elements.emplace_back(
          llvm::ConstantFP::get(element_type, ((double*) tensor->dl_tensor.data)[i]));
      }
    } else {
      CHECK(false) << "CodegenParams: only support 32- or 64-bit floating point; saw "
                   << arr_type.bits() << "-bit array";
    }
    break;

  default:
    CHECK(false) << "Data type not supported";
  }

  return llvm::cast<llvm::ConstantArray>(
    llvm::ConstantArray::get(llvm::ArrayType::get(element_type, num_elements),
                             llvm::ArrayRef<llvm::Constant*>(elements)));
}


static constexpr const char* kFloatCast = "(float)";
static constexpr const char* kDoubleCast = "(double)";

static constexpr const int kMaxLineLength = 80;


void NDArrayDataToC(::tvm::runtime::NDArray arr, int indent_chars, std::ostream& os) {
  auto arr_type = arr.DataType();
  CHECK_EQ(arr_type.lanes(), 1)
      << "CodegenParams: only support generating 1-lane parameters; saw " << arr_type.lanes();

  int one_element_size_bytes = (arr_type.bits() / 4) + (2 /* "0x" */) + (2 /* ", " */);
  if (arr_type.code() == runtime::DataType::TypeCode::kInt) {
    one_element_size_bytes += 1; // sign bit
    if (arr_type.bits() > 32) {
      one_element_size_bytes += 2;  // "UL"
    }
  } else if (arr_type.code() == runtime::DataType::TypeCode::kUInt) {
    if (arr_type.bits() > 32) {
      one_element_size_bytes += 1; // "L"
    }
  } else if (arr_type.code() == runtime::DataType::TypeCode::kFloat) {
    // Floats and doubles are printed as hex but casted.
    one_element_size_bytes += 1 /* sign */ + 1 /* decimal point */ + 1 /* exponent sign */;
  }

  int elements_per_row = 16;
  while (elements_per_row > 1 &&
         (elements_per_row * one_element_size_bytes) > (kMaxLineLength - indent_chars)) {
    elements_per_row /= 2;
  }

  std::string indent_str(indent_chars, ' ');
  os << indent_str;

  auto shape = arr.Shape();
  int num_elements = 1;
  for (auto shape_elem : shape) {
    num_elements *= shape_elem;
  }

  std::unique_ptr<DLManagedTensor, DLManagedTensorDeleter> tensor(arr.ToDLPack());
  auto old_fmtflags = os.flags();
  os.setf(std::ios::right | std::ios::hex | std::ios::fixed | std::ios::scientific,
          std::ios::adjustfield | std::ios::basefield | std::ios::floatfield);
  os.fill('0');
  switch (arr_type.code()) {
  case runtime::DataType::kInt:
    CHECK(arr_type.bits() == 8 ||
          arr_type.bits() == 16 ||
          arr_type.bits() == 32 ||
          arr_type.bits() == 64)
      << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
      << arr_type.bits() << "-bit array";

    if (arr_type.bits() == 8) {
      for (int i = 0; i < num_elements; i++) {
        // NOTE: for special types int8_t and uint8_t, need to promote to int type to avoid printing
        // as a char.
        int8_t elem = static_cast<int8_t*>(tensor->dl_tensor.data)[i];
        uint8_t to_print;
        if (elem < 0) {
          os << "-";
          to_print = -elem;
        } else {
          os << "+";
          to_print = elem;
        }
        os << "0x" << std::setw(2) << +static_cast<std::uint8_t>(to_print);
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else if (arr_type.bits() == 16) {
      for (int i = 0; i < num_elements; i++) {
        int16_t elem = static_cast<int16_t*>(tensor->dl_tensor.data)[i];
        uint16_t to_print;
        if (elem < 0) {
          os << "-";
          to_print = -elem;
        } else {
          os << "+";
          to_print = elem;
        }
        os << "0x" << std::setw(4) << to_print;
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else if (arr_type.bits() == 32) {
      for (int i = 0; i < num_elements; i++) {
        int32_t elem = static_cast<int32_t*>(tensor->dl_tensor.data)[i];
        uint32_t to_print ;
        if (elem < 0) {
          os << "-";
          to_print = -elem;
        } else {
          os << "+";
          to_print = elem;
        }
        os << "0x" << std::setw(8) << to_print;
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else if (arr_type.bits() == 64) {
      for (int i = 0; i < num_elements; i++) {
        int64_t elem = static_cast<int64_t*>(tensor->dl_tensor.data)[i];
        uint64_t to_print;
        if (elem < 0) {
          os << "-";
          to_print = -elem;
        } else {
          os << "+";
          to_print = elem;
        }
        os << "0x" << std::setw(16) << to_print;
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else {
      CHECK(false) << "should not get here";
    }
    break;

  case runtime::DataType::TypeCode::kUInt:
    CHECK(arr_type.bits() == 8 ||
          arr_type.bits() == 16 ||
          arr_type.bits() == 32 ||
          arr_type.bits() == 64)
      << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
      << arr_type.bits() << "-bit array";

    if (arr_type.bits() == 8) {
      for (int i = 0; i < num_elements; i++) {
        // NOTE: for special types int8_t and uint8_t, need to promote to int type to avoid printing
        // as a char.
        os << "0x" << std::setw(2)
           << +static_cast<std::uint8_t>(static_cast<uint8_t*>(tensor->dl_tensor.data)[i]);
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else if (arr_type.bits() == 16) {
      for (int i = 0; i < num_elements; i++) {
        os << "0x" << std::setw(4) << static_cast<uint16_t*>(tensor->dl_tensor.data)[i];
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else if (arr_type.bits() == 32) {
      for (int i = 0; i < num_elements; i++) {
        os << "0x" << std::setw(8) << static_cast<uint32_t*>(tensor->dl_tensor.data)[i];
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else if (arr_type.bits() == 64) {
      for (int i = 0; i < num_elements; i++) {
        os << "0x" << std::setw(16) << static_cast<uint64_t*>(tensor->dl_tensor.data)[i] << "UL";
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else {
      CHECK(false) << "should not get here";
    }
    break;

  case runtime::DataType::TypeCode::kFloat:
    if (arr_type.bits() == 32) {
      for (int i = 0; i < num_elements; i++) {
        os << static_cast<float*>(tensor->dl_tensor.data)[i];
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
      std::cout << "\n";
    } else if (arr_type.bits() == 64) {
      for (int i = 0; i < num_elements; i++) {
        os << static_cast<double*>(tensor->dl_tensor.data)[i];
        if (i < num_elements - 1) { os << ", "; }
        if (((i + 1) % elements_per_row) == 0) { os << "\n" << indent_str; }
      }
    } else {
      CHECK(false) << "CodegenParams: only support 32- or 64-bit floating point; saw "
                   << arr_type.bits() << "-bit array";
    }
    break;

  default:
    CHECK(false) << "Data type not supported";
  }

  if (num_elements % elements_per_row != 0) {
    os << "\n";
  }
  os.flags(old_fmtflags);
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
