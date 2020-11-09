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
        std::cout << std::hex << +static_cast<std::uint8_t>(data_buf[i]) << std::dec << " ";
        if (((i + 1) % 16) == 0) {
          std::cout << std::endl;
        }

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

// void LLVMCodeGenParams(llvm::LLVMContext* ctx,
//                        llvm::Module* module,
//                        int64_t storage_id_offset,
//                        const Map<String,
//                        const std::unordered_map<String, NDArray>& params,
//                        const std::unordered_map<String, int64_t>& param_ids) {
//   CHECK_EQ(params.size(), params_ids.size())
//     << "Expect param_names and params_ids to have equal lengths, but params.size() == "
//     << params.size() << " and params_ids.size() == " << params_ids.size();

//   llvm::ArrayType* t_sid_ptr_ty =
//     llvm::ArrayType::get(llvm::PointerType::getUnqual(llvm::getVoidTy()));
//   std::vector<llvm::Constant*> sid_ptrs;
//   for (auto kv : params) {

//     sid_ptrs.push_back(
//   }

//   llvm::GlobalVaraible* sid_offset_symbol = new llvm::GlobalVariable(
//     *module, llvm::Type::getInt64Ty(), true, llvm::GlobalVariable::InternalLinkage,
//     llvm::ConstantInt::getSigned(ctx, sid_offset), ::tvm::runtime::symbol::tvm_sid_offset);
//   llvm::GlobalVariable* sid_ptrs_symbol = new llvm::GlobalVariable(
//     *module, t_sid_ptr_ty, true, llvm::GlobalValue::InternalLinkage,
//     llvm::ConstantArray::get(t_sid_ptr_ty, sid_ptrs), ::tvm::runtime::symbol::tvm_param_array);
// }

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
