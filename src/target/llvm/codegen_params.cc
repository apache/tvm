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
#ifdef TVM_LLVM_VERSION

#include "codegen_params.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/Casting.h>

#include <algorithm>
#include <type_traits>
#include <vector>

namespace tvm {
namespace codegen {

template <typename T, typename E = void>
struct LLVMConstantGetter {
  static llvm::Constant* getElement(llvm::Type* ty, T t);
};

template <typename T>
struct LLVMConstantGetter<
    T, std::enable_if_t<(std::is_integral<T>::value && std::is_signed<T>::value)>> {
  static llvm::Constant* getElement(llvm::Type* ty, T t) {
    return llvm::ConstantInt::getSigned(ty, t);
  }
};

template <typename T>
struct LLVMConstantGetter<
    T, std::enable_if_t<(std::is_integral<T>::value && !std::is_signed<T>::value)>> {
  static llvm::Constant* getElement(llvm::Type* ty, T t) { return llvm::ConstantInt::get(ty, t); }
};

template <typename T>
struct LLVMConstantGetter<T, std::enable_if_t<std::is_floating_point<T>::value>> {
  static llvm::Constant* getElement(llvm::Type* ty, T t) { return llvm::ConstantFP::get(ty, t); }
};

template <typename T, typename = std::enable_if<std::is_pod<T>::value>>
void BuildLLVMVector(llvm::Type* element_type, void* tensor_data, size_t num_elements,
                     std::vector<llvm::Constant*>* elements) {
  elements->resize(num_elements, nullptr);
  std::transform(static_cast<T*>(tensor_data), static_cast<T*>(tensor_data) + num_elements,
                 elements->begin(),
                 [&](T t) { return LLVMConstantGetter<T>::getElement(element_type, t); });
}

llvm::ConstantArray* NDArrayToLLVMArray(llvm::LLVMContext* ctx, ::tvm::runtime::NDArray arr) {
  llvm::Type* element_type = nullptr;

  auto arr_type = arr.DataType();
  CHECK(arr.IsContiguous()) << "CodegenParams: only support contiguous arrays";
  CHECK_EQ(arr->device.device_type, kDLCPU) << "CodegenParams: only support contiguous arrays";
  CHECK_EQ(arr_type.lanes(), 1) << "CodegenParams: only support generating 1-lane parameters; saw "
                                << arr_type.lanes();

  auto shape = arr.Shape();
  int num_elements = 1;
  for (auto shape_elem : shape) {
    num_elements *= shape_elem;
  }

  std::vector<llvm::Constant*> elements;

  switch (arr_type.code()) {
    case runtime::DataType::kInt:
      CHECK(arr_type.bits() == 8 || arr_type.bits() == 16 || arr_type.bits() == 32 ||
            arr_type.bits() == 64)
          << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
          << arr_type.bits() << "-bit array";
      element_type = llvm::Type::getIntNTy(*ctx, arr_type.bits());

      switch (arr_type.bits()) {
        case 8:
          BuildLLVMVector<int8_t>(element_type, arr->data, num_elements, &elements);
          break;
        case 16:
          BuildLLVMVector<int16_t>(element_type, arr->data, num_elements, &elements);
          break;
        case 32:
          BuildLLVMVector<int32_t>(element_type, arr->data, num_elements, &elements);
          break;
        case 64:
          BuildLLVMVector<int64_t>(element_type, arr->data, num_elements, &elements);
          break;
        default:
          ICHECK(false) << "should not get here";
          break;
      }
      break;

    case runtime::DataType::TypeCode::kUInt:
      CHECK(arr_type.bits() == 8 || arr_type.bits() == 16 || arr_type.bits() == 32 ||
            arr_type.bits() == 64)
          << "CodegenParams: only support generating 8-, 16-, 32-, or 64-bit integer params; saw "
          << arr_type.bits() << "-bit array";
      element_type = llvm::Type::getIntNTy(*ctx, arr_type.bits());

      switch (arr_type.bits()) {
        case 8:
          BuildLLVMVector<uint8_t>(element_type, arr->data, num_elements, &elements);
          break;
        case 16:
          BuildLLVMVector<uint16_t>(element_type, arr->data, num_elements, &elements);
          break;
        case 32:
          BuildLLVMVector<uint32_t>(element_type, arr->data, num_elements, &elements);
          break;
        case 64:
          BuildLLVMVector<uint64_t>(element_type, arr->data, num_elements, &elements);
          break;
        default:
          ICHECK(false) << "should not get here";
          break;
      }
      break;

    case runtime::DataType::TypeCode::kFloat:
      switch (arr_type.bits()) {
        case 16:
          // NOTE: float16 is treated as uint16_t.
          element_type = llvm::Type::getIntNTy(*ctx, arr_type.bits());
          BuildLLVMVector<uint16_t>(element_type, arr->data, num_elements, &elements);
          break;
        case 32:
          element_type = llvm::Type::getFloatTy(*ctx);
          BuildLLVMVector<float>(element_type, arr->data, num_elements, &elements);
          break;
        case 64:
          element_type = llvm::Type::getDoubleTy(*ctx);
          BuildLLVMVector<double>(element_type, arr->data, num_elements, &elements);
          break;
        default:
          CHECK(false) << "CodegenParams: only support 32- or 64-bit floating point; saw "
                       << arr_type.bits() << "-bit array";
          break;
      }
      break;

    case runtime::DataType::TypeCode::kBFloat:
      CHECK(arr_type.bits() == 16)
          << "CodegenParams: only support 16-bit bfloat; saw " << arr_type.bits() << "-bit array";
      element_type = llvm::Type::getIntNTy(*ctx, arr_type.bits());
      BuildLLVMVector<uint16_t>(element_type, arr->data, num_elements, &elements);

    default:
      CHECK(false) << "Data type not supported";
  }

  return llvm::cast<llvm::ConstantArray>(llvm::ConstantArray::get(
      llvm::ArrayType::get(element_type, num_elements), llvm::ArrayRef<llvm::Constant*>(elements)));
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
