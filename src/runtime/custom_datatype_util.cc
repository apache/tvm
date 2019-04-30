/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/runtime/custom_datatype_util.cc
 * \brief Custom datatype lookup functions needed by runtime
 */

#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

// Public function (needed in packed_func.h) for getting a name from a custom type's code
TVM_DLL std::string GetCustomTypeName(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_name");
  CHECK(f) << "Function not found";
  return (*f)(type_code).operator std::string();
}

TVM_DLL uint8_t GetCustomTypeCode(const std::string& type_name) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_code");
  CHECK(f) << "Function not found";
  return (*f)(type_name).operator int();
}

TVM_DLL bool GetCustomTypeRegistered(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_registered");
  CHECK(f) << "Function not found";
  return (*f)(type_code).operator bool();
}

}  // namespace runtime
}  // namespace tvm
