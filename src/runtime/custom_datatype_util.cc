/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/runtime/custom_datatype_util.cc
 * \brief Custom datatype lookup functions needed by runtime
 */

#include "custom_datatype_util.h"
#include <tvm/runtime/registry.h>

// TODO(gus) all of these functions could wrap the Registry::Get with a CHECK

std::string GetTypeName(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_name");
  CHECK(f) << "Function not found";
  return (*f)(type_code).operator std::string();
}

uint8_t GetTypeCode(const std::string& type_name) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_code");
  CHECK(f) << "Function not found";
  return (*f)(type_name).operator int();
}

bool GetTypeRegistered(uint8_t type_code) {
  auto f = tvm::runtime::Registry::Get("_datatype_get_type_registered");
  CHECK(f) << "Function not found";
  return (*f)(type_code).operator int();
}
