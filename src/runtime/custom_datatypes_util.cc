/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/runtime/custom_datatypes_util.cc
 * \brief Custom datatype lookup functions needed by runtime
 */

#include "custom_datatypes_util.h"
#include <tvm/runtime/registry.h>

// TODO(gus) all of these functions could wrap the Registry::Get with a CHECK

std::string GetTypeName(uint8_t type_code) {
  return (*tvm::runtime::Registry::Get("_custom_datatypes_get_type_name"))(
             type_code)
      .
      operator std::string();
}

uint8_t GetTypeCode(const std::string& type_name) {
  return (*tvm::runtime::Registry::Get("_custom_datatypes_get_type_code"))(
             type_name)
      .
      operator int();
}

bool GetTypeRegistered(uint8_t type_code) {
  return (*tvm::runtime::Registry::Get(
      "_custom_datatypes_get_type_registered"))(type_code)
      .
      operator int();
}
