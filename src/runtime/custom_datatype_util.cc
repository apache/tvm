/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/runtime/custom_datatype_util.cc
 * \brief Custom datatype lookup functions needed by runtime
 */

// These extern C functions are needed to support custom datatypes in
// packed_func.h specifically. The custom datatype facilities in packed_func.h
// depend on the TVM registry, but including tvm/runtime/registry.h in
// packed_func.h creates an include cycle. This allows us to avoid the cycle (in
// an admittedly hacky way).

#include <string.h>
#include <tvm/runtime/registry.h>
#include <string>

// TODO(gus) all of these functions could wrap the Registry::Get with a CHECK

// These functions should only be used within the runtime. If you need to look
// up custom datatype information outside of the runtime, use the
// DatatypeRegistry.

// This function returns a string which must be deleted by the caller.
extern "C" void GetTypeName(uint8_t type_code, char** ret) {
  auto name =
      (*tvm::runtime::Registry::Get("_datatype_get_type_name"))(type_code).
      operator std::string();
  *ret = new char[name.length() + 1];
  strcpy(*ret, name.c_str());
}

extern "C" uint8_t GetTypeCode(const std::string& type_name) {
  return (*tvm::runtime::Registry::Get("_datatype_get_type_code"))(type_name).
  operator int();
}
