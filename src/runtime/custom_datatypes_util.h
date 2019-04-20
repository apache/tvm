/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/runtime/custom_datatypes_util.h
 * \brief Custom datatype lookup functions needed by runtime
 */
#ifndef TVM_RUNTIME_CUSTOM_DATATYPES_UTIL_H_
#define TVM_RUNTIME_CUSTOM_DATATYPES_UTIL_H_

#include <string>

std::string GetTypeName(uint8_t type_code);
uint8_t GetTypeCode(const std::string& type_name);
bool GetTypeRegistered(uint8_t type_code);

#endif  // TVM_RUNTIME_CUSTOM_DATATYPES_UTIL_H_
