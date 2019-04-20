/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/codegen/custom_datatypes/datatype_registry.cc
 * \brief Custom datatypes registry
 */

#include "registry.h"
#include <tvm/api_registry.h>

namespace tvm {
namespace custom_datatypes {

TVM_REGISTER_GLOBAL("_custom_datatypes_register")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Registry::Global()->RegisterCustomDatatype(
          args[0], static_cast<uint8_t>(args[1].operator int()));
    });

TVM_REGISTER_GLOBAL("_custom_datatypes_get_type_code")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Registry::Global()->GetTypeCode(args[0]);
    });

TVM_REGISTER_GLOBAL("_custom_datatypes_get_type_name")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Registry::Global()->GetTypeName(args[0].operator int());
    });

TVM_REGISTER_GLOBAL("_custom_datatypes_get_type_registered")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = Registry::Global()->GetTypeRegistered(
          args[0].operator int());
    });

void Registry::RegisterCustomDatatype(const std::string &type_name,
                                        uint8_t type_code) {
  code_to_name[type_code] = type_name;
  name_to_code[type_name] = type_code;
}

uint8_t Registry::GetTypeCode(const std::string &type_name) {
  CHECK(name_to_code.find(type_name) != name_to_code.end())
      << "Type name " << type_name << " not registered";
  return name_to_code[type_name];
}

std::string Registry::GetTypeName(uint8_t type_code) {
  CHECK(code_to_name.find(type_code) != code_to_name.end())
      << "Type code " << static_cast<unsigned>(type_code) << " not registered";
  return code_to_name[type_code];
}

const runtime::PackedFunc *GetCastLowerFunc(const std::string &target,
                                            uint8_t type_code,
                                            uint8_t src_type_code) {
  std::ostringstream ss;
  ss << "tvm.custom_datatypes.lower.";
  ss << target << ".";
  ss << "Cast"
     << ".";

  if (Registry::Global()->GetTypeRegistered(type_code)) {
    ss << Registry::Global()->GetTypeName(type_code);
  } else {
    ss << runtime::TypeCode2Str(type_code);
  }

  ss << ".";

  if (Registry::Global()->GetTypeRegistered(src_type_code)) {
    ss << Registry::Global()->GetTypeName(src_type_code);
  } else {
    ss << runtime::TypeCode2Str(src_type_code);
  }

  return runtime::Registry::Get(ss.str());
}

uint64_t ConvertConstScalar(uint8_t type_code, double value) {
  std::ostringstream ss;
  ss << "tvm.custom_datatypes.convertconstscalar.float.";
  ss << Registry::Global()->GetTypeName(type_code);
  auto make_const_scalar_func = runtime::Registry::Get(ss.str());
  return (*make_const_scalar_func)(value).operator uint64_t();
}

}  // namespace custom_datatypes
}  // namespace tvm
