/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/codegen/datatype/datatype_registry.cc
 * \brief Custom datatypes registry
 */

#include "datatype_registry.h"
#include <tvm/api_registry.h>

namespace tvm {

TVM_REGISTER_GLOBAL("_datatype_register")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      DatatypeRegistry::Global()->RegisterDatatype(
          args[0], static_cast<uint8_t>(args[1].operator int()),
          args[2].operator size_t());
    });

TVM_REGISTER_GLOBAL("_datatype_get_type_code")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = DatatypeRegistry::Global()->GetTypeCode(args[0]);
    });

TVM_REGISTER_GLOBAL("_datatype_get_type_name")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = DatatypeRegistry::Global()->GetTypeName(args[0].operator int());
    });

TVM_REGISTER_GLOBAL("_datatype_registered")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = DatatypeRegistry::Global()->DatatypeRegistered(
          args[0].operator int());
    });

void DatatypeRegistry::RegisterDatatype(const std::string &type_name,
                                        uint8_t type_code,
                                        size_t storage_size) {
  code_to_name[type_code] = type_name;
  name_to_code[type_name] = type_code;
  code_to_storage_size[type_code] = storage_size;
}

uint8_t DatatypeRegistry::GetTypeCode(const std::string &type_name) {
  CHECK(name_to_code.find(type_name) != name_to_code.end())
    << "Type name " << type_name << " not registered";
  return name_to_code[type_name];
}

std::string DatatypeRegistry::GetTypeName(uint8_t type_code) {
  CHECK(code_to_name.find(type_code) != code_to_name.end())
    << "Type code " << static_cast<unsigned>(type_code)
    << " not registered";
  return code_to_name[type_code];
}

const runtime::PackedFunc *GetCastLowerFunc(const std::string &target,
                                            uint8_t type_code,
                                            uint8_t src_type_code) {
  std::ostringstream ss;
  ss << "tvm.datatype.lower.";
  ss << target << ".";
  ss << "Cast"
     << ".";

  if (DatatypeRegistry::Global()->DatatypeRegistered(type_code)) {
    ss << DatatypeRegistry::Global()->GetTypeName(type_code);
  } else {
    ss << runtime::TypeCode2Str(type_code);
  }

  ss << ".";

  if (DatatypeRegistry::Global()->DatatypeRegistered(src_type_code)) {
    ss << DatatypeRegistry::Global()->GetTypeName(src_type_code);
  } else {
    ss << runtime::TypeCode2Str(src_type_code);
  }

  return runtime::Registry::Get(ss.str());
}

}  // namespace tvm
