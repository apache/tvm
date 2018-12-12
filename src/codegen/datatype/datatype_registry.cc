#include "datatype_registry.h"
#include <tvm/api_registry.h>

namespace tvm {

TVM_REGISTER_GLOBAL("_register_datatype")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      DatatypeRegistry::Global()->RegisterDatatype(
          args[0], (uint8_t)args[1].operator int());
    });

TVM_REGISTER_GLOBAL("_get_type_code")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = DatatypeRegistry::Global()->GetTypeCode(args[0]);
    });

TVM_REGISTER_GLOBAL("_get_type_name")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = DatatypeRegistry::Global()->GetTypeName(args[0].operator int());
    });

void DatatypeRegistry::RegisterDatatype(const std::string& type_name,
                                        uint8_t type_code) {
  code_to_name[type_code] = type_name;
  name_to_code[type_name] = type_code;
}

uint8_t DatatypeRegistry::GetTypeCode(const std::string& type_name) {
  return name_to_code[type_name];
}

std::string DatatypeRegistry::GetTypeName(uint8_t type_code) {
  return code_to_name[type_code];
}

const runtime::PackedFunc* GetCastLowerFunc(const std::string& target,
                                            uint8_t type_code,
                                            uint8_t src_type_code) {
  std::ostringstream ss;
  ss << "tvm.datatypes.lower.";
  ss << target << ".";
  ss << "cast"
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

const runtime::PackedFunc* GetAddLowerFunc(const std::string& target,
                                           uint8_t type_code) {
  internal_assert(DatatypeRegistry::Global()->DatatypeRegistered(type_code));
  return runtime::Registry::Get(
      "tvm.datatypes." + target + ".lower.add." +
      DatatypeRegistry::Global()->GetTypeName(type_code));
}

}  // namespace tvm
