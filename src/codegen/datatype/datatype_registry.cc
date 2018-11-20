#include "datatype_registry.h"
#include <tvm/api_registry.h>
#include <iostream>

namespace tvm {

TVM_REGISTER_API("_register_datatype")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    DatatypeRegistry::RegisterDatatype(args[0], (uint8_t)args[1].operator int());
  });

void DatatypeRegistry::RegisterDatatype(const std::string& type_name, uint8_t type_code) {
  auto inst = Global();
  inst->code_to_name[type_code] = type_name;
  inst->name_to_code[type_name] = type_code;
}

uint8_t DatatypeRegistry::GetTypeCode(const std::string& type_name) {
  auto inst = Global();
  return inst->name_to_code[type_name];
}

std::string DatatypeRegistry::GetTypeName(uint8_t type_code) {
  auto inst = Global();
  return inst->code_to_name[type_code];
}

} // namespace tvm
