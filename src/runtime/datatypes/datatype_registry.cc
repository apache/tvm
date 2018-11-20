#include "datatype_registry.h"
#include <tvm/api_registry.h>

namespace tvm {

TVM_REGISTER_API("_register_datatype")
.set_body([](TVMArgs args,  TVMRetValue* ret) {
    DatatypeRegistry::RegisterDatatype(args[0], (uint8_t)args[1].operator int());
  });

void DatatypeRegistry::RegisterDatatype(std::string type_name, uint8_t type_code) {
  auto inst = Global();
  inst->code_to_name[type_code] = type_name;
}

} // namespace tvm
