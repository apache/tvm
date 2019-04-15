// TODO(gus) think about renaming this file
// TODO(gus) where's a good place for this file?

#include <tvm/runtime/registry.h>
#include <string>

// TODO(gus) all of these functions could wrap the Registry::Get with a CHECK
// TODO(gus) this is generating warnings due to returning a string.
extern "C" std::string GetTypeName(uint8_t type_code) {
  return (*tvm::runtime::Registry::Get("_datatype_get_type_name"))(type_code).
  operator std::string();
}

extern "C" uint8_t GetTypeCode(const std::string& type_name) {
  return (*tvm::runtime::Registry::Get("_datatype_get_type_code"))(type_name).
  operator int();
}

extern "C" uint8_t GetCustomDatatypeRegistered(uint8_t type_code) {
  return (*tvm::runtime::Registry::Get("_datatype_registered"))(type_code).
    operator unsigned long();
}
