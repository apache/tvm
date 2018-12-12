#include <tvm/runtime/registry.h>
#include <string>

// TODO(gus) this is generating warnings due to returning a string.
extern "C" std::string GetTypeName(uint8_t type_code) {
  return (*tvm::runtime::Registry::Get("_get_type_name"))(type_code).
  operator std::string();
}

extern "C" uint8_t GetTypeCode(const std::string& type_name) {
  return (*tvm::runtime::Registry::Get("_get_type_code"))(type_name).
  operator int();
}
