#ifndef DATATYPE_REGISTRY_H_
#define DATATYPE_REGISTRY_H_

#include <unordered_map>
#include <string>

namespace tvm {

class DatatypeRegistry {
 public:
  static void RegisterDatatype(const std::string& type_name, uint8_t type_code);
  static uint8_t GetTypeCode(const std::string& type_name);
  static std::string GetTypeName(uint8_t type_code);

private:
  static inline DatatypeRegistry* Global() {
    static DatatypeRegistry inst;
    return &inst;
  }

  // TODO(gus): ...what's the normal way to do this?
  std::unordered_map<uint8_t, std::string> code_to_name;
  std::unordered_map<std::string, uint8_t> name_to_code;
};

} // namespace tvm

#endif
