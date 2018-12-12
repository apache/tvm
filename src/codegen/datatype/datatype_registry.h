#ifndef DATATYPE_REGISTRY_H_
#define DATATYPE_REGISTRY_H_

#include <tvm/runtime/packed_func.h>
#include <string>
#include <unordered_map>

namespace tvm {

const runtime::PackedFunc* GetCastLowerFunc(const std::string& target,
                                            uint8_t type_code,
                                            uint8_t src_type_code);
const runtime::PackedFunc* GetAddLowerFunc(const std::string& target,
                                           uint8_t type_code);

/*!
 * \brief Registry for custom datatypes.
 *
 * Adding custom datatypes currently requires two steps:
 * 1. Register the datatype with the registry via a call to
 *    DatatypeRegistry::RegisterDatatype. This can also be done in Python
 *    directly---see the TVM globals registered in the corresponding .cc file.
 *    Currently, user should manually choose a type name and a type code,
 *    ensuring that neither conflict with existing types.
 * 2. Use TVM_REGISTER_GLOBAL to register the lowering functions needed to
 *    lower the custom datatype. In general, these will look like:
 *      For Casts: tvm.datatypes.lower.cast.<target>.<type>.<src_type>
 *        Example: tvm.datatypes.lower.cast.llvm.myfloat.float for a Cast from
 *                 float to myfloat.
 *      For other ops: tvm.datatypes.lower.<op>.<target>.<type>
 *        Example: tvm.datatypes.lower.add.llvm.myfloat
 */
class DatatypeRegistry {
 public:
  static inline DatatypeRegistry* Global() {
    static DatatypeRegistry inst;
    return &inst;
  }

  void RegisterDatatype(const std::string& type_name, uint8_t type_code);

  uint8_t GetTypeCode(const std::string& type_name);

  std::string GetTypeName(uint8_t type_code);

  inline bool DatatypeRegistered(uint8_t type_code) {
    return code_to_name.find(type_code) != code_to_name.end();
  }

 private:
  // TODO(gus) is there a typedef for the code?
  std::unordered_map<uint8_t, std::string> code_to_name;
  std::unordered_map<std::string, uint8_t> name_to_code;
};

}  // namespace tvm

#endif
