#ifndef DATATYPE_REGISTRY_H_
#define DATATYPE_REGISTRY_H_

#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <string>
#include <unordered_map>

namespace tvm {

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
 *      For Casts: tvm.datatype.lower.cast.<target>.<type>.<src_type>
 *        Example: tvm.datatype.lower.cast.llvm.myfloat.float for a Cast from
 *                 float to myfloat.
 *      For other ops: tvm.datatype.lower.<op>.<target>.<type>
 *        Example: tvm.datatype.lower.add.llvm.myfloat
 */
class DatatypeRegistry {
 public:
  static inline DatatypeRegistry* Global() {
    static DatatypeRegistry inst;
    return &inst;
  }

  void RegisterDatatype(const std::string& type_name, uint8_t type_code, size_t storage_size);

  uint8_t GetTypeCode(const std::string& type_name);

  std::string GetTypeName(uint8_t type_code);

  size_t GetStorageSize(uint8_t type_code);

  inline bool DatatypeRegistered(uint8_t type_code) {
    return code_to_name.find(type_code) != code_to_name.end();
  }
  inline bool DatatypeRegistered(std::string type_name) {
    return name_to_code.find(type_name) != name_to_code.end();
  }

 private:
  // TODO(gus) is there a typedef for the code?
  std::unordered_map<uint8_t, std::string> code_to_name;
  std::unordered_map<std::string, uint8_t> name_to_code;
  std::unordered_map<uint8_t, size_t> code_to_storage_size;
};

const runtime::PackedFunc* GetCastLowerFunc(const std::string& target,
                                            uint8_t type_code,
                                            uint8_t src_type_code);

  // TODO(gus) Could use asserts here
#define DEFINE_GET_LOWER_FUNC_(OP)                                  \
  inline const runtime::PackedFunc* Get##OP##LowerFunc(             \
      const std::string& target, uint8_t type_code) {               \
    return runtime::Registry::Get(                                  \
        "tvm.datatype.lower." + target + "." #OP "." +             \
        DatatypeRegistry::Global()->GetTypeName(type_code));        \
  }

DEFINE_GET_LOWER_FUNC_(Add)
DEFINE_GET_LOWER_FUNC_(Sub)
DEFINE_GET_LOWER_FUNC_(Mul)
DEFINE_GET_LOWER_FUNC_(Div)
DEFINE_GET_LOWER_FUNC_(Mod)
DEFINE_GET_LOWER_FUNC_(Min)
DEFINE_GET_LOWER_FUNC_(Max)
DEFINE_GET_LOWER_FUNC_(EQ)
DEFINE_GET_LOWER_FUNC_(NE)
DEFINE_GET_LOWER_FUNC_(LT)
DEFINE_GET_LOWER_FUNC_(LE)
DEFINE_GET_LOWER_FUNC_(GT)
DEFINE_GET_LOWER_FUNC_(GE)
DEFINE_GET_LOWER_FUNC_(Select)
DEFINE_GET_LOWER_FUNC_(Load)
DEFINE_GET_LOWER_FUNC_(Ramp)
DEFINE_GET_LOWER_FUNC_(Broadcast)
DEFINE_GET_LOWER_FUNC_(Let)
DEFINE_GET_LOWER_FUNC_(Call)
DEFINE_GET_LOWER_FUNC_(Variable)
DEFINE_GET_LOWER_FUNC_(Shuffle)

}  // namespace tvm

#endif
