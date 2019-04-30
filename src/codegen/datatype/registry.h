/*!
 *  Copyright (c) 2019 by Contributors
 * \file tvm/src/codegen/datatype/registry.h
 * \brief Custom datatypes registry
 */

#ifndef TVM_CODEGEN_DATATYPE_REGISTRY_H_
#define TVM_CODEGEN_DATATYPE_REGISTRY_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <string>
#include <unordered_map>

namespace tvm {
namespace datatype {

/*!
 * \brief Registry for custom datatypes.
 *
 * Adding custom datatypes currently requires two steps:
 * 1. Register the datatype with the registry via a call to
 *    datatype::Registry::Register. This can also be done in Python
 *    directly---see the TVM globals registered in the corresponding .cc file.
 *    Currently, user should manually choose a type name and a type code,
 *    ensuring that neither conflict with existing types.
 * 2. Use TVM_REGISTER_GLOBAL to register the lowering functions needed to
 *    lower the custom datatype. In general, these will look like:
 *      For Casts: tvm.datatype.lower.Cast.<target>.<type>.<src_type>
 *        Example: tvm.datatype.lower.Cast.llvm.myfloat.float for a Cast from
 *                 float to myfloat.
 *        Example: tvm.datatype.lower.add.llvm.myfloat
 *  For other ops: tvm.datatype.lower.<op>.<target>.<type>
 */
class Registry {
 public:
  static inline Registry *Global() {
    static Registry inst;
    return &inst;
  }

  void Register(const std::string &type_name, uint8_t type_code);

  uint8_t GetTypeCode(const std::string &type_name);

  std::string GetTypeName(uint8_t type_code);

  inline bool GetTypeRegistered(uint8_t type_code) {
    return code_to_name.find(type_code) != code_to_name.end();
  }
  inline bool GetTypeRegistered(std::string type_name) {
    return name_to_code.find(type_name) != name_to_code.end();
  }

 private:
  // TODO(gus) is there a typedef for the code?
  std::unordered_map<uint8_t, std::string> code_to_name;
  std::unordered_map<std::string, uint8_t> name_to_code;
};

// For the custom datatype specified by type_code, convert the value to this
// datatype and return the bits within a uint64_t.
uint64_t ConvertConstScalar(uint8_t type_code, double value);

const runtime::PackedFunc *GetCastLowerFunc(const std::string &target,
                                            uint8_t type_code,
                                            uint8_t src_type_code);

#define DEFINE_GET_LOWER_FUNC_(OP)                                                       \
  inline const runtime::PackedFunc* Get##OP##LowerFunc(const std::string& target,        \
                                                       uint8_t type_code) {              \
    return runtime::Registry::Get("tvm.datatype.lower." + target + "." #OP "." +         \
                                  datatype::Registry::Global()->GetTypeName(type_code)); \
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
// DEFINE_GET_LOWER_FUNC_(Select)
// DEFINE_GET_LOWER_FUNC_(Ramp)
// DEFINE_GET_LOWER_FUNC_(Broadcast)
// DEFINE_GET_LOWER_FUNC_(Let)
// DEFINE_GET_LOWER_FUNC_(Call)
// DEFINE_GET_LOWER_FUNC_(Variable)
// DEFINE_GET_LOWER_FUNC_(Shuffle)

}  // namespace datatype
}  // namespace tvm

#endif  // TVM_CODEGEN_DATATYPE_REGISTRY_H_
