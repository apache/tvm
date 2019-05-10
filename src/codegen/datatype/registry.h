/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
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
 *      For Casts: tvm.datatype.lower.<target>.Cast.<type>.<src_type>
 *        Example: tvm.datatype.lower.llvm.Cast.myfloat.float for a Cast from
 *                 float to myfloat.
 *  For other ops: tvm.datatype.lower.<target>.<op>.<type>
 *       Examples: tvm.datatype.lower.llvm.Add.myfloat
 *                 tvm.datatype.lower.llvm.FloatImm.posit
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
    return code_to_name_.find(type_code) != code_to_name_.end();
  }
  inline bool GetTypeRegistered(std::string type_name) {
    return name_to_code_.find(type_name) != name_to_code_.end();
  }

 private:
  // TODO(gus) is there a typedef for the code?
  std::unordered_map<uint8_t, std::string> code_to_name_;
  std::unordered_map<std::string, uint8_t> name_to_code_;
};

// For the custom datatype specified by type_code, convert the value to this
// datatype and return the bits within a uint64_t.
uint64_t ConvertConstScalar(uint8_t type_code, double value);

const runtime::PackedFunc *GetCastLowerFunc(const std::string &target,
                                            uint8_t type_code,
                                            uint8_t src_type_code);

const runtime::PackedFunc* GetFloatImmLowerFunc(const std::string& target, uint8_t type_code);

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
// Later changes may need to add more lowering functions as we support workloads with more ops.

}  // namespace datatype
}  // namespace tvm

#endif  // TVM_CODEGEN_DATATYPE_REGISTRY_H_
