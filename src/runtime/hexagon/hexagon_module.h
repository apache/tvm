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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_MODULE_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_MODULE_H_

#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>

#include <array>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Create a Hexagon module from data.
 * \param data          The module data.
 * \param fmt           The format of the data, can be "obj".
 * \param fmap          The function information map of each function.
 * \param asm_str       String with the generated assembly source.
 * \param obj_str       String with the object file data.
 * \param ir_str        String with the disassembled LLVM IR source.
 * \param bc_str        String with the bitcode LLVM IR.
 */
Module HexagonModuleCreate(std::string data, std::string fmt,
                           std::unordered_map<std::string, FunctionInfo> fmap, std::string asm_str,
                           std::string obj_str, std::string ir_str, std::string bc_str);

/*!
  \brief Module implementation for compiled Hexagon binaries. It is suitable
         for managing cross-compiled Hexagon code on a host machine.
         See docstring for HexagonModuleCreate for
         construction parameter details.
 */
class HexagonModuleNode : public runtime::ModuleNode {
 public:
  HexagonModuleNode(std::string data, std::string fmt,
                    std::unordered_map<std::string, FunctionInfo> fmap, std::string asm_str,
                    std::string obj_str, std::string ir_str, std::string bc_str);
  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override;
  String GetSource(const String& format) override;
  const char* type_key() const final { return "hexagon"; }
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const override {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kDSOExportable |
           ModulePropertyMask::kRunnable;
  }
  void SaveToFile(const String& file_name, const String& format) override;
  void SaveToBinary(dmlc::Stream* stream) override;

 protected:
  std::string data_;
  std::string fmt_;
  std::unordered_map<std::string, FunctionInfo> fmap_;
  std::string asm_;
  std::string obj_;
  std::string ir_;
  std::string bc_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_MODULE_H_
