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

/*!
 * \file src/contrib/msc/plugin/tvm_codegen.h
 * \brief Codegen for tvm plugin.
 */
#ifndef TVM_CONTRIB_MSC_PLUGIN_TVM_CODEGEN_H_
#define TVM_CONTRIB_MSC_PLUGIN_TVM_CODEGEN_H_

#include <set>
#include <string>

#include "base_codegen.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen config for tvm plugin
 */
struct TVMPluginCodeGenConfig {
  bool as_relay{false};
  std::string tvm_root{"tvm"};
  PLUGIN_CODEGEN_CONFIG_MEMBERS
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "as_relay") {
        reader->Read(&as_relay);
      } else if (key == "tvm_root") {
        reader->Read(&tvm_root);
      } else {
        PLUGIN_CODEGEN_CONFIG_PARSE
      }
    }
  }
};

class TVMPluginCodeGen : public BasePluginCodeGen<TVMPluginCodeGenConfig> {
 public:
  /*!
   * \brief The constructor of TVMPluginCodeGen
   * \param config the options for codegen.
   */
  explicit TVMPluginCodeGen(const std::string& config = "")
      : BasePluginCodeGen<TVMPluginCodeGenConfig>(config) {}

 protected:
  /*! \brief Codegen plugin attr declare*/
  void CodeGenAttrDeclare(const Plugin& plugin) final;

  /*! \brief Codegen plugin attr define*/
  void CodeGenAttrDefine(const Plugin& plugin) final;

  /*! \brief Codegen plugin op declare*/
  void CodeGenOpDeclare(const Plugin& plugin) final;

  /*! \brief Codegen plugin op define*/
  void CodeGenOpDefine(const Plugin& plugin) final;

  /*! \brief Codegen plugin runtime*/
  void CodeGenOpRuntime(const Plugin& plugin) final;

  /*! \brief Codegen cmake file*/
  void CodeGenCmake(const std::set<String>& devices) final;

  /*! \brief Codegen manager depends*/
  void CodeGenManagerDepends() final;

  /*! \brief Codegen manager methods*/
  void CodeGenManagerMethods() final;

  /*! \brief Codegen manager member for plugin*/
  void CodeGenOpBuilder(const Plugin& plugin) final;

 private:
  /*! \brief Func name of compute*/
  const String ComputeName(const Plugin& plugin) { return plugin->name + "_compute"; }

  /*! \brief Codegen compute*/
  void CodeGenCompute(const Plugin& plugin, const String& device);

  /*! \brief Type name in tvm*/
  const String ToTVMType(const String& type) {
    if (type == "string") {
      return "StringImm";
    }
    if (StringUtils::StartsWith(type, "float")) {
      return "FloatImm";
    }
    if (type == "bool" || StringUtils::StartsWith(type, "int")) {
      return "IntImm";
    }
    if (IsListType(type)) {
      return "Tuple";
    }
    return BasePluginCodeGen<TVMPluginCodeGenConfig>::ToCppType(type);
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_PLUGIN_TVM_CODEGEN_H_
