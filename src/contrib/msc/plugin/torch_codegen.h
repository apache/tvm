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
 * \file src/contrib/msc/plugin/torch_codegen.h
 * \brief Codegen for torch plugin.
 */
#ifndef TVM_CONTRIB_MSC_PLUGIN_TORCH_CODEGEN_H_
#define TVM_CONTRIB_MSC_PLUGIN_TORCH_CODEGEN_H_

#include <set>
#include <string>

#include "base_codegen.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen config for torch plugin
 */
struct TorchPluginCodeGenConfig {
  bool is_training{false};
  std::string torch_prefix{"torch"};
  PLUGIN_CODEGEN_CONFIG_MEMBERS
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "is_training") {
        reader->Read(&is_training);
      } else if (key == "torch_prefix") {
        reader->Read(&torch_prefix);
      } else {
        PLUGIN_CODEGEN_CONFIG_PARSE
      }
    }
  }
};

class TorchPluginCodeGen : public BasePluginCodeGen<TorchPluginCodeGenConfig> {
 public:
  /*!
   * \brief The constructor of TorchPluginCodeGen
   * \param config the options for codegen.
   */
  explicit TorchPluginCodeGen(const std::string& config = "")
      : BasePluginCodeGen<TorchPluginCodeGenConfig>(config) {}

 protected:
  /*! \brief Codegen plugin attr declare*/
  void CodeGenAttrDeclare(const Plugin& plugin) final;

  /*! \brief Codegen plugin attr define*/
  void CodeGenAttrDefine(const Plugin& plugin) final;

  /*! \brief Codegen plugin op declare*/
  void CodeGenOpDeclare(const Plugin& plugin) final;

  /*! \brief Codegen plugin op define*/
  void CodeGenOpDefine(const Plugin& plugin) final;

  /*! \brief Codegen cmake file*/
  void CodeGenCmake(const std::set<String>& devices) final;

  /*! \brief Codegen manager depends*/
  void CodeGenManagerDepends() final;

  /*! \brief Codegen manager methods*/
  void CodeGenManagerMethods() final;

  /*! \brief Codegen manager member for plugin*/
  void CodeGenOpBuilder(const Plugin& plugin) final;

  /*! \brief Codegen convert depends*/
  void CodeGenConvertDepends() final;

  /*! \brief Codegen convert function for plugin*/
  const String CodeGenOpConvert(const Plugin& plugin) final;

 private:
  /*! \brief Codegen malloc for outputs/buffers*/
  void CodeGenMalloc(const Plugin& plugin, const Array<PluginTensor>& tensors,
                     const String& collect);

  /*! \brief Codegen compute*/
  void CodeGenCompute(const Plugin& plugin, const String& device);

  /*! \brief Entry name of torch function*/
  const String EntryName(const Plugin& plugin) {
    std::string lower_name;
    const std::string& name = std::string(plugin->name);
    for (size_t i = 0; i < name.size(); i++) {
      const char& lower_c = tolower(name[i]);
      if (lower_c != name[i] && i > 0) {
        lower_name += "_";
      }
      lower_name += lower_c;
    }
    return lower_name + "_entry";
  }

  /*! \brief Type name in torch*/
  const String ToTorchType(const String& type) {
    if (type == "float") {
      return "double";
    }
    if (IsListType(type)) {
      const auto& ele_type = GetEleType(type);
      return "c10::arrayRef<" + ToTorchType(ele_type) + ">";
    }
    return BasePluginCodeGen<TorchPluginCodeGenConfig>::ToCppType(type);
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_PLUGIN_TORCH_CODEGEN_H_
