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
 * \file src/contrib/msc/plugin/tensorrt_codegen.h
 * \brief Codegen for tensorrt plugin.
 */
#ifndef TVM_CONTRIB_MSC_PLUGIN_TENSORRT_CODEGEN_H_
#define TVM_CONTRIB_MSC_PLUGIN_TENSORRT_CODEGEN_H_

#include <set>
#include <string>

#include "base_codegen.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

/*!
 * \brief CodeGen config for tensorrt plugin
 */
struct TensorRTPluginCodeGenConfig {
  std::string tensorrt_root{"/usr/local/cuda"};
  PLUGIN_CODEGEN_CONFIG_MEMBERS
  void Load(dmlc::JSONReader* reader) {
    std::string key;
    reader->BeginObject();
    while (reader->NextObjectItem(&key)) {
      if (key == "tensorrt_root") {
        reader->Read(&tensorrt_root);
      } else {
        PLUGIN_CODEGEN_CONFIG_PARSE
      }
    }
  }
};

class TensorRTPluginCodeGen : public BasePluginCodeGen<TensorRTPluginCodeGenConfig> {
 public:
  /*!
   * \brief The constructor of TensorRTPluginCodeGen
   * \param config the options for codegen.
   */
  explicit TensorRTPluginCodeGen(const std::string& config = "")
      : BasePluginCodeGen<TensorRTPluginCodeGenConfig>(config) {}

 protected:
  /*! \brief Codegen plugin attr declare*/
  void CodeGenAttrDeclare(const Plugin& plugin) final;

  /*! \brief Codegen plugin attr define*/
  void CodeGenAttrDefine(const Plugin& plugin) final;

  /*! \brief Header of plugin files*/
  void CodeGenOpHeader(const Plugin& plugin) final;

  /*! \brief Codegen plugin op declare*/
  void CodeGenOpDeclare(const Plugin& plugin) final;

  /*! \brief Codegen plugin op define*/
  void CodeGenOpDefine(const Plugin& plugin) final;

  /*! \brief Codegen cmake file*/
  void CodeGenCmake(const std::set<String>& devices) final;

  /*! \brief Codegen manager methods*/
  void CodeGenManagerMethods() final;

 private:
  /*! \brief Op class name of plugin*/
  const String OpCls(const Plugin& plugin, bool dynamic) const {
    return plugin->name + (dynamic ? "DynamicPlugin" : "Plugin");
  }

  /*! \brief Creator class name of plugin*/
  const String CreatorCls(const Plugin& plugin, bool dynamic) const {
    return plugin->name + (dynamic ? "DynamicCreator" : "Creator");
  }

  bool IsMixPrecision(const Plugin& plugin) {
    for (const auto& dtypes : GetDtypeMatrix(plugin)) {
      String ref_dtype = "";
      for (const auto& pair : dtypes) {
        if (ref_dtype.size() == 0) {
          ref_dtype = pair.second;
        } else if (ref_dtype != pair.second) {
          return true;
        }
      }
    }
    return false;
  }

  /*! \brief codegen plugin op common methods declare*/
  void CodegenOpCommonMethods(const Plugin& plugin, bool dynamic, bool in_declare);

  /*! \brief codegen plugin op members define*/
  void CodegenOpMembers(const Plugin& plugin, bool dynamic);

  /*! \brief codegen plugin creator*/
  void CodegenCreator(const Plugin& plugin, bool dynamic, bool in_declare);

  /*! \brief codegen infer output func*/
  void CodegenOutputInfer(const Plugin& plugin, bool as_desc = false);

  /*! \brief codegen infer buffer func*/
  void CodegenBufferInfer(const Plugin& plugin);

  /*! \brief codegen enqueue func*/
  void CodegenEnqueue(const Plugin& plugin, bool dynamic);
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_PLUGIN_TENSORRT_CODEGEN_H_
