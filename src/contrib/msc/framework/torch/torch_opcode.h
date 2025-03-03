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
 * \file src/contrib/msc/framework/torch/torch_opcode.h
 * \brief Torch codegen for MSCJoint.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TORCH_TORCH_OPCODE_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TORCH_TORCH_OPCODE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../core/codegen/base_codegen.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

class TorchOpCode;
typedef OpCodeStack<TorchOpCode> TorchOpCodeStack;

/*!
 * \brief CodeGen for torch op
 */
class TorchOpCode : public BaseOpCode<TorchCodeGenConfig, TorchCodeGenHelper> {
 public:
  /*!
   * \brief The constructor of BaseOpDocsifier
   * \param func_name the function name for the node.
   * \param config the config json for the node.
   */
  explicit TorchOpCode(const String& module_name, const String& func_name)
      : BaseOpCode<TorchCodeGenConfig, TorchCodeGenHelper>(func_name) {
    module_name_ = module_name;
  }

  /*! \brief Config the TorchOpCode*/
  void Config(const MSCJoint& node, const std::shared_ptr<TorchCodeGenConfig> config, bool is_init,
              const Map<String, String>& prims) {
    BaseOpCode<TorchCodeGenConfig, TorchCodeGenHelper>::Config(node, config, prims);
    is_init_ = is_init;
    module_ref_ = "self." + StringUtils::Replace(node->name, ".", "_");
  }

  /*! \brief Get return describe for default node*/
  const String IdxNode() final {
    return is_init_ ? module_ref_ : BaseOpCode<TorchCodeGenConfig, TorchCodeGenHelper>::IdxNode();
  };

  /*! \brief Get dtype string*/
  const String DType(const DataType& dtype) final {
    return "torch." + BaseOpCode<TorchCodeGenConfig, TorchCodeGenHelper>::DType(dtype);
  }

  /*! \brief Get func_name for the default node*/
  const String callee_name() final {
    if (is_init_) {
      return module_name_;
    }
    if (module_name_.size() > 0) {
      return module_ref_;
    }
    return BaseOpCode<TorchCodeGenConfig, TorchCodeGenHelper>::callee_name();
  }

  /*! \brief Convert node to docs*/
  const Array<Doc> GetDocs() final;

 protected:
  TorchOpCodeStack stack_;

  /*! \brief Convert op build*/
  virtual void CodeGenInit();

  /*! \brief Convert op build*/
  virtual void CodeGenForward();

  /*! \brief Get the padding from op*/
  const StrictListDoc GetPadding(const String& key = "padding");

  /*! \brief Get the is_init_ of codegen*/
  bool is_init() { return is_init_; }

  /*! \brief Get the module_name of codegen*/
  const String module_name() { return module_name_; }

  /*! \brief Get the module_ref of codegen*/
  const String module_ref() { return module_ref_; }

 private:
  bool is_init_;
  String module_name_;
  String module_ref_;
};

/*!
 * \brief Get the map of available TorchOpCode, use optype as key
 * \return Map of <string, TorchOpCode>
 */
const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TorchOpCode>>> GetTorchOpCodes();

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TORCH_TORCH_OPCODE_H_
