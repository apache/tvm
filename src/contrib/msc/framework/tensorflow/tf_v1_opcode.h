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
 * \file src/contrib/msc/framework/tensorflow/tf_v1_opcode.h
 * \brief Tensorflow codegen for MSCJoint, use v1 format.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TENSORFLOW_TF_V1_OPCODE_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TENSORFLOW_TF_V1_OPCODE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../core/codegen/base_codegen.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

class TFV1OpCode;
typedef OpCodeStack<TFV1OpCode> TFV1OpCodeStack;

/*!
 * \brief CodeGen for tensorflow op
 */
class TFV1OpCode : public BaseOpCode<TensorflowCodeGenConfig, TFV1CodeGenHelper> {
 public:
  /*!
   * \brief The constructor of BaseOpDocsifier
   * \param func_name the function name for the node.
   * \param config the config json for the node.
   */
  explicit TFV1OpCode(const String& func_name)
      : BaseOpCode<TensorflowCodeGenConfig, TFV1CodeGenHelper>(func_name) {}

  /*! \brief Convert node to docs*/
  const Array<Doc> GetDocs() final;

  /*! \brief Get dtype string*/
  const String DType(const DataType& dtype) final {
    return "tf_v1." + BaseOpCode<TensorflowCodeGenConfig, TFV1CodeGenHelper>::DType(dtype);
  }

 protected:
  TFV1OpCodeStack stack_;

  /*! \brief Convert op build*/
  virtual void CodeGenBuild() = 0;

  /*! \brief Get padding mode or array*/
  const std::pair<String, Array<String>> GetPadding(const String& strides_key,
                                                    const String& kernel_key = "",
                                                    const String& padding_key = "padding");
};

/*!
 * \brief Get the map of available TFV1OpCode, use optype as key
 * \return Map of <string, TFV1OpCode>
 */
const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TFV1OpCode>>> GetTFV1OpCodes();

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORFLOW_TF_V1_OPCODE_H_
