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
 * \file src/contrib/msc/framework/tensorrt/tensorrt_opcode.h
 * \brief TensorRT codegen for MSCJoint.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_TENSORRT_OPCODE_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_TENSORRT_OPCODE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../core/codegen/base_codegen.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

class TensorRTOpCode;
typedef OpCodeStack<TensorRTOpCode> TensorRTOpCodeStack;

/*!
 * \brief CodeGen for relax op
 */
class TensorRTOpCode : public BaseOpCode<TensorRTCodeGenConfig, TensorRTCodeGenHelper> {
 public:
  /*!
   * \brief The constructor of BaseOpDocsifier
   * \param func_name the function name for the node.
   * \param config the config json for the node.
   */
  explicit TensorRTOpCode(const String& func_name)
      : BaseOpCode<TensorRTCodeGenConfig, TensorRTCodeGenHelper>(func_name) {}

  /*! \brief Convert node to docs*/
  const Array<Doc> GetDocs() final;

  /*! \brief Get func_name for the default node*/
  const String callee_name() final {
    return "network->add" + BaseOpCode<TensorRTCodeGenConfig, TensorRTCodeGenHelper>::callee_name();
  }

  /*! \brief Get valid return name for the default node*/
  const String ret_name() final { return "auto " + IdxNode(); }

  /*! \brief Get the dtype from the datatype*/
  const String DType(const DataType& dtype) final;

 protected:
  TensorRTOpCodeStack stack_;

  /*! \brief Convert op build*/
  virtual void CodeGenBuild() = 0;

  /*! \brief Set padding for the layer*/
  void SetPadding(const String& key = "padding");

  /*! \brief Declare the inputs*/
  const String DeclareInputs(bool simplify = true);

  /*! \brief Get the tensorrt dims from dims*/
  template <typename T>
  const String ToDims(const std::vector<T>& dims, bool use_ndim = true);
  const String ToDims(const Array<Integer>& dims, bool use_ndim = true);

  /*! \brief Get the tensorrt dims from attribute*/
  const String AttrToDims(const String& key, bool use_ndim = true);

  /*! \brief Get the tensorrt reduce axis from dims*/
  const size_t ToReduceAxis(const std::vector<int>& axes, size_t ndim = 0);

  /*! \brief Get the tensorrt reduce axis from attribute*/
  const size_t AttrToReduceAxis(const String& key = "axis", size_t ndim = 0);

  /*! \brief Get the attribute axis from attribute*/
  const size_t AttrToAxis(const String& key = "axis", size_t ndim = 0);

  /*! \brief Set layer by attribute*/
  template <typename T>
  void SetLayerByAttr(const String& method, const String& key);

  /*! \brief Set layer by value*/
  template <typename T>
  void SetLayerByValue(const String& method, const T& value);

  /*! \brief Set layer by dims attribute*/
  void SetLayerByDimsAttr(const String& method, const String& key, bool use_ndim = true);

  /*! \brief Set layer by dims value*/
  template <typename T>
  void SetLayerByDimsValue(const String& method, const std::vector<T>& value, bool use_ndim = true);
  void SetLayerByDimsValue(const String& method, const Array<Integer>& value, bool use_ndim = true);
};

/*!
 * \brief Get the map of available TensorRTOpCode, use optype as key
 * \return Map of <string, TensorRTOpCode>
 */
const std::shared_ptr<std::unordered_map<String, std::shared_ptr<TensorRTOpCode>>>
GetTensorRTOpCodes();

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_TENSORRT_OPCODE_H_
