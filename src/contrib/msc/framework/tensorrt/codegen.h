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
 * \file src/contrib/msc/framework/tensorrt/codegen.h
 * \brief Relax codegen for MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_H_

#include <string>
#include <vector>

#include "../../core/codegen/base_codegen.h"
#include "../../core/codegen/cpp_codegen.h"
#include "codegen_utils.h"
#include "tensorrt_opcode.h"

namespace tvm {
namespace contrib {
namespace msc {

class TensorRTCodeGen : public CppCodeGen<TensorRTCodeGenConfig, TensorRTCodeGenHelper> {
 public:
  /*!
   * \brief The constructor of TensorRTCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit TensorRTCodeGen(const MSCGraph& graph, const std::string& config = "")
      : CppCodeGen<TensorRTCodeGenConfig, TensorRTCodeGenHelper>(graph, config) {}

  /*! \brief Stack the docs for the class declare*/
  void CodeGenClassDeclare() final;

  /*! \brief Stack the docs for the class define*/
  void CodeGenClassDefine() final;

  /*! \brief Stack the docs for the main func*/
  void CodeGenMain() final;

  /*! \brief Stack the docs for the class define*/
  void CodeGenCmake() final;

 protected:
  /*! \brief Get the docs for the op*/
  const Array<Doc> GetOpCodes(const MSCJoint& node) final;

  /*! \brief Get the tensor context for codegen_tensor*/
  const Map<String, String> GetTensorCtx(const MSCTensor& tensor) final;

  /*! \brief Get the step context for codegen_step*/
  const Map<String, String> GetStepCtx() final;

  /*! \brief Generate return on fail codes*/
  void ReturnOnFail(const String& flag, const String& err);

  /*! \brief Get the index tensor*/
  const String IdxTensor(const MSCTensor& tensor);

  /*! \brief Get the dtype from the datatype*/
  const String CppDType(const DataType& dtype);

  /*! \brief Generate describe for tensor bytes*/
  const String GetTensorBytes(const MSCTensor& tensor);

  /*! \brief Get the tensorrt dims from dims*/
  template <typename T>
  const String ToDims(const std::vector<T>& dims, bool use_ndim = true);
  const String ToDims(const Array<Integer>& dims, bool use_ndim = true);

 private:
  Array<String> before_build_codes_;
  Array<String> after_build_codes_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TENSORRT_CODEGEN_H_
