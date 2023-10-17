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
 * \file src/contrib/msc/framework/torch/codegen.h
 * \brief Torch codegen for MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TORCH_CODEGEN_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TORCH_CODEGEN_H_

#include <string>

#include "../../core/codegen/base_codegen.h"
#include "../../core/codegen/py_codegen.h"
#include "codegen_utils.h"
#include "torch_opcode.h"

namespace tvm {
namespace contrib {
namespace msc {

class TorchCodeGen : public PyCodeGen<TorchCodeGenConfig, TorchCodeGenHelper> {
 public:
  /*!
   * \brief The constructor of TorchCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit TorchCodeGen(const MSCGraph& graph, const std::string& config = "")
      : PyCodeGen<TorchCodeGenConfig, TorchCodeGenHelper>(graph, config) {}

 protected:
  /*! \brief Stack the docs for the header*/
  void CodeGenHeader() final;

  /*! \brief Stack the docs for the graph*/
  void CodeGenGraph() final;

  /*! \brief Stack the docs for the graph inference*/
  void CodeGenInference() final;

  /*! \brief Get the docs for the op*/
  const Array<Doc> GetOpCodes(const MSCJoint& node) final;

  /*! \brief Get tensor type of the framework*/
  const String TensorType() const final { return "torch.Tensor"; }

 private:
  bool is_init_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TORCH_CODEGEN_H_
