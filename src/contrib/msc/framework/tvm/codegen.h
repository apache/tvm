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
 * \file src/contrib/msc/framework/tvm/codegen.h
 * \brief Relax codegen for MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TVM_CODEGEN_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TVM_CODEGEN_H_

#include <string>

#include "../../core/codegen/base_codegen.h"
#include "../../core/codegen/py_codegen.h"
#include "codegen_utils.h"
#include "relax_opcode.h"

namespace tvm {
namespace contrib {
namespace msc {

class RelaxCodeGen : public PyCodeGen<RelaxCodeGenConfig, RelaxCodeGenHelper> {
 public:
  /*!
   * \brief The constructor of RelaxCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit RelaxCodeGen(const MSCGraph& graph, const std::string& config = "")
      : PyCodeGen<RelaxCodeGenConfig, RelaxCodeGenHelper>(graph, config) {}

 protected:
  /*! \brief Stack the docs for the header*/
  void CodeGenHeader() final;

  /*! \brief Stack the docs for the graph*/
  void CodeGenGraph() final;

  /*! \brief Stack the docs for the graph inference*/
  void CodeGenInference() final;

  /*! \brief Describe the prim*/
  const String DescribePrim(const MSCPrim& prim) final;

  /*! \brief Get the docs for the op*/
  const Array<Doc> GetOpCodes(const MSCJoint& node) final;

  /*! \brief Get tensor type of the framework*/
  const String TensorType() const final { return "relax.Expr"; }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TVM_CODEGEN_H_
