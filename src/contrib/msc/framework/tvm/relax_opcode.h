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
 * \file src/contrib/msc/framework/tvm/relax_opcode.h
 * \brief Relax codegen for MSCJoint.
 */
#ifndef TVM_CONTRIB_MSC_FRAMEWORK_TVM_RELAX_OPCODE_H_
#define TVM_CONTRIB_MSC_FRAMEWORK_TVM_RELAX_OPCODE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../core/codegen/base_codegen.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

class RelaxOpCode;
typedef OpCodeStack<RelaxOpCode> RelaxOpCodeStack;

/*!
 * \brief CodeGen for relax op
 */
class RelaxOpCode : public BaseOpCode<RelaxCodeGenConfig, RelaxCodeGenHelper> {
 public:
  /*!
   * \brief The constructor of BaseOpDocsifier
   * \param func_name the function name for the node.
   * \param config the config json for the node.
   */
  explicit RelaxOpCode(const String& func_name)
      : BaseOpCode<RelaxCodeGenConfig, RelaxCodeGenHelper>(func_name) {}

  /*! \brief Convert node to docs*/
  const Array<Doc> GetDocs() final;

 protected:
  RelaxOpCodeStack stack_;

  /*! \brief Convert op build*/
  virtual void CodeGenBuild() = 0;

  /*! \brief coda stack emit docs*/
  void BuilderEmit(const String& ret, const String& name = "");

  /*! \brief Get the out_dtype attribute*/
  const ExprDoc GetOutDtype(const String& key = "out_dtype", int input_idx = 0);

  /*! \brief Get the axes attribute*/
  const std::vector<int> GetAxes(const String& key = "axes");
};

/*!
 * \brief Get the map of available RelaxOpCode, use optype as key
 * \return Map of <string, RelaxOpCode>
 */
const std::shared_ptr<std::unordered_map<String, std::shared_ptr<RelaxOpCode>>> GetRelaxOpCodes();

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_FRAMEWORK_TVM_RELAX_OPCODE_H_
