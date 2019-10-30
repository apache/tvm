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
 * \file src/relay/backend/contrib/contrib_codegen.h
 * \brief The base class for external codegen tools.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CONTRIB_CODEGEN_H_
#define TVM_RELAY_BACKEND_CONTRIB_CONTRIB_CODEGEN_H_

#include <tvm/relay/expr.h>
#include <string>
#include "../../../runtime/contrib/extern_common.h"

namespace tvm {
namespace relay {
namespace contrib {

class ExternCodegenBase {
 public:
  ExternCodegenBase() = default;

  /*!
   * \brief Compile the external library.
   */
  virtual void CompileExternLib() = 0;

  /*!
   * \brief Build the shared library of external ops.
   *
   * \param ref The subgraph Relay expression/module to be executed using extern ops.
   *
   */
  virtual void Build(const NodeRef& ref) = 0;

  /*!
   * \brief Split the Relay function name to tokens.
   *
   * \param func The provided function.
   *
   * \return A vector of tokenized function name splitted by "_".
   */
  std::string GetSubgraphID(const Function& func) const {
    const auto name_node =
        FunctionGetAttr(func, "func_name").as<tvm::ir::StringImm>();
    CHECK(name_node != nullptr) << "Fail to retrieve subgraph name.";
    std::string name = name_node->value;
    return runtime::contrib::GetSubgraphID(name);
  }
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CONTRIB_CODEGEN_H_
