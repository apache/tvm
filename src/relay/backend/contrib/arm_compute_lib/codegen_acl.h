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
 * \file src/relay/backend/contrib/arm_compute_lib/codegen_acl.h
 * \brief The Relay -> ACL JSON schema compiler.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_ARM_COMPUTE_LIB_CODEGEN_ACL_H_
#define TVM_RELAY_BACKEND_CONTRIB_ARM_COMPUTE_LIB_CODEGEN_ACL_H_

#include <tvm/relay/expr_functor.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace arm_compute_lib {

/*!
 * \brief Generates an ACLModule from a relay expression. This "compilation"
 * does not require ACL since the actual conversion using ACL APIs is
 * deferred until creation of the runtime. This step simply serializes the
 * relay program into a JSON string.
 */
class ACLJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  ACLJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override;
  std::vector<JSONGraphNodeEntry> VisitExpr_(const ConstantNode* cn) override;

  /*!
   * \brief Get the constant data transposed when pre-processing the
   * input function.
   *
   * \return An array of constants
   */
  Array<runtime::NDArray> GetParamsData();

 private:
  /*!
   * \brief Create a JSON representation of an operator.
   *
   * \param call The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateOpJSONNode(const CallNode* cn);
  std::shared_ptr<JSONGraphNode> CreateCompositeConvJSONNode(const CallNode* cn);

  /* \brief Transposed constant tensors to serialize. Arm Compute Library expects constant tensors
   * in OHWI format. */
  Array<runtime::NDArray> constants_;
};

/*!
 * \brief Pre-process a module containing functions ready for ACL codegen.
 *
 * For now we enforce OHWI kernel layout and fold the transforms away.
 *
 * \param mod The module to be pre-processed.
 * \return The processed module.
 */
IRModule PreProcessModule(const IRModule& mod);

/*!
 * \brief Create a runtime module for ACL.
 *
 * This consists of a series of "serialized functions" which each represent a
 * sub-graph to be computed by ACL and will each be executed independently from
 * one another. Each function consists of serialized JSON describing the sub-graph
 * and serialized constant tensors.
 *
 * \note The ACL runtime module only currently supports a single operator per
 * sub-graph currently.
 *
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module ACLCompiler(const ObjectRef& ref);

/*!
 * \brief Check whether ACL graph runtime is used.
 * \return True if ACL graph runtime is enabled, False if not.
 */
inline constexpr bool IsACLRuntimeEnabled();

}  // namespace arm_compute_lib
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ARM_COMPUTE_LIB_CODEGEN_ACL_H_
