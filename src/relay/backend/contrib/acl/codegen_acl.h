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
 * \file src/relay/backend/contrib/acl/codegen_acl.h
 * \brief The Relay -> ACL JSON schema compiler.
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_ACL_CODEGEN_ACL_H_
#define TVM_RELAY_BACKEND_CONTRIB_ACL_CODEGEN_ACL_H_

#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/relay/expr_functor.h>

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "acl_api.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace acl {

/*!
 * \brief Generates an ACLModule from a relay expression. This "compilation"
 * does not require ACL since the actual conversion using ACL APIs is
 * deferred until creation of the runtime. This step simply serializes the
 * relay program into a JSON string.
 */
class CodegenACL : public MixedModeVisitor {
 public:
  CodegenACL() = default;
  void VisitExpr_(const CallNode* node) final;
  void VisitLeaf(const Expr& expr) final;

  /*!
   * \brief Create a runtime module for ACL.
   *
   * This consists of a series of "serialized functions" which each represent a
   * subgraph to be computed by ACL and will each be executed independently from
   * one another. Each function consists of serialized JSON describing the subgraph
   * and serialized constant tensors.
   *
   * \note The ACL runtime module only currently supports a single operator per
   * subgraph currently.
   *
   * \param ref The ext_func Relay expression/module to be executed using extern ops.
   * \return A runtime module.
   */
  runtime::Module CreateRuntimeModule(const ObjectRef& ref);

  /*!
   * \brief Create a JSON representation of a subgraph.
   *
   * \param func The function to be represented.
   * \return A JSON representation of the function.
   */
  JSONSubGraph CreateJSONSubgraph(const Function& func);

 private:
  /*!
   * \brief Serialize a single subgraph which can be saved to disk.
   *
   * A subgraph is serialized so that the output is as follows.
   * - Serialized JSON.
   * - Number of constant tensors.
   * - Serialized constant tensors.
   *
   * \param ref Reference to the function to be serialized.
   * \param serialized_functions A vector of serialized functions to add to.
   */
  void SerializeFunction(const ObjectRef& ref,
                         std::vector<std::pair<std::string, std::string>>* serialized_functions);

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
   * \brief Create a JSON representation of an operator.
   *
   * \param call The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  JSONOp MakeConvolutionOp(const Call& call);
  static JSONOp MakeMaxPool2DOp(const Call& call);
  static JSONOp MakeReshapeOp(const Call& call);

  /*!
   * \brief Make a JSON representation of a (constant)tensor.
   *
   * \param expr Expression of a tensor to be represented.
   * \return A JSON representation of a tensor.
   */
  static JSONTensor MakeJSONTensor(const Expr& expr);
  JSONTensor MakeJSONConstTensor(const Expr& expr);

  /*!
   * \brief Check whether CallNode is a composite function and has the same
   * op_name.
   *
   * \param call The current call node.
   * \param op_name The expected name of the call node to check.
   * \return True if the call node is composite and has the same name as
   * op_name, False otherwise.
   */
  bool IsAclFunc(const CallNode* call, const std::string& op_name) const;

  /*!
   * \brief Get composite expression from call node.
   *
   * \param call The call node to get expression from.
   * \return Expression for composite function.
   */
  static Expr GetCompositeExpr(const Call& call);

  /*!
   * \brief Convert a relay array to std::vector.
   *
   * \param array A relay array to be converted.
   * \return std::vector.
   */
  static std::vector<int> ToVector(const Array<IndexExpr>& array);

  /*!
   * \brief Create a padding vector compatible with ACL.
   *
   * Currently TVM has many ways to pad a an operator, so each method is taken care of here.
   *
   * \param pad Padding array.
   * \return ACL compatible padding vector.
   */
  static std::vector<int> GetPadVector(const Array<Array<IndexExpr>>& pad);
  static std::vector<int> GetPadVector(const Array<IndexExpr>& pad);

  /*! \brief A vector of constants to be serialized after the JSON representation is constructed. */
  std::vector<runtime::NDArray> constants_;
  /*! \brief A look-up table from Expr to JSONOp. */
  std::map<Expr, JSONOp> layer_table_;
};

/*!
 * \brief The external ACL compiler/codegen tool. It takes a Relay
 * expression/module and compiles it into a runtime module.
 */
runtime::Module ACLCompiler(const ObjectRef& ref) {
  CodegenACL acl_codegen;
  return acl_codegen.CreateRuntimeModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.acl").set_body_typed(ACLCompiler);

/*!
 * \brief Check whether ACL graph runtime is used.
 * \return True if ACL graph runtime is enabled, False if not.
 */
inline constexpr bool IsACLRuntimeEnabled() {
#if TVM_GRAPH_RUNTIME_ACL
  return true;
#else
  return false;
#endif
}

TVM_REGISTER_GLOBAL("relay.op.is_acl_runtime_enabled").set_body_typed(IsACLRuntimeEnabled);

}  // namespace acl
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_ACL_CODEGEN_ACL_H_
