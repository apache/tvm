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
 * \file src/relax/backend/contrib/adreno_accl/codegen.cc
 * \brief Implementation of the Adreno_ACCL JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/relax/type.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../transform/utils.h"
#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using OpAttrExtractor = backend::contrib::OpAttrExtractor;
using JSONSerializer = backend::contrib::JSONSerializer;

class AdrenoACCLJSONSerializer;

/*!
 * \brief Collect the constants and attributes from all operator calls in the body
 * of a "Composite" function.
 */
class CollectFromCompositeFunctionBody : public ExprVisitor {
 public:
  explicit CollectFromCompositeFunctionBody(AdrenoACCLJSONSerializer* serializer)
      : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const ConstantNode* constant_node) final;
  void VisitExpr_(const CallNode* call_node) final;

  void SetGenericAttributes(const CallNode* call_node) {
    OpAttrExtractor extractor(node_);
    const Object* attr_obj = call_node->attrs.get();
    extractor.Extract(const_cast<Object*>(attr_obj));
  }

  AdrenoACCLJSONSerializer* serializer_;
  /*! \brief Accumulated translated arguments. */
  std::vector<JSONGraphNodeEntry> args_;
  /*!
   * \brief Temporary node into which we'll accumulate attributes. Ideally this would be the
   * final JSONGraphNode however we don't yet know how many inputs that will have.
   */
  JSONGraphObjectPtr node_;
};

/*!
 * \brief Generates an AdrenoACCL Module from a relax expression by serializing the expression to a
 * json representation. AdrenoACCL is not required here because use of AdrenoACCL APIs is deferred
 * until runtime.
 */
class AdrenoACCLJSONSerializer : public JSONSerializer {
 public:
  explicit AdrenoACCLJSONSerializer(Map<Constant, String> constant_names, Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* call_node) final {
    // The call must be to an inline "Composite" function
    const auto* fn_var = call_node->op.as<VarNode>();
    ICHECK(fn_var);
    const auto fn = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);

    auto opt_composite = fn->GetAttr<String>(attr::kComposite);
    ICHECK(opt_composite.defined());
    std::string name = opt_composite.value();

    std::shared_ptr<JSONGraphNode> node;

    // Collect the constants and attributes of all operator calls inside the composite body.
    CollectFromCompositeFunctionBody collector(this);
    collector.VisitExpr(fn->body);

    // Capture the args to the "Composite" function as inputs for this node.
    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    // Capture constants from the composite function body as additional inputs for this node.
    for (const auto& node : collector.args_) {
      inputs.emplace_back(node);
    }

    // Create the final node.
    node = std::make_shared<JSONGraphNode>(name,
                                           /*op_type=*/"kernel", inputs,
                                           /*num_output=*/1);

    // Transfer attributes from the collector's node to the final node.
    node->CaptureAttrs(*collector.node_);

    VLOG(1) << name << " has " << node->GetInputs().size() << " inputs";

    return AddNode(node, GetRef<Expr>(call_node));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  Map<Var, Expr> bindings_;
};

void CollectFromCompositeFunctionBody::VisitExpr_(const ConstantNode* constant_node) {
  for (const auto& entry : serializer_->VisitExpr(GetRef<Constant>(constant_node))) {
    args_.emplace_back(entry);
  }
}

void CollectFromCompositeFunctionBody::VisitExpr_(const CallNode* call_node) {
  SetGenericAttributes(call_node);
  ExprVisitor::VisitExpr_(call_node);
}

/*!
 * \brief Create runtime modules for AdrenoACCL.
 * \param functions The extern functions to be compiled via AdrenoACCL
 * \return Runtime modules.
 */
Array<runtime::Module> AdrenoACCLCompiler(Array<Function> functions,
                                          Map<String, ObjectRef> /*unused*/,
                                          Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;
  for (const auto& func : functions) {
    VLOG(1) << "AdrenoACCLCompiler partition:" << std::endl << func;
    AdrenoACCLJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    std::string graph_json = serializer.GetJSON();
    VLOG(1) << "AdrenoACCLCompiler JSON:" << std::endl << graph_json;
    auto constant_names = serializer.GetConstantNames();
    const auto* pf = runtime::Registry::Get("runtime.adreno_accl_runtime_create");
    ICHECK(pf != nullptr) << "Cannot find AdrenoACCLCompiler runtime module create function.";
    std::string func_name = GetExtSymbol(func);
    VLOG(1) << "Creating AdrenoACCL runtime::Module for '" << func_name << "'";
    compiled_functions.push_back((*pf)(func_name, graph_json, constant_names));
  }
  return compiled_functions;
}

TVM_REGISTER_GLOBAL("relax.ext.adreno_accl").set_body_typed(AdrenoACCLCompiler);

/*!
 * \brief Check whether AdrenoACCL graph executor is enabled.
 * \return True if enabled, False if not.
 */
inline constexpr bool IsAdrenoACCLRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_ADRENO_ACCL
  return true;
#else
  return false;
#endif  // TVM_GRAPH_EXECUTOR_ADRENO_ACCL
}

TVM_REGISTER_GLOBAL("relax.is_adreno_accl_runtime_enabled")
    .set_body_typed(IsAdrenoACCLRuntimeEnabled);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
