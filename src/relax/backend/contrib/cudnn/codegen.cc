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
 * \file src/relax/backend/contrib/cudnn/codegen.cc
 * \brief Implementation of the cuDNN JSON serializer.
 */
#include <tvm/ir/module.h>

#include <string>

#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONSerializer = backend::contrib::JSONSerializer;
using backend::contrib::NodeEntries;

class cuDNNJSONSerializer : public JSONSerializer {
 public:
  cuDNNJSONSerializer(Map<Constant, String> constant_names, Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  NodeEntries VisitExpr_(const CallNode* call_node) final {
    const auto* fn_var = call_node->op.as<VarNode>();
    ICHECK(fn_var);
    const auto fn = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr<String>(attr::kComposite);
    ICHECK(composite_opt.defined()) << "Only composite functions are supported.";

    std::string composite_name = composite_opt.value();

    NodeEntries inputs_tmp;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs_tmp.insert(inputs_tmp.end(), res.begin(), res.end());
    }

    ICHECK(inputs_tmp.size() <= 3);
    NodeEntries inputs(inputs_tmp.size());

    auto arg_idx = backend::ExtractArgIdx(composite_name, fn);
    inputs[0] = inputs_tmp[arg_idx["input"]->value];
    inputs[1] = inputs_tmp[arg_idx["weight"]->value];
    if (inputs_tmp.size() == 3) {
      inputs[2] = inputs_tmp[arg_idx["bias"]->value];
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);

    const CallNode* root_call = backend::GetOpInFunction(fn, "relax.nn.conv2d");
    SetCallNodeAttribute(node, root_call);
    return AddNode(node, GetRef<Expr>(call_node));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  Map<Var, Expr> bindings_;
};

Array<runtime::Module> cuDNNCompiler(Array<Function> functions, Map<String, ObjectRef> /*unused*/,
                                     Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;

  for (const auto& func : functions) {
    cuDNNJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto* pf = runtime::Registry::Get("runtime.cuDNNJSONRuntimeCreate");
    ICHECK(pf != nullptr) << "Cannot find cuDNN runtime module create function.";
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back((*pf)(func_name, graph_json, constant_names));
  }

  return compiled_functions;
}

TVM_REGISTER_GLOBAL("relax.ext.cudnn").set_body_typed(cuDNNCompiler);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
