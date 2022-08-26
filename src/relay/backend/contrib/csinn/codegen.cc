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
 * \file src/relay/backend/contrib/csinn/codegen.cc
 * \brief Implementation of CSINN codegen APIs.
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/tir/analysis.h>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

class CSINNJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  CSINNJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    if (cn->op.as<OpNode>()) {
      return JSONSerializer::VisitExpr_(cn);
    }
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "CSINN JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "CSINN JSON runtime only supports composite functions.";
    name = comp.value();

    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "csinn.conv2d") {
      json_node = CreateCompositeConvJSONNode(cn);
    } else {
      LOG(FATAL) << "Unrecognized CSINN pattern: " << name;
    }
    return AddNode(json_node, GetRef<Expr>(cn));
  }

 private:
  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports both nn.conv2d and qnn.conv2d.
   */
  struct CompositeConvNode {
    const CallNode* conv = nullptr;
    const CallNode* bias = nullptr;
  };

  /*!
   * \brief Extract convolution nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeConvNode UnpackCompositeConvolution(const CallNode* cn) {
    CompositeConvNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);

    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.bias_add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    nodes.conv = current_call;

    return nodes;
  }

  /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeConvJSONNode(const CallNode* cn) {
    CompositeConvNode nodes = UnpackCompositeConvolution(cn);

    const auto* conv_attr = nodes.conv->attrs.as<Conv2DAttrs>();
    ICHECK(conv_attr);

    std::string name;
    std::string name_prefix = "csinn";

    // Distinguish between normal and depth-wise convolution
    if (conv_attr->channels.defined() &&
        tvm::tir::ExprDeepEqual()(conv_attr->channels, conv_attr->groups) &&
        conv_attr->groups != 1) {
      name = "depthwise_conv2d";
    } else {
      name = "conv2d";
    }

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);

    auto json_node = std::make_shared<JSONGraphNode>(name_prefix + "." + name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);

    return json_node;
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module CSINNCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  CSINNJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.const_names();

  const auto* pf = runtime::Registry::Get("runtime.CSINNJSONRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}
TVM_REGISTER_GLOBAL("relay.ext.csinn").set_body_typed(CSINNCompiler);

inline constexpr bool IsCSINNRuntimeEnabled() { return true; }

TVM_REGISTER_GLOBAL("relay.op.is_csinn_runtime_enabled").set_body_typed(IsCSINNRuntimeEnabled);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
