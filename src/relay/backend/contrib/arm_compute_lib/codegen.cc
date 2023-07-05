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
 * \file src/relay/backend/contrib/arm_compute_lib/codegen.cc
 * \brief Implementation of the Relay -> ACL JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>
#include <tvm/tir/analysis.h>

#include <memory>
#include <string>
#include <vector>

#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

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

  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports both nn.conv2d and qnn.conv2d.
   */
  struct CompositeConvNode {
    const CallNode* pad = nullptr;
    const CallNode* conv = nullptr;
    const CallNode* bias = nullptr;
    const CallNode* activation = nullptr;
    const CallNode* requantize = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * dense layer. Supports both nn.dense and qnn.dense.
   */
  struct CompositeDenseNode {
    const CallNode* dense = nullptr;
    const CallNode* bias = nullptr;
    const CallNode* requantize = nullptr;
  };

  /*!
   * \brief Visit call nodes and generate appropriate JSON node.
   *
   * \param cn The current call node.
   * \return A list of graph entry nodes.
   */
  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    if (cn->op.as<OpNode>()) {
      return JSONSerializer::VisitExpr_(cn);
    }
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "Arm Compute Library JSON runtime does not support calls to "
                 << cn->op->GetTypeKey();
    }
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "Arm Compute Library JSON runtime only supports composite functions.";
    const std::string name = comp.value();
    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "arm_compute_lib.conv2d" || name == "arm_compute_lib.qnn_conv2d") {
      json_node = CreateCompositeConvJSONNode(cn);
    } else if (name == "arm_compute_lib.dense" || name == "arm_compute_lib.qnn_dense") {
      json_node = CreateCompositeDenseJSONNode(cn);
    } else if (name == "arm_compute_lib.avg_pool2d") {
      json_node = CreateCompositeAvgPool2DJSONNode(cn);
    } else if (name == "arm_compute_lib.l2_pool2d") {
      json_node = CreateCompositeL2Pool2DJSONNode(cn);
    } else if (name == "arm_compute_lib.concatenate") {
      return AddCommonSingleJSONNode(cn, "concatenate");
    } else {
      LOG(FATAL) << "Unrecognized Arm Compute Library pattern: " << name;
    }
    return AddNode(json_node, GetRef<Expr>(cn));
  }

 private:
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

    // Traverse composite convolution function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "qnn.requantize")) {
      nodes.requantize = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    // Enforce a convolution node exists at this point during traversal
    if (nodes.requantize) {
      ICHECK(backend::IsOp(current_call, "qnn.conv2d"));
    } else {
      ICHECK(backend::IsOp(current_call, "nn.conv2d"));
    }
    nodes.conv = current_call;
    if (!current_call->args.empty() && current_call->args[0]->IsInstance<CallNode>()) {
      current_call = current_call->args[0].as<CallNode>();
      if (backend::IsOp(current_call, "nn.pad")) {
        nodes.pad = current_call;
      }
    }
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
    std::string name_prefix = "nn";

    // Distinguish between normal and depth-wise convolution
    if (conv_attr->channels.defined() &&
        tvm::tir::ExprDeepEqual()(conv_attr->channels, conv_attr->groups) &&
        conv_attr->groups != 1) {
      name = "depthwise_conv2d";
      ICHECK(conv_attr->kernel_layout == "IHWO")
          << "Kernel layout must be IHWO, has the module been pre-processed correctly?";
    } else {
      name = "conv2d";
      ICHECK(conv_attr->kernel_layout == "OHWI")
          << "Kernel layout must be OHWI, has the module been pre-processed correctly?";
    }

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    if (nodes.requantize) {
      name_prefix = "qnn";
      inputs.push_back(VisitExpr(nodes.conv->args[2])[0]);  // input zero-point
      inputs.push_back(VisitExpr(nodes.conv->args[3])[0]);  // kernel zero-point
      inputs.push_back(VisitExpr(nodes.conv->args[4])[0]);  // input scale
      inputs.push_back(VisitExpr(nodes.conv->args[5])[0]);  // kernel scale
    }
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    if (nodes.requantize) {
      inputs.push_back(VisitExpr(nodes.requantize->args[3])[0]);  // output scale
      inputs.push_back(VisitExpr(nodes.requantize->args[4])[0]);  // output zero-point
    }

    auto json_node = std::make_shared<JSONGraphNode>(name_prefix + "." + name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);

    // Override attributes
    if (nodes.pad) {
      const auto* pad_attr = nodes.pad->attrs.as<PadAttrs>();
      ICHECK(pad_attr);
      auto p = pad_attr->pad_width;
      // Convert to TVM layout for now, conversion to ACL layout takes place in runtime.
      // Standard convolution pad layout for TVM: top, left, bottom, right.
      std::vector<std::string> padding = {std::to_string(p[1][0].as<IntImmNode>()->value),
                                          std::to_string(p[2][0].as<IntImmNode>()->value),
                                          std::to_string(p[1][1].as<IntImmNode>()->value),
                                          std::to_string(p[2][1].as<IntImmNode>()->value)};
      std::vector<dmlc::any> padding_attr;
      padding_attr.emplace_back(padding);
      json_node->SetAttr("padding", padding_attr);
    }
    if (nodes.activation) {
      std::vector<std::string> activation_type = {"relu"};
      std::vector<dmlc::any> act_attr;
      act_attr.emplace_back(activation_type);
      json_node->SetAttr("activation_type", act_attr);
    }
    return json_node;
  }

  /*!
   * \brief Extract dense nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeDenseNode UnpackCompositeDense(const CallNode* cn) {
    CompositeDenseNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);

    // Traverse composite dense function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "qnn.requantize")) {
      nodes.requantize = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    // Enforce a dense node exists at this point during traversal
    if (nodes.requantize) {
      ICHECK(backend::IsOp(current_call, "qnn.dense"));
    } else {
      ICHECK(backend::IsOp(current_call, "nn.dense"));
    }
    nodes.dense = current_call;
    return nodes;
  }

  /*!
   * \brief Create a JSON representation of a composite dense (fully-connected) operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeDenseJSONNode(const CallNode* cn) {
    CompositeDenseNode nodes = UnpackCompositeDense(cn);
    std::string name = "nn.dense";

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.dense->args[1])[0]);
    if (nodes.requantize) {
      name = "qnn.dense";
      inputs.push_back(VisitExpr(nodes.dense->args[2])[0]);  // input zero-point
      inputs.push_back(VisitExpr(nodes.dense->args[3])[0]);  // weight zero-point
      inputs.push_back(VisitExpr(nodes.dense->args[4])[0]);  // input scale
      inputs.push_back(VisitExpr(nodes.dense->args[5])[0]);  // weight scale
    }
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    if (nodes.requantize) {
      inputs.push_back(VisitExpr(nodes.requantize->args[3])[0]);  // output scale
      inputs.push_back(VisitExpr(nodes.requantize->args[4])[0]);  // output zero-point
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.dense);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite (global) average pooling operator.
   *
   * A composite function is only created when using the int8/uint8 datatype for these operators.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeAvgPool2DJSONNode(const CallNode* cn) {
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* cast = fn->body.as<CallNode>();
    ICHECK(cast);
    const auto* avg_pool = cast->args[0].as<CallNode>();
    ICHECK(avg_pool);
    const auto* avg_pool_op = avg_pool->op.as<OpNode>();
    ICHECK(avg_pool_op);
    const std::string name = avg_pool_op->name;

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, avg_pool);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite L2 pooling operator.
   *
   * \note Relay does not have an operator for L2 pooling, instead we can create
   * an equivalent from power(2) + nn.avg_pool2d + sqrt.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeL2Pool2DJSONNode(const CallNode* cn) {
    const std::string name = "nn.l2_pool2d";
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* sqrt = fn->body.as<CallNode>();
    ICHECK(sqrt);
    const auto* avg_pool = sqrt->args[0].as<CallNode>();
    ICHECK(avg_pool);
    const auto* pow = avg_pool->args[0].as<CallNode>();
    ICHECK(pow);
    const auto* exponent = pow->args[1].as<ConstantNode>();
    ICHECK(exponent);
    ICHECK_EQ(*static_cast<float*>(exponent->data->data), 2) << "Exponent must be 2 for L2 pooling";

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, avg_pool);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a single operator.
   * \param cn The call to be represented.
   * \param name The name of the operator.
   * \return A list of graph entry nodes.
   */
  std::vector<JSONGraphNodeEntry> AddCommonSingleJSONNode(const CallNode* cn, std::string name) {
    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);

    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* callNode = fn->body.as<CallNode>();
    ICHECK(callNode);
    SetCallNodeAttribute(node, callNode);
    return AddNode(node, GetRef<Expr>(cn));
  }
};

/*!
 * \brief Create a runtime module for ACL.
 *
 * This consists of a series of "serialized functions" which each represent a
 * sub-graph to be computed by ACL and will each be executed independently from
 * one another. Each function consists of serialized JSON describing the sub-graph
 * and serialized constant tensors.
 *
 * \note The ACL runtime module only supports a single operator per
 * sub-graph currently.
 *
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module ACLCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  ACLJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();

  // Note that serializer.const_name_to_constant() is ignored. Instead the TECompiler invokes
  // a callback which calls backend::UpdateConstants to capture the map before the function
  // 'disappears' into lowered form, on the assumption the visit order and thus constant
  // names match those generated by the JSONSerializer.

  const auto* pf = runtime::Registry::Get("runtime.arm_compute_lib_runtime_create");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  runtime::Module lib = (*pf)(func_name, graph_json, serializer.const_names());
  return lib;
}

TVM_REGISTER_GLOBAL("relay.ext.arm_compute_lib").set_body_typed(ACLCompiler);

/*!
 * \brief Check whether ACL graph executor is used.
 *
 * \return True if ACL graph executor is enabled, False if not.
 */
inline constexpr bool IsACLRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_ARM_COMPUTE_LIB
  return true;
#else
  return false;
#endif
}

TVM_REGISTER_GLOBAL("relay.op.is_arm_compute_runtime_enabled").set_body_typed(IsACLRuntimeEnabled);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
