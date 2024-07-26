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
 * \file src/relay/backend/contrib/clml/codegen.cc
 * \brief Implementation of the Relay -> CLML JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>
#include <tvm/tir/analysis.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {

constexpr const char* kCLMLTargetVersion = "relay.ext.clml.target_version";
TVM_REGISTER_PASS_CONFIG_OPTION(kCLMLTargetVersion, Integer);

namespace relay {
namespace contrib {

/*!
 * \brief Generates an CLMLModule from a relay expression. This "compilation"
 * does not require CLML since the actual conversion using CLML APIs is
 * deferred until creation of the runtime. This step simply serializes the
 * relay program into a JSON string.
 */
class CLMLJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  CLMLJSONSerializer(const std::string& symbol, const Expr& expr)
      : JSONSerializer(symbol, expr), clml_symbol_(symbol) {}

  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports nn.conv2d
   */
  struct CompositeConvNode {
    const CallNode* pad = nullptr;
    const CallNode* conv = nullptr;
    const CallNode* bn = nullptr;
    const CallNode* bias = nullptr;
    const CallNode* activation = nullptr;
    std::string act_type;
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
      LOG(FATAL) << "CLML JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "CLML JSON runtime only supports composite functions.";
    const std::string name = comp.value();
    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "clml.conv2d" || name == "clml.pad_conv2d" || name == "clml.conv2d_transpose") {
      json_node = CreateCompositeConvJSONNode(cn);
    } else if (name == "clml.batch_norm") {
      json_node = CreateBatchNormJSONNode(cn);
    } else if (name == "clml.dense1d" || name == "clml.dense2d") {
      json_node = CreateDenseJSONNode(cn);
    } else if (name == "clml.pad") {
      json_node = CreatePadJSONNode(cn);
    } else if (name == "clml.concat") {
      json_node = CreateConcatJSONNode(cn);
    } else {
      json_node = CreateGenericJSONNode(cn);
    }
    return AddNode(json_node, GetRef<Expr>(cn));
  }

  /*!
   * \brief Visit call nodes and generate ordered params.
   *
   * \param cn The current constant node.
   * \return A list of graph entry nodes.
   */
  std::vector<JSONGraphNodeEntry> VisitExpr_(const ConstantNode* cn) override {
    std::string name = "clml_" + clml_symbol_ + "_const_" + std::to_string(clml_params_.size());
    clml_params_.push_back(name);
    clml_params_map_[name] = cn->data;
    auto node = std::make_shared<JSONGraphNode>(name, "const" /* op_type_ */);
    return AddNode(node, GetRef<Expr>(cn));
  }

  Array<String> GetParams() const { return clml_params_; }
  Map<String, runtime::NDArray> GetParamsMap() const {
    return Map<String, runtime::NDArray>(clml_params_map_);
  }

 private:
  std::string clml_symbol_;
  Array<String> clml_params_;
  std::unordered_map<String, runtime::NDArray> clml_params_map_;
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
    if (fn->body.as<TupleGetItemNode>()) {
      auto tuple_item = fn->body.as<TupleGetItemNode>();
      current_call = tuple_item->tuple.as<CallNode>();
    } else {
      current_call = fn->body.as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      nodes.act_type = "relu";
      if (current_call->args[0].as<TupleGetItemNode>()) {
        auto tuple_item = current_call->args[0].as<TupleGetItemNode>();
        current_call = tuple_item->tuple.as<CallNode>();
      } else {
        current_call = current_call->args[0].as<CallNode>();
      }
    } else if (backend::IsOp(current_call, "clip")) {
      nodes.activation = current_call;
      nodes.act_type = "relu6";
      if (current_call->args[0].as<TupleGetItemNode>()) {
        auto tuple_item = current_call->args[0].as<TupleGetItemNode>();
        current_call = tuple_item->tuple.as<CallNode>();
      } else {
        current_call = current_call->args[0].as<CallNode>();
      }
    }
    if (backend::IsOp(current_call, "nn.batch_norm")) {
      nodes.bn = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "add") || backend::IsOp(current_call, "nn.bias_add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    // Enforce a convolution node exists at this point during traversal
    if (!backend::IsOp(current_call, "nn.conv2d") &&
        !backend::IsOp(current_call, "nn.conv2d_transpose")) {
      LOG(FATAL) << "Can't find primary op in Convolution node";
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

    std::string name;
    std::string name_prefix = "nn";
    if (backend::IsOp(nodes.conv, "nn.conv2d")) {
      const auto* conv_attr = nodes.conv->attrs.as<Conv2DAttrs>();
      ICHECK(conv_attr);
      if (conv_attr->channels.defined() &&
          tvm::tir::ExprDeepEqual()(conv_attr->channels, conv_attr->groups) &&
          conv_attr->groups != 1) {
        name = "depthwise_conv2d";
        ICHECK(conv_attr->kernel_layout == "IOHW")
            << "Kernel layout must be IHWO, has the module been pre-processed correctly?";
      } else {
        name = "conv2d";
        ICHECK(conv_attr->kernel_layout == "OIHW")
            << "Kernel layout must be OHWI, has the module been pre-processed correctly?";
      }
    } else if (backend::IsOp(nodes.conv, "nn.conv2d_transpose")) {
      name = "conv2d_transpose";
      const auto* conv_transpose_attr = nodes.conv->attrs.as<Conv2DTransposeAttrs>();
      ICHECK(conv_transpose_attr);
      ICHECK(conv_transpose_attr->kernel_layout == "OIHW")
          << "Kernel layout must be OHWI, has the module been pre-processed correctly?";
    }

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;

    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    // Deal with Batchnorm Fusing here
    if (nodes.bn) {
      inputs.push_back(VisitExpr(nodes.bn->args[1])[0]);
      inputs.push_back(VisitExpr(nodes.bn->args[2])[0]);
      inputs.push_back(VisitExpr(nodes.bn->args[3])[0]);
      inputs.push_back(VisitExpr(nodes.bn->args[4])[0]);
    }

    auto json_node = std::make_shared<JSONGraphNode>(name_prefix + "." + name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);

    if (nodes.bn) {
      const auto* bn_attr = nodes.bn->attrs.as<BatchNormAttrs>();
      std::vector<dmlc::any> bn_any_attr;
      std::vector<std::string> bn_args = {
          std::to_string(bn_attr->axis), std::to_string(bn_attr->epsilon),
          std::to_string(bn_attr->center), std::to_string(bn_attr->scale)};
      bn_any_attr.emplace_back(bn_args);
      json_node->SetAttr("batchnorm", bn_any_attr);
    }

    // Override attributes
    if (nodes.pad) {
      const auto* pad_attr = nodes.pad->attrs.as<PadAttrs>();
      ICHECK(pad_attr);
      auto p = pad_attr->pad_width;
      // Standard convolution pad layout for TVM: dimension wise pair of pre and post padding.
      // CLML takes dimension wise pre-padding followed by dimension wise post-padding.
      std::vector<std::string> padding = {std::to_string(p[2][0].as<IntImmNode>()->value),
                                          std::to_string(p[3][0].as<IntImmNode>()->value),
                                          std::to_string(p[2][1].as<IntImmNode>()->value),
                                          std::to_string(p[3][1].as<IntImmNode>()->value)};
      std::vector<dmlc::any> padding_attr;
      padding_attr.emplace_back(padding);
      json_node->SetAttr("padding", padding_attr);
    }

    if (nodes.activation) {
      std::vector<std::string> activation_type = {nodes.act_type};
      std::vector<dmlc::any> act_attr;
      act_attr.emplace_back(activation_type);
      json_node->SetAttr("activation_type", act_attr);
    }
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a Batchnorm operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateBatchNormJSONNode(const CallNode* cn) {
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* tuple_item = fn->body.as<TupleGetItemNode>();
    ICHECK(tuple_item);
    const auto* bn = tuple_item->tuple.as<CallNode>();
    ICHECK(bn);
    const auto* bn_op = bn->op.as<OpNode>();
    ICHECK(bn_op);
    const std::string name = bn_op->name;

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(bn->args[1])[0]);
    inputs.push_back(VisitExpr(bn->args[2])[0]);
    inputs.push_back(VisitExpr(bn->args[3])[0]);
    inputs.push_back(VisitExpr(bn->args[4])[0]);
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, bn);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a Concat operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateConcatJSONNode(const CallNode* cn) {
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* concat = fn->body.as<CallNode>();

    ICHECK(backend::IsOp(concat, "concatenate"));
    const auto* concat_op = concat->op.as<OpNode>();
    ICHECK(concat_op);
    const std::string name = concat_op->name;

    std::vector<JSONGraphNodeEntry> inputs;
    for (auto arg : cn->args) {
      inputs.push_back(VisitExpr(arg)[0]);
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, concat);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a Dense operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateDenseJSONNode(const CallNode* cn) {
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* dense = fn->body.as<CallNode>();
    const CallNode* bias = nullptr;

    if (backend::IsOp(dense, "add") || backend::IsOp(dense, "nn.bias_add")) {
      bias = dense;
      dense = dense->args[0].as<CallNode>();
    }
    ICHECK(backend::IsOp(dense, "nn.dense"));
    const auto* dense_op = dense->op.as<OpNode>();
    ICHECK(dense_op);
    const std::string name = dense_op->name;

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(dense->args[1])[0]);
    if (bias) {
      inputs.push_back(VisitExpr(bias->args[1])[0]);
    }
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, dense);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a Pad operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreatePadJSONNode(const CallNode* cn) {
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* pad = fn->body.as<CallNode>();
    const auto* pad_op = pad->op.as<OpNode>();
    ICHECK(pad_op);
    const std::string name = pad_op->name;

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);

    const auto* pad_attr = pad->attrs.as<PadAttrs>();
    ICHECK(pad_attr);
    auto p = pad_attr->pad_width;
    // TVM padding format: Dimension wise pair of pre and post padding.
    // CLML padding format: Dimension wise pre padding followed by dimension wise post padding.
    std::vector<std::string> padding = {std::to_string(p[2][0].as<IntImmNode>()->value),
                                        std::to_string(p[2][1].as<IntImmNode>()->value),
                                        std::to_string(p[3][0].as<IntImmNode>()->value),
                                        std::to_string(p[3][1].as<IntImmNode>()->value)};
    std::vector<dmlc::any> padding_attr;
    padding_attr.emplace_back(padding);
    json_node->SetAttr("pad_width", padding_attr);

    std::vector<std::string> pad_mode = {pad_attr->pad_mode};
    std::vector<dmlc::any> pad_mode_attr;
    pad_mode_attr.emplace_back(pad_mode);
    json_node->SetAttr("pad_mode", pad_mode_attr);

    return json_node;
  }

  std::shared_ptr<JSONGraphNode> CreateGenericJSONNode(const CallNode* cn) {
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);
    const auto* node = fn->body.as<CallNode>();

    const auto* node_op = node->op.as<OpNode>();
    ICHECK(node_op);
    const std::string name = node_op->name;

    std::vector<JSONGraphNodeEntry> inputs;
    unsigned int i = 0;
    for (i = 0; i < cn->args.size(); i++) {
      inputs.push_back(VisitExpr(cn->args[i])[0]);
    }
    for (unsigned int j = i; j < node->args.size(); j++) {
      inputs.push_back(VisitExpr(node->args[j])[0]);
    }
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, node);
    return json_node;
  }
};

/*!
 * \brief Create a runtime module for CLML.
 *
 * This consists of a series of "serialized functions" which each represent a
 * sub-graph to be computed by CLML and will each be executed independently from
 * one another. Each function consists of serialized JSON describing the sub-graph
 * and serialized constant tensors.
 *
 * \note The CLML runtime module only supports a single operator per
 * sub-graph currently.
 *
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module CLMLCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  CLMLJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto param_names = serializer.GetParams();
  const auto* pf = runtime::Registry::Get("runtime.clml_runtime_create");
  ICHECK(pf != nullptr) << "Cannot find CLML runtime module to create";
  runtime::Module lib = (*pf)(func_name, graph_json, param_names);
  return lib;
}

TVM_REGISTER_GLOBAL("relay.ext.clml").set_body_typed(CLMLCompiler);

/*!
 * \brief Check whether CLML graph runtime is used.
 *
 * \return True if CLML graph runtime is enabled, False if not.
 */
inline constexpr bool IsCLMLRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_CLML
  return true;
#else
  return false;
#endif
}

TVM_REGISTER_GLOBAL("relay.op.is_clml_runtime_enabled").set_body_typed(IsCLMLRuntimeEnabled);

Map<String, runtime::NDArray> CLMLConstantUpdater(Expr func, std::string symbol) {
  CLMLJSONSerializer serializer(symbol, func);
  serializer.serialize();
  auto pmap = serializer.GetParamsMap();
  return pmap;
}

TVM_REGISTER_GLOBAL("relay.ext.clml.constant_updater").set_body_typed(CLMLConstantUpdater);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
