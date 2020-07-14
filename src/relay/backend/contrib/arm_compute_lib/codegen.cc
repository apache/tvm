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
 * \file src/relay/backend/contrib/arm_compute_lib/codegen_acl.cc
 * \brief Implementation of the Relay -> ACL JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>

#include "../../utils.h"
#include "codegen_acl.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace arm_compute_lib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

std::vector<JSONGraphNodeEntry> ACLJSONSerializer::VisitExpr_(const CallNode* cn) {
  Expr expr = GetRef<Expr>(cn);
  std::string name;
  std::shared_ptr<JSONGraphNode> json_node;

  if (cn->op.as<OpNode>()) {
    json_node = CreateOpJSONNode(cn);
  } else if (const auto* fn = cn->op.as<FunctionNode>()) {
    auto comp = fn->GetAttr<String>(attr::kComposite);
    CHECK(comp.defined()) << "Arm Compute Library JSON runtime only supports composite functions.";
    name = comp.value();
    if (name == "arm_compute_lib.conv2d") {
      json_node = CreateCompositeConvJSONNode(cn);
    } else {
      LOG(FATAL) << "Unrecognized Arm Compute Library pattern: " << name;
    }
  } else {
    LOG(FATAL) << "Arm Compute Library JSON runtime does not support calls to "
               << cn->op->GetTypeKey();
  }

  return AddNode(json_node, GetRef<Expr>(cn));
}

std::vector<JSONGraphNodeEntry> ACLJSONSerializer::VisitExpr_(const ConstantNode* cn) {
  this->constants_.push_back(cn->data);
  return JSONSerializer::VisitExpr_(cn);
}

std::shared_ptr<JSONGraphNode> ACLJSONSerializer::CreateOpJSONNode(const CallNode* cn) {
  const auto* op = cn->op.as<OpNode>();
  CHECK(op);
  const std::string name = op->name;
  // Collect inputs
  std::vector<JSONGraphNodeEntry> inputs;
  for (const auto& arg : cn->args) {
    auto res = VisitExpr(arg);
    inputs.insert(inputs.end(), res.begin(), res.end());
  }
  // Create JSON op
  auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
  SetCallNodeAttribute(json_node, cn);
  return json_node;
}

std::shared_ptr<JSONGraphNode> ACLJSONSerializer::CreateCompositeConvJSONNode(const CallNode* cn) {
  const std::string name = "arm_compute_lib.conv2d";
  const CallNode* pad = nullptr;
  const CallNode* conv;
  const CallNode* bias = nullptr;
  bool has_activation = false;

  // Unpack composite function
  const auto* fn = cn->op.as<FunctionNode>();
  CHECK(fn);
  const auto* current_call = fn->body.as<CallNode>();
  if (backend::IsOp(current_call, "nn.relu")) {
    has_activation = true;
    current_call = current_call->args[0].as<CallNode>();
  }
  if (backend::IsOp(current_call, "nn.bias_add")) {
    bias = current_call;
    current_call = current_call->args[0].as<CallNode>();
  }
  CHECK(backend::IsOp(current_call, "nn.conv2d"));
  conv = current_call;
  if (!current_call->args.empty() && current_call->args[0]->IsInstance<CallNode>()) {
    current_call = current_call->args[0].as<CallNode>();
    if (backend::IsOp(current_call, "nn.pad")) {
      pad = current_call;
    }
  }

  const auto* conv_attr = conv->attrs.as<Conv2DAttrs>();
  CHECK(conv_attr);
  CHECK(conv_attr->kernel_layout == "OHWI")
      << "Kernel layout must be OHWI, has the module been pre-processed correctly?";

  std::vector<JSONGraphNodeEntry> inputs;
  inputs.push_back(VisitExpr(cn->args[0])[0]);
  inputs.push_back(VisitExpr(conv->args[1])[0]);
  if (bias) {
    inputs.push_back(VisitExpr(bias->args[1])[0]);
  }

  auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
  SetCallNodeAttribute(json_node, conv);

  // Override attributes
  if (pad) {
    const auto* pad_attr = pad->attrs.as<PadAttrs>();
    CHECK(pad_attr);
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
  if (has_activation) {
    std::vector<std::string> activation_type = {"relu"};
    std::vector<dmlc::any> act_attr;
    act_attr.emplace_back(activation_type);
    json_node->SetAttr("activation_type", act_attr);
  }
  return json_node;
}

Array<runtime::NDArray> ACLJSONSerializer::GetParamsData() { return constants_; }

IRModule PreProcessModule(const IRModule& mod) {
  IRModule preprocessed_module;
  tvm::Map<String, Array<String>> desired_layouts = {
      {"nn.conv2d", {String("NHWC"), String("OHWI")}}};
  preprocessed_module = transform::ConvertLayout(desired_layouts)(mod);
  preprocessed_module = transform::FoldConstant()(preprocessed_module);
  return preprocessed_module;
}

runtime::Module ACLCompiler(const ObjectRef& ref) {
  CHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  IRModule mod;
  mod->Add(GlobalVar(func_name), func);
  mod = PreProcessModule(mod);

  CHECK(mod->functions.size() == 1) << "Module should only contain single function";
  Function processed_func = Downcast<Function>(mod->functions.begin().operator*().second);

  ACLJSONSerializer serializer(func_name, processed_func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto param_names = serializer.GetParams();
  auto param_data = serializer.GetParamsData();
  const auto* pf = runtime::Registry::Get("runtime.arm_compute_lib_runtime_create");
  CHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  runtime::Module lib = (*pf)(func_name, graph_json, param_names, param_data);
  return lib;
}

TVM_REGISTER_GLOBAL("relay.ext.arm_compute_lib").set_body_typed(ACLCompiler);

inline constexpr bool IsACLRuntimeEnabled() {
#if TVM_GRAPH_RUNTIME_ARM_COMPUTE_LIB
  return true;
#else
  return false;
#endif
}

TVM_REGISTER_GLOBAL("relay.op.is_arm_compute_runtime_enabled").set_body_typed(IsACLRuntimeEnabled);

}  // namespace arm_compute_lib
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
