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
 * \file src/relay/backend/contrib/acl/codegen_acl.cc
 * \brief Implementation of the Relay -> ACL JSON schema compiler.
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>

#include "../../utils.h"
#include "codegen_acl.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace acl {

void CodegenACL::VisitLeaf(const Expr& expr) {
  if (expr->IsInstance<ConstantNode>()) {
    const auto* constant_node = expr.as<ConstantNode>();
    this->constants_.push_back(constant_node->data);
  } else if (!expr->IsInstance<FunctionNode>()) {
    // Don't enter functions
    MixedModeVisitor::VisitLeaf(expr);
  }
}

void CodegenACL::VisitExpr_(const CallNode* node) {
  Call call = GetRef<Call>(node);
  if (this->layer_table_.find(call) == this->layer_table_.end()) {
    for (const auto& arg : call->args) {
      this->VisitExpr(arg);
    }
    // Determine call -> ACL mapping
    JSONOp layer;
    if (IsAclFunc(node, "acl.conv2d") || backend::IsOp(node, "nn.conv2d")) {
      layer = MakeConvolutionOp(call);
    } else if (backend::IsOp(node, "nn.max_pool2d")) {
      layer = MakeMaxPool2DOp(call);
    } else if (backend::IsOp(node, "reshape")) {
      layer = MakeReshapeOp(call);
    } else {
      LOG(FATAL) << "Unsupported op: " << AsText(node->op, false);
    }
    this->layer_table_[call] = layer;
  }
}

runtime::Module CodegenACL::CreateRuntimeModule(const ObjectRef& ref) {
  std::vector<std::pair<std::string, std::string>> serialized_functions;
  if (ref->IsInstance<FunctionNode>()) {
    IRModule mod;
    Function func = Downcast<Function>(ref);
    auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    CHECK(name_node.defined()) << "Failed to retrieve external symbol";
    mod->Add(GlobalVar(name_node.value()), func);
    mod = this->PreProcessModule(mod);
    for (const auto& it : mod->functions) {
      this->SerializeFunction(it.second, &serialized_functions);
    }
  } else {
    LOG(FATAL) << "The input ref is expected to be a Relay function.";
  }
  std::string data;
  dmlc::MemoryStringStream fs(&data);
  dmlc::SeekStream* strm = &fs;
  strm->Write(serialized_functions.size());
  for (const auto& it : serialized_functions) {
    strm->Write(it.first);
    strm->Write(it.second);
  }
  strm->Seek(0);
  std::string make_acl_module = "runtime.module.loadbinary_acl";
  auto pf = tvm::runtime::Registry::Get(make_acl_module);
  if (pf) {
    return (*pf)(strm);
  } else {
    return runtime::Module();
  }
}

JSONSubGraph CodegenACL::CreateJSONSubgraph(const Function& func) {
  Expr body = func->body;
  this->layer_table_.clear();
  this->constants_.clear();
  this->VisitExpr(body);
  std::vector<JSONOp> ops;
  for (const auto& it : this->layer_table_) {
    ops.push_back(it.second);
  }
  CHECK_EQ(layer_table_.size(), 1) << "ACL codegen expects only a single op per function.";
  return JSONSubGraph(ops[0]);
}

void CodegenACL::SerializeFunction(
    const ObjectRef& ref, std::vector<std::pair<std::string, std::string>>* serialized_functions) {
  Function func = Downcast<Function>(ref);
  JSONSubGraph subgraph = this->CreateJSONSubgraph(func);
  const auto name_node = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  CHECK(name_node != "") << "Fail to retrieve external symbol";
  std::string serialized_pair = SerializeSubgraph(subgraph, this->constants_);
  serialized_functions->emplace_back(name_node.value(), serialized_pair);
}

IRModule CodegenACL::PreProcessModule(const IRModule& mod) {
  IRModule preprocessed_module;
  tvm::Map<String, Array<String>> desired_layouts = {
      {"nn.conv2d", {String("NHWC"), String("OHWI")}}};
  preprocessed_module = transform::ConvertLayout(desired_layouts)(mod);
  preprocessed_module = transform::FoldConstant()(preprocessed_module);
  return preprocessed_module;
}

JSONOp CodegenACL::MakeConvolutionOp(const Call& call) {
  JSONOp op("conv2d");
  const CallNode* pad = nullptr;
  const CallNode* conv;
  const CallNode* bias = nullptr;
  bool has_activation = false;
  if (call->op->IsInstance<FunctionNode>()) {
    Expr composite_conv = GetCompositeExpr(call);
    // Unpack composite function
    const auto* current_call = composite_conv.as<CallNode>();
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
  } else {
    conv = call.as<CallNode>();
  }
  const auto* conv_attr = conv->attrs.as<Conv2DAttrs>();
  CHECK(conv_attr);
  CHECK(conv_attr->kernel_layout == "OHWI")
      << "Kernel layout must be OHWI, has the module been pre-processed correctly?";
  if (pad) {
    op.inputs.push_back(MakeJSONTensor(pad->args[0]));
  } else {
    op.inputs.push_back(MakeJSONTensor(conv->args[0]));
  }
  op.inputs.push_back(MakeJSONConstTensor(conv->args[1]));
  op.outputs.push_back(MakeJSONTensor(GetRef<Expr>(conv)));
  if (bias) {
    op.inputs.push_back(MakeJSONConstTensor(bias->args[1]));
  }
  // It seems there are two different methods for padding a convolution:
  // - using nn.pad operator before convolution
  // - using conv2d_attrs to add padding
  //
  // Cover both cases here.
  std::vector<int> padding;
  if (pad) {
    const auto* pad_attr = pad->attrs.as<PadAttrs>();
    CHECK(pad_attr);
    padding = GetPadVector(pad_attr->pad_width);
  } else {
    padding = GetPadVector(conv_attr->padding);
  }
  op.attrs["padding"] = padding;
  op.attrs["groups"] = conv_attr->groups;
  op.attrs["strides"] = ToVector(conv_attr->strides);
  if (has_activation) op.attrs["activation_type"] = std::string("relu");
  return op;
}

JSONOp CodegenACL::MakeMaxPool2DOp(const Call& call) {
  JSONOp op("max_pool");
  const auto* attr = call->attrs.as<MaxPool2DAttrs>();
  CHECK(attr);
  op.inputs.push_back(MakeJSONTensor(call->args[0]));
  op.outputs.push_back(MakeJSONTensor(call));
  op.attrs["padding"] = GetPadVector(attr->padding);
  op.attrs["strides"] = ToVector(attr->strides);
  op.attrs["pooling_type"] = std::string("max");
  op.attrs["pool_size"] = ToVector(attr->pool_size);
  return op;
}

JSONOp CodegenACL::MakeReshapeOp(const Call& call) {
  JSONOp op("reshape");
  const auto* attr = call->attrs.as<ReshapeAttrs>();
  CHECK(attr);
  op.inputs.push_back(MakeJSONTensor(call->args[0]));
  op.outputs.push_back(MakeJSONTensor(call));
  return op;
}

JSONTensor CodegenACL::MakeJSONTensor(const Expr& expr) {
  const auto* ttnode = expr->checked_type().as<TensorTypeNode>();
  CHECK(ttnode);
  std::vector<int> shape = ToVector(ttnode->shape);
  return JSONTensor("var", shape);
}

JSONTensor CodegenACL::MakeJSONConstTensor(const Expr& expr) {
  const auto* ttnode = expr->checked_type().as<TensorTypeNode>();
  CHECK(ttnode);
  std::vector<int> shape = ToVector(ttnode->shape);
  VisitExpr(expr);
  return JSONTensor("const", shape);
}

bool CodegenACL::IsAclFunc(const CallNode* call, const std::string& op_name) const {
  if (call->op->IsInstance<FunctionNode>()) {
    Function func = Downcast<Function>(call->op);
    CHECK(func.defined());
    auto name_node = func->GetAttr<String>(attr::kComposite);
    return name_node.value() == op_name;
  }
  return false;
}

Expr CodegenACL::GetCompositeExpr(const Call& call) {
  Function composite_function = Downcast<Function>(call->op);
  Expr composite_expr = composite_function->body;
  CHECK(composite_expr->IsInstance<CallNode>());
  return composite_expr;
}

std::vector<int> CodegenACL::ToVector(const Array<IndexExpr>& array) {
  std::vector<int> stl_vector;
  for (auto it : array) {
    const auto* val = it.as<IntImmNode>();
    CHECK(val);
    stl_vector.push_back(val->value);
  }
  return stl_vector;
}

std::vector<int> CodegenACL::GetPadVector(const Array<Array<IndexExpr>>& pad) {
  // TVM nn.pad: top, bottom, left, right -> ACL Pad: left, right, top, bottom
  auto acl_pad = {pad[2][0], pad[2][1], pad[1][0], pad[1][1]};
  return ToVector(acl_pad);
}

std::vector<int> CodegenACL::GetPadVector(const Array<IndexExpr>& pad) {
  Array<IndexExpr> acl_pad;
  switch (pad.size()) {
    case 1:
      acl_pad = {pad[0], pad[0], pad[0], pad[0]};
      break;
    case 2:
      // TVM Pad: height, width -> ACL Pad: left, right, top, bottom
      acl_pad = {pad[1], pad[1], pad[0], pad[0]};
      break;
    case 4:
      // TVM Pad: top, left, bottom, right -> ACL Pad: left, right, top, bottom
      acl_pad = {pad[1], pad[3], pad[0], pad[2]};
      break;
    default:
      LOG(FATAL) << "Unsupported padding dimensions";
  }
  return ToVector(acl_pad);
}

}  // namespace acl
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
