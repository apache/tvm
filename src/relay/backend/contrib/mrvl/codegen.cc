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
 * \file src/relay/backend/contrib/mrvl/codegen.cc
 * \brief Marvell MLIP specific API
 */

#include <stdio.h>
#include <tvm/ir/module.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>
#include <tvm/tir/analysis.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "picojson.h"

#define USE_JSON_RUNTIME 1
#ifdef USE_JSON_RUNTIME

#include "../../../../runtime/contrib/json/json_node.h"
#include "../../../qnn/utils.h"
#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

#else

// TODO(ccjoechou): TBA if needed -- follow "per layer" C-codegen example

#endif

namespace tvm {
namespace relay {
namespace contrib {
namespace mrvl {

using namespace backend;

extern "C" bool g_mrvlExtJsonObjInstantized;

extern "C" void InstantiateMrvlExtJsonObj();

#ifndef USE_JSON_RUNTIME

// TODO(ccjoechou): TBA if needed -- follow "per layer" C-codegen example

#else

/*!
 * \brief Generates an MrvlModule from a relay expression. This "compilation"
 * does not require Mrvl driver since the actual conversion using Mrvl APIs is
 * deferred until creation of the runtime. This step simply serializes the
 * relay program into a JSON string.
 */
class MrvlJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  MrvlJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {
    func_name_ = symbol;
  }

  /*!
   * \brief Struct to capture original frontend NN model's
   *        first/last operator names for fused Mrvl layer
   */
  struct FrontendOpNames {
    std::string first_op_name = "unknown";
    std::string last_op_name = "unknown";
  };

  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports both nn.conv2d and qnn.conv2d.
   */
  struct CompositeConvNode {
    const CallNode* pad = nullptr;
    const CallNode* conv = nullptr;
    const CallNode* add = nullptr;
    const CallNode* batch_norm = nullptr;
    const CallNode* activation = nullptr;
    FrontendOpNames op_names;
  };

  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports sum2d
   */
  struct CompositeSum2DNode {
    const CallNode* add = nullptr;
    const CallNode* activation = nullptr;
    FrontendOpNames op_names;
  };

  /*!
   * \brief A series of operators that form a composite
   * maxpool or avgpool. Supports both nn.max_pool2d and qnn.conv2d.
   */
  struct CompositePoolNode {
    const CallNode* pad = nullptr;
    const CallNode* pool = nullptr;
    FrontendOpNames op_names;
  };

  /*!
   * \brief A series of operators that form a composite
   * fc layer. Supports both nn.fc_ni2no and qnn.fc_ni2no.
   */
  struct CompositeFcNode {
    const CallNode* fc = nullptr;
    const CallNode* add = nullptr;
    const CallNode* activation = nullptr;
    FrontendOpNames op_names;
  };

  /*!
   * \brief Visit call nodes and generate appropriate JSON node.
   *
   * \param cn The current call node.
   * \return A list of graph entry nodes.
   */
  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    const auto* op_node = cn->op.as<OpNode>();
    if (op_node) {
      // handle certain op node types specially
      String op_name = tvm::Op::GetOpName(GetRef<Op>(op_node));
      bool handle_by_mrvl = (op_name == "reshape") || (op_name == "layout_transform") ||
                            (op_name == "nn.batch_flatten") || (op_name == "transpose");
      if (!handle_by_mrvl) {
        return JSONSerializer::VisitExpr_(cn);
      }

      // setup json attributes and then add the Mrvl Layer to JSON files
      std::shared_ptr<JSONGraphNode> json_node;
      json_node = CreateMrvlLayer4OpNode(cn);
      return AddNode(json_node, GetRef<Expr>(cn));
    }

    // handle only mrvl composite functions
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "Mrvl JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "Mrvl JSON runtime only supports composite functions.";
    const std::string name = comp.value();
    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "mrvl.conv2d_nhwc2nhwc") {
      json_node = CreateCompositeMrvlConv2DLayer(cn);
    } else if (name == "mrvl.fc_ni2no") {
      json_node = CreateCompositeMrvlFcLayer(cn);
    } else if (name == "mrvl.maxpool2d_nhwc2nhwc") {
      json_node = CreateCompositeMrvlMaxpool2DLayer(cn);
    } else if (name == "mrvl.avgpool2d_nhwc2nhwc") {
      json_node = CreateCompositeMrvlAvgpool2DLayer(cn);
    } else if (name == "mrvl.sum2d") {
      json_node = CreateCompositeMrvlSum2DLayer(cn);
    } else {
      LOG(FATAL) << "Unrecognized Mrvl pattern: " << name;
    }
    // calling codegen_json.h::AddNode()
    return AddNode(json_node, GetRef<Expr>(cn));
  }

 private:
  std::string func_name_;

  void JsonNodeSetAttr(std::shared_ptr<JSONGraphNode> json_node, const std::string& key,
                       const std::vector<std::string>& string_vec) {
    std::vector<dmlc::any> json_attr;
    json_attr.emplace_back(string_vec);
    json_node->SetAttr(key, json_attr);
  }

  void JsonNodeSetVecAttr(std::shared_ptr<JSONGraphNode> json_node, const std::string& key,
                          const std::vector<int64_t>& tvec) {
    size_t tvec_size = tvec.size();
    std::vector<std::string> tvec_str;
    if (tvec_size == 4) {
      tvec_str = {std::to_string(tvec[0]), std::to_string(tvec[1]), std::to_string(tvec[2]),
                  std::to_string(tvec[3])};
    } else if (tvec_size == 3) {
      tvec_str = {std::to_string(tvec[0]), std::to_string(tvec[1]), std::to_string(tvec[2])};
    } else if (tvec_size == 2) {
      tvec_str = {std::to_string(tvec[0]), std::to_string(tvec[1])};
    } else if (tvec_size == 1) {
      tvec_str = {std::to_string(tvec[0])};
    } else {
      LOG(INFO) << "Vector size (" << tvec_size << ") is not supported.";
    }
    std::vector<dmlc::any> json_attr;
    json_attr.emplace_back(tvec_str);
    json_node->SetAttr(key, json_attr);
  }

  void setMrvlLayerCommonAttrs(std::shared_ptr<JSONGraphNode> json_node, const CallNode* cn,
                               const std::string& func_name, const std::string& mrvlLayerName,
                               const std::string& data_layout, const std::string& kernel_layout,
                               const std::string& out_layout) {
    // MUST use the JSONGraphAttrs attrs_ style
    //   as described in tvm/src/relay/contrib/json/json_node.h

    // add other mrvl-specific attributes
    JsonNodeSetAttr(json_node, "layer_name", {mrvlLayerName});
    JsonNodeSetAttr(json_node, "func_node_name", {func_name});
    std::vector<int64_t> data_layout_vec;
    GetInputTensorShapeViaArg0(cn, &data_layout_vec);
    JsonNodeSetVecAttr(json_node, "data_layout_shape", data_layout_vec);
    std::vector<int64_t> out_layout_vec;
    GetOutputTensorShape(cn, &out_layout_vec);
    JsonNodeSetVecAttr(json_node, "out_layout_shape", out_layout_vec);
    if (data_layout != "") {
      std::vector<std::string> data_layout_format_vec = {data_layout};
      JsonNodeSetAttr(json_node, "data_layout", data_layout_format_vec);
    }
    if (kernel_layout != "") {
      std::vector<std::string> kernel_layout_format_vec = {kernel_layout};
      JsonNodeSetAttr(json_node, "kernel_layout", kernel_layout_format_vec);
    }
    if (out_layout != "") {
      std::vector<std::string> out_layout_format_vec = {out_layout};
      JsonNodeSetAttr(json_node, "out_layout", out_layout_format_vec);
    }
  }

  void SetMrvlSpecificJsonNodeAttrs(std::shared_ptr<JSONGraphNode> json_node, const CallNode* cn,
                                    const CallNode* cn_pad, const CallNode* cn_pool,
                                    const CallNode* cn_conv, const CallNode* cn_fc,
                                    const CallNode* cn_add, const CallNode* cn_batch_norm,
                                    const CallNode* cn_activation, const std::string& mrvlLayerName,
                                    const std::string& data_layout,
                                    const std::string& kernel_layout, const std::string& out_layout,
                                    const std::string& bias_layout,
                                    const std::string& activation_op,
                                    const FrontendOpNames& op_names) {
    // MUST use the JSONGraphAttrs attrs_ style
    //   as described in tvm/src/relay/contrib/json/json_node.h
    setMrvlLayerCommonAttrs(json_node, cn, func_name_, mrvlLayerName, data_layout, kernel_layout,
                            out_layout);
    //
    if (cn_conv || cn_fc) {
      std::vector<std::string> kernel_const_name = {func_name_ + "_const_0"};
      JsonNodeSetAttr(json_node, "kernel_const_name", kernel_const_name);
    }
    //
    if (cn_add) {
      if (mrvlLayerName == "Sum2D") {
        // FIXME: any specific attributes to add here for Sum2D?
        JsonNodeSetAttr(json_node, "out_layout", {out_layout});
      } else {
        std::vector<std::string> bias_const_name = {func_name_ + "_const_1"};
        JsonNodeSetAttr(json_node, "bias_const_name", bias_const_name);
        JsonNodeSetAttr(json_node, "bias_layout", {bias_layout});
      }
    }
    if (cn_batch_norm) {
      std::string gamma_const_name_postfix;
      std::string beta_const_name_postfix;
      std::string mean_const_name_postfix;
      std::string var_const_name_postfix;
      if (cn_add) {
        gamma_const_name_postfix = "_const_2";
        beta_const_name_postfix = "_const_3";
        mean_const_name_postfix = "_const_4";
        var_const_name_postfix = "_const_5";
      } else {
        gamma_const_name_postfix = "_const_1";
        beta_const_name_postfix = "_const_2";
        mean_const_name_postfix = "_const_3";
        var_const_name_postfix = "_const_4";
      }
      std::string batch_norm_layout = "-O";
      std::vector<std::string> gamma_const_name = {func_name_ + gamma_const_name_postfix};
      JsonNodeSetAttr(json_node, "gamma_const_name", gamma_const_name);
      JsonNodeSetAttr(json_node, "gamma_layout", {batch_norm_layout});
      std::vector<std::string> beta_const_name = {func_name_ + beta_const_name_postfix};
      JsonNodeSetAttr(json_node, "beta_const_name", beta_const_name);
      JsonNodeSetAttr(json_node, "beta_layout", {batch_norm_layout});
      std::vector<std::string> mean_const_name = {func_name_ + mean_const_name_postfix};
      JsonNodeSetAttr(json_node, "mean_const_name", mean_const_name);
      JsonNodeSetAttr(json_node, "mean_layout", {batch_norm_layout});
      std::vector<std::string> var_const_name = {func_name_ + var_const_name_postfix};
      JsonNodeSetAttr(json_node, "var_const_name", var_const_name);
      JsonNodeSetAttr(json_node, "var_layout", {batch_norm_layout});
    }
    //
    if (cn_pool && (mrvlLayerName == "Maxpool2D")) {
      auto pool_attrs = cn_pool->attrs.as<MaxPool2DAttrs>();
      ICHECK(pool_attrs != nullptr);
      std::vector<int64_t> kernel_layout_vec;
      kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[0])));
      kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[1])));
      JsonNodeSetVecAttr(json_node, "kernel_layout_shape", kernel_layout_vec);
    }
    if (cn_pool && (mrvlLayerName == "Avgpool2D")) {
      auto pool_attrs = cn_pool->attrs.as<AvgPool2DAttrs>();
      ICHECK(pool_attrs != nullptr);
      std::vector<int64_t> kernel_layout_vec;
      kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[0])));
      kernel_layout_vec.push_back(*(tir::as_const_int(pool_attrs->pool_size[1])));
      JsonNodeSetVecAttr(json_node, "kernel_layout_shape", kernel_layout_vec);
    }
    //
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK((fn != nullptr) && fn->IsInstance<FunctionNode>());
    auto composite = fn->GetAttr<String>(attr::kComposite);
    ICHECK(composite.defined());
    std::string composite_name = composite.value();
    JsonNodeSetAttr(json_node, "composite_name", {composite_name});
    JsonNodeSetAttr(json_node, "first_op_name", {op_names.first_op_name});
    JsonNodeSetAttr(json_node, "last_op_name", {op_names.last_op_name});

    // Override attributes, if nn.pad() found
    // - for 2D: h * w: h-begin (top), w-begin (left), h-end (bottom), w-end (right)
    //   -- for Conv-2D op and Pool-2D op
    if (cn_pad) {
      const auto* pad_attr = cn_pad->attrs.as<PadAttrs>();
      ICHECK(pad_attr);
      auto p = pad_attr->pad_width;
      // Convert to TVM layout for now, conversion to Mrvl layout takes place in runtime.
      // Standard convolution pad layout for TVM: top, left, bottom, right.
      std::vector<std::string> padding = {std::to_string(p[1][0].as<IntImmNode>()->value),
                                          std::to_string(p[2][0].as<IntImmNode>()->value),
                                          std::to_string(p[1][1].as<IntImmNode>()->value),
                                          std::to_string(p[2][1].as<IntImmNode>()->value)};
      std::vector<dmlc::any> padding_attr;
      padding_attr.emplace_back(padding);
      json_node->SetAttr("padding", padding_attr);
    }
    // Override attributes
    if (cn_activation) {
      std::vector<std::string> activation_type = {activation_op};
      std::vector<dmlc::any> act_attr;
      act_attr.emplace_back(activation_type);
      json_node->SetAttr("activation_type", act_attr);
    }
  }

  /*!
   * \brief Extract convolution nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeConvNode UnpackCompositeConvolution(const CallNode* call) {
    CompositeConvNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn);
    // we can see the pattern below (but call graph starts from right most & backward):
    // - conv2d + [ bias_add ] + [ batch_norm + tuple.getitem(0) ] + [ relu ]
    //
    // Thus, we need to handle following cases:
    // - case1: conv2d
    // - case2: conv2d + relu
    // - case3: conv2d + batch_norm + tuple.getitem(0)
    // - case4: conv2d + batch_norm + tuple.getitem(0) + relu
    // - case5: conv2d + add
    // - case6: conv2d + add + relu
    // - case7: conv2d + add + batch_norm + tuple.getitem(0)
    // - case8: conv2d + add + batch_norm + tuple.getitem(0) + relu

    // Traverse composite convolution function from child to parent
    const TupleGetItemNode* tuple_get_item_node = nullptr;
    const CallNode* current_call = fn->body.as<CallNode>();
    if (current_call) {
      // for case1, case2, case4, case5, case6, case8
      if (backend::IsOp(current_call, "nn.relu")) {
        // for case2, case4, case6, case8
        nodes.activation = current_call;

        if (current_call->args[0].as<TupleGetItemNode>()) {
          // fall through for case4, case8
          tuple_get_item_node = current_call->args[0].as<TupleGetItemNode>();
        } else {
          // fall through for case2, case6: to use current_call as CallNode*
          current_call = current_call->args[0].as<CallNode>();
        }
      } else {
        // fall through for case1, case5: to use current_call as CallNode*
        ICHECK(current_call);
      }
    } else {
      // for case3, case7
      tuple_get_item_node = fn->body.as<TupleGetItemNode>();
    }

    // it can be a call node for add op or conv2d op
    //   OR it can be a TupleGetItem node followed by a batch_norm
    if (tuple_get_item_node != nullptr) {
      // for case3, case4, case7, case8
      ICHECK(tuple_get_item_node);
      ICHECK(tuple_get_item_node->index == 0);
      current_call = tuple_get_item_node->tuple.as<CallNode>();

      ICHECK(backend::IsOp(current_call, "nn.batch_norm"));
      nodes.batch_norm = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    ICHECK(current_call);
    if (backend::IsOp(current_call, "add")) {
      nodes.add = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    ICHECK(backend::IsOp(current_call, "nn.conv2d"));
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
   * \brief Extract sum2d nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite sum2d nodes.
   */
  static CompositeSum2DNode UnpackCompositeSum2D(const CallNode* call) {
    CompositeSum2DNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn);

    // Traverse composite convolution function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    ICHECK(backend::IsOp(current_call, "add"));
    nodes.add = current_call;

    return nodes;
  }

  /*!
   * \brief Extract maxpool nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite maxpool nodes.
   */
  static CompositePoolNode UnpackCompositePool(const CallNode* call,
                                               const std::string& mrvlLayerName) {
    CompositePoolNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn);

    // Traverse composite maxpool function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (mrvlLayerName == "Maxpool2D") {
      ICHECK(backend::IsOp(current_call, "nn.max_pool2d"));
    } else {
      ICHECK(mrvlLayerName == "Avgpool2D");
      ICHECK(backend::IsOp(current_call, "nn.avg_pool2d"));
    }
    nodes.pool = current_call;
    if (!current_call->args.empty() && current_call->args[0]->IsInstance<CallNode>()) {
      current_call = current_call->args[0].as<CallNode>();
      if (backend::IsOp(current_call, "nn.pad")) {
        nodes.pad = current_call;
      }
    }
    return nodes;
  }

  void GetInputTensorShapeViaArg0(const CallNode* call_node_ptr,
                                  std::vector<int64_t>* tensor_shape) {
    ICHECK(!call_node_ptr->args.empty());
    const TensorTypeNode* tensor_type = nullptr;
    if (call_node_ptr->args[0].as<CallNode>()) {
      const auto* arg0 = call_node_ptr->args[0].as<CallNode>();
      tensor_type = arg0->checked_type_.as<TensorTypeNode>();
    } else if (call_node_ptr->args[0].as<VarNode>()) {
      const auto* arg0 = call_node_ptr->args[0].as<VarNode>();
      ICHECK((arg0 != nullptr) && arg0->IsInstance<VarNode>());
      tensor_type = arg0->checked_type_.as<TensorTypeNode>();
    } else {
      LOG(INFO) << "TVM Mrvl runtime does not support calls to "
                << call_node_ptr->args[0]->GetTypeKey();
    }

    ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>());
    // use only data types supported by json.h (e.g., int or int64_t or size_t)
    for (IndexExpr dim_val : tensor_type->shape) {
      tensor_shape->push_back(*(tir::as_const_int(dim_val)));
    }
  }

  void GetTensorShape(const VarNode* var_node_ptr, std::vector<int64_t>* tensor_shape) {
    ICHECK((var_node_ptr != nullptr) && var_node_ptr->IsInstance<VarNode>());
    const TensorTypeNode* tensor_type = var_node_ptr->checked_type_.as<TensorTypeNode>();
    ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>());
    // use only data types supported by json.h (e.g., int or int64_t or size_t)
    for (IndexExpr dim_val : tensor_type->shape) {
      tensor_shape->push_back(*(tir::as_const_int(dim_val)));
    }
  }

  void GetOutputTensorShape(const CallNode* call_node_ptr, std::vector<int64_t>* tensor_shape) {
    ICHECK(call_node_ptr != nullptr);
    const TensorTypeNode* tensor_type = call_node_ptr->checked_type_.as<TensorTypeNode>();
    ICHECK((tensor_type != nullptr) && tensor_type->IsInstance<TensorTypeNode>());
    // use only data types supported by json.h (e.g., int or int64_t or size_t)
    for (IndexExpr dim_val : tensor_type->shape) {
      tensor_shape->push_back(*(tir::as_const_int(dim_val)));
    }
  }

  /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlConv2DLayer(const CallNode* cn) {
    CompositeConvNode nodes = UnpackCompositeConvolution(cn);
    const auto* conv_attrs = nodes.conv->attrs.as<Conv2DAttrs>();
    ICHECK(conv_attrs);

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    // data input tensor
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    // weight tensor
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    if (nodes.add) {
      // bias tensor
      inputs.push_back(VisitExpr(nodes.add->args[1])[0]);
    }

    // Distinguish between normal and depth-wise convolution
    std::string name;
    std::string mrvlLayerName = "";
    std::string data_layout = conv_attrs->data_layout;
    std::string kernel_layout = conv_attrs->kernel_layout;
    std::string out_layout = conv_attrs->out_layout;
    int groups = conv_attrs->groups;
    if ((groups != 1) && conv_attrs->channels.defined() &&
        tvm::tir::ExprDeepEqual()(conv_attrs->channels, conv_attrs->groups)) {
      name = "nn.dw_conv2d_nhwc2nhwc";
      mrvlLayerName = "DW_Conv2D";
      ICHECK(kernel_layout == "IHWO")
          << "Kernel layout must be IHWO, has the module been pre-processed correctly?";
    } else {
      name = "nn.conv2d_nhwc2nhwc";
      mrvlLayerName = "Conv2D";
      ICHECK(data_layout == "NHWC")
          << "Data layout must be NHWC, has the module been pre-processed correctly?";
      ICHECK(kernel_layout == "OHWI")
          << "Kernel layout must be OHWI, has the module been pre-processed correctly?";
      ICHECK(out_layout == "NHWC")
          << "Out layout must be NHWC, has the module been pre-processed correctly?";
    }
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    // following attributes will be set in json_node:
    // - strides, paddings, dilation,
    // - groups
    // - data_layout, kernel_layout, out_layout
    SetCallNodeAttribute(json_node, nodes.conv);

    // add other mrvl-specific attributes
    SetMrvlSpecificJsonNodeAttrs(
        json_node, cn, nodes.pad, nullptr /* no cn_pool */, nodes.conv, nullptr /* no cn_fc */,
        nodes.add, nodes.batch_norm, nodes.activation, mrvlLayerName,
        "" /* data_layout given in nodes.conv attrs */,
        "" /* kernel_layout given in nodes.conv attrs */,
        "" /* out_layout given in nodes.conv attrs */, "---O" /* if node.bias: as bias_layout */,
        "relu" /* if node.activation: as activation_op */, nodes.op_names);

    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite sum2d.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlSum2DLayer(const CallNode* cn) {
    CompositeSum2DNode nodes = UnpackCompositeSum2D(cn);
    ICHECK(nodes.add != nullptr) << "attribute add can't be nullptr";
    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    // data input tensor 1
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    // data input tensor 2
    inputs.push_back(VisitExpr(cn->args[1])[0]);
    std::string mrvlLayerName = "Sum2D";
    std::string name = "sum2d";
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);

    // add other mrvl-specific attributes
    SetMrvlSpecificJsonNodeAttrs(
        json_node, cn, nullptr, nullptr /* no cn_pool */, nullptr /* no cn_conv */,
        nullptr /* no cn_fc */, nodes.add, nullptr /* no cn_batch_norm */, nodes.activation,
        mrvlLayerName, "NHWC" /* data_layout */, "" /* kernel_layout */, "NHWC" /* out_layout */,
        "" /* bias_layout */, "relu" /* activation_op */, nodes.op_names);
    return json_node;
  }

  /*!
   * \brief Extract fc nodes from a composite function.
   *
   * \param call The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeFcNode UnpackCompositeFc(const CallNode* call) {
    CompositeFcNode nodes{};
    const auto* fn = call->op.as<FunctionNode>();
    ICHECK(fn);

    // Traverse composite fc function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.relu")) {
      nodes.activation = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "add")) {
      nodes.add = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    ICHECK(backend::IsOp(current_call, "nn.dense"));
    nodes.fc = current_call;
    return nodes;
  }

  /*!
   * \brief Create a JSON representation of a composite fc (fully-connected) operator.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlFcLayer(const CallNode* cn) {
    CompositeFcNode nodes = UnpackCompositeFc(cn);
    std::string name = "nn.fc_ni2no";
    std::string mrvlLayerName = "FC";
    std::string data_layout = "NC";
    std::string kernel_layout = "OI";
    std::string out_layout = "NC";

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.fc->args[1])[0]);
    if (nodes.add) {
      inputs.push_back(VisitExpr(nodes.add->args[1])[0]);
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.fc);

    // add other mrvl-specific attributes
    SetMrvlSpecificJsonNodeAttrs(json_node, cn, nullptr /* no node.pad */, nullptr /* no cn_pool */,
                                 nullptr /* no cn_conv */, nodes.fc, nodes.add,
                                 nullptr /* no cn_batch_norm */, nodes.activation, mrvlLayerName,
                                 "NC" /* data_layout */, "OI" /* kernel_layout */,
                                 "NC" /* out_layout */, "-O" /* if node.bias: as bias_layout */,
                                 "relu" /* if node.activation: as activation_op */, nodes.op_names);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite (global) maxpooling operator.
   *
   * A composite function is only created when using the uint8 datatype for these operators.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlMaxpool2DLayer(const CallNode* cn) {
    std::string mrvlLayerName = "Maxpool2D";
    std::string name = "nn.maxpool2d_nhwc2nhwc";
    CompositePoolNode nodes = UnpackCompositePool(cn, mrvlLayerName);
    const auto* maxpool_attr = nodes.pool->attrs.as<MaxPool2DAttrs>();
    ICHECK(maxpool_attr);
    ICHECK(maxpool_attr->layout == "NHWC")
        << "Layout must be NHWC, has the module been pre-processed correctly?";

    std::string data_layout = maxpool_attr->layout;
    std::string out_layout = maxpool_attr->layout;

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.pool);

    // add other mrvl-specific attributes
    SetMrvlSpecificJsonNodeAttrs(
        json_node, cn, nodes.pad, nodes.pool, nullptr /* no cn_conv */, nullptr /* no cn_fc */,
        nullptr /* no cn_add */, nullptr /* no cn_batch_norm */, nullptr /* no cn_activation */,
        mrvlLayerName, "NHWC" /* data_layout */, "HW" /* kernel_layout */, "NHWC" /* out_layout */,
        "" /* bias_layout */, "" /* activation_op */, nodes.op_names);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite (global) avgpooling operator.
   *
   * A composite function is only created when using the uint8 datatype for these operators.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeMrvlAvgpool2DLayer(const CallNode* cn) {
    std::string mrvlLayerName = "Avgpool2D";
    std::string name = "nn.avgpool2d_nhwc2nhwc";
    CompositePoolNode nodes = UnpackCompositePool(cn, mrvlLayerName);

    const auto* avgpool_attr = nodes.pool->attrs.as<AvgPool2DAttrs>();
    ICHECK(avgpool_attr);
    ICHECK(avgpool_attr->layout == "NHWC")
        << "Layout must be NHWC, has the module been pre-processed correctly?";

    std::string data_layout = avgpool_attr->layout;
    std::string out_layout = avgpool_attr->layout;

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.pool);

    // add other mrvl-specific attributes
    SetMrvlSpecificJsonNodeAttrs(
        json_node, cn, nodes.pad, nodes.pool, nullptr /* no cn_conv */, nullptr /* no cn_fc */,
        nullptr /* no cn_add */, nullptr /* no cn_batch_norm */, nullptr /* no cn_activation */,
        mrvlLayerName, "NHWC" /* data_layout */, "HW" /* kernel_layout */, "NHWC" /* out_layout */,
        "" /* bias_layout */, "" /* activation_op */, nodes.op_names);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite (global) maxpooling operator.
   *
   * A composite function is only created when using the uint8 datatype for these operators.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateMrvlLayer4OpNode(const CallNode* cn) {
    const auto* op_node = cn->op.as<OpNode>();
    ICHECK(op_node);
    String op_name = tvm::Op::GetOpName(GetRef<Op>(op_node));

    std::string name = op_name;
    std::string mrvlLayerName = op_name;
    std::string data_layout = "";
    std::string out_layout = "";
    if (op_name == "transpose") {
      // do nothing for now
    } else if ((op_name == "reshape") || (op_name == "nn.batch_flatten")) {
      // FIXME: hard coded for now -- when input data dim is 4D and output dim is 2D
      {
        // check for cases currently support
        std::vector<int64_t> layout_vec;
        GetInputTensorShapeViaArg0(cn, &layout_vec);
        ICHECK(layout_vec.size() == 4)
            << "Reshape or nn.batch_flatten with input tensor dim != 4 is not supported yet.";
        layout_vec.clear();
        GetOutputTensorShape(cn, &layout_vec);
        ICHECK(layout_vec.size() == 2)
            << "Reshape or nn.batch_flatten with output tensor dim != 2 is not supported yet.";
      }
      data_layout = "NHWC";
      out_layout = "NC";
    } else if (op_name == "layout_transform") {
      auto layout_transform_attr = cn->attrs.as<LayoutTransformAttrs>();
      data_layout = layout_transform_attr->src_layout;
      out_layout = layout_transform_attr->dst_layout;
    } else {
      LOG(FATAL) << "Can't handle this OpNode: " << AsText(GetRef<Call>(cn), false);
    }

    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    struct FrontendOpNames op_names;
    JsonNodeSetAttr(json_node, "first_op_name", {op_names.first_op_name});
    JsonNodeSetAttr(json_node, "last_op_name", {op_names.last_op_name});
    setMrvlLayerCommonAttrs(json_node, cn, func_name_, mrvlLayerName, data_layout,
                            "" /* no kernel_layout */, out_layout);
    return json_node;
  }
};

#endif

/*!
 * \brief Create a runtime module for Mrvl.
 *
 * This consists of a series of "serialized functions" which each represent a
 * sub-graph to be computed by Mrvl and will each be executed independently from
 * one another. Each function consists of serialized JSON describing the sub-graph
 * and serialized constant tensors.
 *
 * \note The Mrvl runtime module only supports a single operator per
 * sub-graph currently.
 *
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module MrvlCompiler(const ObjectRef& ref) {
#ifdef USE_JSON_RUNTIME

  // "per mrvl layer" MrvlCompiler call
  // - i.e., this is not a per mrvl "network" call
  if (!g_mrvlExtJsonObjInstantized) {
    // For the Mrvl BYOC flow's GraphExecutorCodegen() object, we need to register the
    //   Mrvl-BYOC's external JSON callback function in order to generate JSON files
    //   following the Mrvl-BYOC format
    // - TODO(ccjoechou): not sure this the best place to instantiate a MrvlExtJson object,
    //   which also register the callback function
    InstantiateMrvlExtJsonObj();
  }

  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  MrvlJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto param_names = serializer.GetParams();
  const auto* pf = runtime::Registry::Get("runtime.mrvl_runtime_create");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  runtime::Module lib = (*pf)(func_name, graph_json, param_names);

  return lib;

#else

  // TODO(ccjoechou): TBA if needed -- follow "per layer" C-codegen example

#endif
}

// NOTE: called by compile_engine.cc CompileEngineImpl::LowerExternalFunctions()
TVM_REGISTER_GLOBAL("relay.ext.mrvl").set_body_typed(MrvlCompiler);

/*!
 * \brief Check whether Mrvl graph executor is used.
 *
 * \return True if Mrvl graph executor is enabled, False if not.
 */
inline constexpr bool IsMrvlRuntimeEnabled() { return true; }

TVM_REGISTER_GLOBAL("relay.op.is_mrvl_runtime_enabled").set_body_typed(IsMrvlRuntimeEnabled);

}  // namespace mrvl
}  // namespace contrib

}  // namespace relay
}  // namespace tvm
