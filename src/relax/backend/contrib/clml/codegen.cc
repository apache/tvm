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
 * \file src/relax/backend/contrib/clml/codegen.cc
 * \brief Implementation of the OpenCLML JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
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

/*! \brief Attributes to store the compiler options for OpenCLML. */
struct OpenCLMLCompilerConfigNode : public tvm::AttrsNode<OpenCLMLCompilerConfigNode> {
  Integer clml_version;

  TVM_DECLARE_ATTRS(OpenCLMLCompilerConfigNode, "relax.ext.attrs.OpenCLMLCompilerConfigNode") {
    TVM_ATTR_FIELD(clml_version)
        .describe("OpenCLML version as (major, minor, patch).")
        .set_default(Integer(3));
  }
};

class OpenCLMLCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(OpenCLMLCompilerConfig, Attrs,
                                            OpenCLMLCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(OpenCLMLCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relax.ext.clml.options", OpenCLMLCompilerConfig);

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using OpAttrExtractor = backend::contrib::OpAttrExtractor;
using JSONSerializer = backend::contrib::JSONSerializer;

class OpenCLMLJSONSerializer;

/*!
 * \brief Collect the constants and attributes from all operator calls in the body
 * of a "Composite" function.
 */
class CollectCLMLFromCompositeFunctionBody : public ExprVisitor {
 public:
  explicit CollectCLMLFromCompositeFunctionBody(OpenCLMLJSONSerializer* serializer)
      : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const ConstantNode* constant_node) final;
  void VisitExpr_(const CallNode* call_node) final;

  void SetGenericAttributes(const CallNode* call_node) {
    if (backend::IsOp(call_node, "relax.nn.relu")) {
      std::vector<std::string> activation_type = {"relu"};
      std::vector<dmlc::any> act_attr;
      act_attr.emplace_back(activation_type);
      node_->SetAttr("activation_type", act_attr);
    }

    OpAttrExtractor extractor(node_);
    const Object* attr_obj = call_node->attrs.get();
    extractor.Extract(const_cast<Object*>(attr_obj));
  }

  OpenCLMLJSONSerializer* serializer_;
  /*! \brief Accumulated translated arguments. */
  std::vector<JSONGraphNodeEntry> args_;
  /*!
   * \brief Temporary node into which we'll accumulate attributes. Ideally this would be the
   * final JSONGraphNode however we don't yet know how many inputs that will have.
   */
  JSONGraphObjectPtr node_;
};

/*!
 * \brief Generates an OpenCLMLModule from a relax expression by serializing the expression to a
 * json representation. OpenCLML is not required here because use of OpenCLML APIs is deferred until
 * runtime.
 */
class OpenCLMLJSONSerializer : public JSONSerializer {
 public:
  explicit OpenCLMLJSONSerializer(Map<Constant, String> constant_names, Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

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

    if (backend::EndsWithPattern(name, "nn.conv2d") ||
        backend::EndsWithPattern(name, "nn.pad_conv2d") ||
        backend::EndsWithPattern(name, "nn.pad_conv2d_transpose")) {
      node = CreateCompositeConvJSONNode(call_node);
    } else {
      // Collect the constants and attributes of all operator calls inside the composite body.
      CollectCLMLFromCompositeFunctionBody collector(this);
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

      // Capture global settings on the JSON node.
      SaveGlobalAttributes(node);

      VLOG(1) << name << " has " << node->GetInputs().size() << " inputs";
    }

    return AddNode(node, GetRef<Expr>(call_node));
  }

  /*!
   * \brief Extract convolution nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  CompositeConvNode UnpackCompositeConvolution(const CallNode* cn) {
    CompositeConvNode nodes{};

    const auto* fn_var = cn->op.as<VarNode>();
    ICHECK(fn_var);
    const auto fn = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    auto opt_composite = fn->GetAttr<String>(attr::kComposite);
    ICHECK(opt_composite.defined());

    nodes.pad = backend::TryGetOpInFunction(fn, "relax.nn.pad");
    nodes.conv = backend::TryGetOpInFunction(fn, "relax.nn.conv2d");

    if (!nodes.conv) {
      nodes.conv = backend::TryGetOpInFunction(fn, "relax.nn.conv2d_transpose");
    }
    ICHECK(nodes.conv) << "No Convolution op found in composite function";
    nodes.bn = backend::TryGetOpInFunction(fn, "relax.nn.batch_norm");
    nodes.bias = backend::TryGetOpInFunction(fn, "relax.add");
    nodes.activation = backend::TryGetOpInFunction(fn, "relax.nn.relu");
    nodes.act_type = "relu";
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

    const auto* fn_var = cn->op.as<VarNode>();
    ICHECK(fn_var);
    const auto fn = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    auto opt_composite = fn->GetAttr<String>(attr::kComposite);
    ICHECK(opt_composite.defined());
    std::string name = opt_composite.value();

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

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
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
      // Pad layout for TVM: dimension wise pre and post padding.
      // CLML takes dimension wise pre-padding followed by dimension wise post-padding for W, H.
      std::vector<std::string> padding = {std::to_string(p[4].as<IntImmNode>()->value),
                                          std::to_string(p[6].as<IntImmNode>()->value),
                                          std::to_string(p[5].as<IntImmNode>()->value),
                                          std::to_string(p[7].as<IntImmNode>()->value)};
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

  static void SaveGlobalAttributes(std::shared_ptr<JSONGraphNode> node) {
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfig<OpenCLMLCompilerConfig>("relax.ext.clml.options");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<OpenCLMLCompilerConfig>();
    }
    std::vector<std::string> clml_version = {std::to_string(cfg.value()->clml_version.IntValue())};
    std::vector<dmlc::any> clml_version_attr;
    clml_version_attr.emplace_back(clml_version);
    node->SetAttr("clml_version", clml_version_attr);
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  Map<Var, Expr> bindings_;
};

void CollectCLMLFromCompositeFunctionBody::VisitExpr_(const ConstantNode* constant_node) {
  for (const auto& entry : serializer_->VisitExpr(GetRef<Constant>(constant_node))) {
    args_.emplace_back(entry);
  }
}

void CollectCLMLFromCompositeFunctionBody::VisitExpr_(const CallNode* call_node) {
  SetGenericAttributes(call_node);
  ExprVisitor::VisitExpr_(call_node);
}

/*!
 * \brief Create runtime modules for OpenCLML.
 * \param functions The extern functions to be compiled via OpenCLML
 * \return Runtime modules.
 */
Array<runtime::Module> OpenCLMLCompiler(Array<Function> functions, Map<String, Any> /*unused*/,
                                        Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;
  for (const auto& func : functions) {
    VLOG(1) << "OpenCLML partition:" << std::endl << func;
    OpenCLMLJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    std::string graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.clml_runtime_create");
    std::string func_name = GetExtSymbol(func);
    VLOG(1) << "Creating clml runtime::Module for '" << func_name << "'";
    compiled_functions.push_back(pf(func_name, graph_json, constant_names).cast<runtime::Module>());
  }
  return compiled_functions;
}

TVM_FFI_REGISTER_GLOBAL("relax.ext.openclml").set_body_typed(OpenCLMLCompiler);

/*!
 * \brief Check whether OpenCLML graph executor is enabled.
 * \return True if enabled, False if not.
 */
inline constexpr bool IsOpenCLMLRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_CLML
  return true;
#else
  return false;
#endif  // TVM_GRAPH_EXECUTOR_CLML
}

/*!
 * \brief Get OpenCLML version that TVM is built against.
 * \return The OpenCLML SDK version.
 */
Integer GetOpenCLMLVersion() {
#if TVM_GRAPH_EXECUTOR_CLML
  return Integer(TVM_CLML_VERSION);
#else
  return Integer(3);
#endif  // TVM_GRAPH_EXECUTOR_CLML
}

TVM_FFI_REGISTER_GLOBAL("relax.is_openclml_runtime_enabled")
    .set_body_typed(IsOpenCLMLRuntimeEnabled);
TVM_FFI_REGISTER_GLOBAL("relax.get_openclml_version").set_body_typed(GetOpenCLMLVersion);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
