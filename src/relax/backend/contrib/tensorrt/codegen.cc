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
 * \file src/relax/backend/contrib/tensorrt/codegen.cc
 * \brief Implementation of the TensorRT JSON serializer.
 */
#include <tvm/ir/module.h>
// TODO(sunggg): add operator attribute when it's ready
// #include <tvm/relax/attrs/nn.h>
#include <tvm/relax/type.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../transform/utils.h"
#include "../codegen_json/codegen_json.h"
#include "../utils.h"

#if TVM_GRAPH_EXECUTOR_TENSORRT
#include "NvInfer.h"
#endif

namespace tvm {
namespace relax {
namespace contrib {

/*! \brief Attributes to store the compiler options for TensorRT. */
struct TensorRTCompilerConfigNode : public tvm::AttrsNode<TensorRTCompilerConfigNode> {
  Array<Integer> tensorrt_version;
  bool use_implicit_batch;
  size_t max_workspace_size;
  bool remove_no_mac_subgraphs;
  bool use_fp16;
  bool use_uint8;

  TVM_DECLARE_ATTRS(TensorRTCompilerConfigNode, "relax.ext.attrs.TensorRTCompilerConfigNode") {
    TVM_ATTR_FIELD(tensorrt_version)
        .describe("TensorRT version as (major, minor, patch).")
        .set_default(Array<Integer>({6, 0, 1}));
    TVM_ATTR_FIELD(use_implicit_batch).set_default(true);
    TVM_ATTR_FIELD(max_workspace_size).set_default(size_t(1) << 30);
    TVM_ATTR_FIELD(remove_no_mac_subgraphs).set_default(false);
    TVM_ATTR_FIELD(use_fp16).set_default(false);
    TVM_ATTR_FIELD(use_uint8).set_default(false);
  }
};

class TensorRTCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorRTCompilerConfig, Attrs,
                                            TensorRTCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(TensorRTCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relax.ext.tensorrt.options", TensorRTCompilerConfig);

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using OpAttrExtractor = backend::contrib::OpAttrExtractor;
using JSONSerializer = backend::contrib::JSONSerializer;

class TensorRTJSONSerializer;

/*!
 * \brief Collect the constants and attributes from all operator calls in the body
 * of a "Composite" function.
 */
class CollectFromCompositeFunctionBody : public ExprVisitor {
 public:
  explicit CollectFromCompositeFunctionBody(TensorRTJSONSerializer* serializer)
      : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const ConstantNode* constant_node) final;
  void VisitExpr_(const CallNode* call_node) final;

  void SetGenericAttributes(const CallNode* call_node) {
    OpAttrExtractor extractor(node_);
    const Object* attr_obj = call_node->attrs.get();
    extractor.Extract(const_cast<Object*>(attr_obj));
  }

  TensorRTJSONSerializer* serializer_;
  /*! \brief Accumulated translated arguments. */
  std::vector<JSONGraphNodeEntry> args_;
  /*!
   * \brief Temporary node into which we'll accumulate attributes. Ideally this would be the
   * final JSONGraphNode however we don't yet know how many inputs that will have.
   */
  JSONGraphObjectPtr node_;
};

/*!
 * \brief Generates an TensorRTModule from a relax expression by serializing the expression to a
 * json representation. TensorRT is not required here because use of TensorRT APIs is deferred until
 * runtime.
 */
class TensorRTJSONSerializer : public JSONSerializer {
 public:
  explicit TensorRTJSONSerializer(Map<Constant, String> constant_names, Map<Var, Expr> bindings)
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
    auto node = std::make_shared<JSONGraphNode>(name,
                                                /*op_type=*/"kernel", inputs,
                                                /*num_output=*/1);

    // Transfer attributes from the collector's node to the final node.
    node->CaptureAttrs(*collector.node_);

    // Capture global settings on the JSON node.
    SaveGlobalAttributes(node);

    VLOG(1) << name << " has " << node->GetInputs().size() << " inputs";

    return AddNode(node, GetRef<Expr>(call_node));
  }

  static void SaveGlobalAttributes(std::shared_ptr<JSONGraphNode> node) {
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfig<TensorRTCompilerConfig>("relax.ext.tensorrt.options");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<TensorRTCompilerConfig>();
    }
    ICHECK_EQ(cfg.value()->tensorrt_version.size(), 3);
    std::vector<std::string> tensorrt_version = {
        std::to_string(cfg.value()->tensorrt_version[0].IntValue()),
        std::to_string(cfg.value()->tensorrt_version[1].IntValue()),
        std::to_string(cfg.value()->tensorrt_version[2].IntValue())};
    std::vector<std::string> use_implicit_batch = {std::to_string(cfg.value()->use_implicit_batch)};
    std::vector<std::string> max_workspace_size = {std::to_string(cfg.value()->max_workspace_size)};
    std::vector<std::string> use_fp16 = {std::to_string(cfg.value()->use_fp16)};
    std::vector<std::string> use_uint8 = {std::to_string(cfg.value()->use_uint8)};
    std::vector<dmlc::any> tensorrt_version_attr, use_implicit_batch_attr, max_workspace_size_attr,
        use_fp16_attr, use_uint8_attr;
    tensorrt_version_attr.emplace_back(tensorrt_version);
    use_implicit_batch_attr.emplace_back(use_implicit_batch);
    max_workspace_size_attr.emplace_back(max_workspace_size);
    use_fp16_attr.emplace_back(use_fp16);
    use_uint8_attr.emplace_back(use_uint8);
    node->SetAttr("tensorrt_version", tensorrt_version_attr);
    node->SetAttr("use_implicit_batch", use_implicit_batch_attr);
    node->SetAttr("max_workspace_size", max_workspace_size_attr);
    node->SetAttr("use_fp16", use_fp16_attr);
    node->SetAttr("use_uint8", use_uint8_attr);
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
 * \brief Create runtime modules for TensorRT.
 * \param functions The extern functions to be compiled via TensorRT
 * \return Runtime modules.
 */
Array<runtime::Module> TensorRTCompiler(Array<Function> functions,
                                        Map<String, ObjectRef> /*unused*/,
                                        Map<Constant, String> constant_names) {
  Array<runtime::Module> compiled_functions;
  for (const auto& func : functions) {
    VLOG(1) << "TensorRT partition:" << std::endl << func;
    TensorRTJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    std::string graph_json = serializer.GetJSON();
    VLOG(1) << "TensorRT JSON:" << std::endl << graph_json;
    auto constant_names = serializer.GetConstantNames();
    const auto* pf = runtime::Registry::Get("runtime.tensorrt_runtime_create");
    ICHECK(pf != nullptr) << "Cannot find TensorRT runtime module create function.";
    std::string func_name = GetExtSymbol(func);
    VLOG(1) << "Creating tensorrt runtime::Module for '" << func_name << "'";
    compiled_functions.push_back((*pf)(func_name, graph_json, constant_names));
  }
  return compiled_functions;
}

TVM_REGISTER_GLOBAL("relax.ext.tensorrt").set_body_typed(TensorRTCompiler);

/*!
 * \brief Check whether TensorRT graph executor is enabled.
 * \return True if enabled, False if not.
 */
inline constexpr bool IsTensorRTRuntimeEnabled() {
#if TVM_GRAPH_EXECUTOR_TENSORRT
  return true;
#else
  return false;
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT
}

/*!
 * \brief Get TensorRT version that TVM is built against.
 * \return Array of three integers for major, minor, and patch, or empty array if TensorRT graph
 * runtime is not enabled.
 */
Array<Integer> GetTensorRTVersion() {
#if TVM_GRAPH_EXECUTOR_TENSORRT
  return {Integer(NV_TENSORRT_MAJOR), Integer(NV_TENSORRT_MINOR), Integer(NV_TENSORRT_PATCH)};
#else
  return {};
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT
}

TVM_REGISTER_GLOBAL("relax.is_tensorrt_runtime_enabled").set_body_typed(IsTensorRTRuntimeEnabled);
TVM_REGISTER_GLOBAL("relax.get_tensorrt_version").set_body_typed(GetTensorRTVersion);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
