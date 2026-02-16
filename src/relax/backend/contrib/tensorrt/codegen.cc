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
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/attrs/nn.h>
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
struct TensorRTCompilerConfigNode : public AttrsNodeReflAdapter<TensorRTCompilerConfigNode> {
  ffi::Array<Integer> tensorrt_version;
  bool use_implicit_batch;
  size_t max_workspace_size;
  bool remove_no_mac_subgraphs;
  bool use_fp16;
  bool use_uint8;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TensorRTCompilerConfigNode>()
        .def_ro("tensorrt_version", &TensorRTCompilerConfigNode::tensorrt_version,
                "TensorRT version as (major, minor, patch).",
                refl::DefaultValue(ffi::Array<Integer>({6, 0, 1})))
        .def_ro("use_implicit_batch", &TensorRTCompilerConfigNode::use_implicit_batch,
                "Use implicit batch", refl::DefaultValue(true))
        .def_ro("max_workspace_size", &TensorRTCompilerConfigNode::max_workspace_size,
                "Max workspace size", refl::DefaultValue(size_t(1) << 30))
        .def_ro("remove_no_mac_subgraphs", &TensorRTCompilerConfigNode::remove_no_mac_subgraphs,
                "Remove no-mac subgraphs", refl::DefaultValue(false))
        .def_ro("use_fp16", &TensorRTCompilerConfigNode::use_fp16, "Use FP16",
                refl::DefaultValue(false))
        .def_ro("use_uint8", &TensorRTCompilerConfigNode::use_uint8, "Use uint8",
                refl::DefaultValue(false));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("relax.ext.attrs.TensorRTCompilerConfig",
                                    TensorRTCompilerConfigNode, BaseAttrsNode);
};

class TensorRTCompilerConfig : public Attrs {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(TensorRTCompilerConfig, Attrs,
                                                TensorRTCompilerConfigNode);
};

TVM_FFI_STATIC_INIT_BLOCK() { TensorRTCompilerConfigNode::RegisterReflection(); }

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
  explicit TensorRTJSONSerializer(ffi::Map<Constant, ffi::String> constant_names,
                                  ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* call_node) final {
    // The call must be to an inline "Composite" function
    const auto* fn_var = call_node->op.as<VarNode>();
    ICHECK(fn_var);
    const auto fn = Downcast<Function>(bindings_[ffi::GetRef<Var>(fn_var)]);

    auto opt_composite = fn->GetAttr<ffi::String>(attr::kComposite);
    ICHECK(opt_composite.has_value());
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

    return AddNode(node, ffi::GetRef<Expr>(call_node));
  }

  static void SaveGlobalAttributes(std::shared_ptr<JSONGraphNode> node) {
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfig<TensorRTCompilerConfig>("relax.ext.tensorrt.options");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<TensorRTCompilerConfig>();
    }
    ICHECK_EQ(cfg.value()->tensorrt_version.size(), 3);
    ffi::Array<int64_t> tensorrt_version = {cfg.value()->tensorrt_version[0].IntValue(),
                                            cfg.value()->tensorrt_version[1].IntValue(),
                                            cfg.value()->tensorrt_version[2].IntValue()};
    node->SetAttr("tensorrt_version", std::move(tensorrt_version));
    node->SetAttr("use_implicit_batch", static_cast<int64_t>(cfg.value()->use_implicit_batch));
    node->SetAttr("max_workspace_size", static_cast<int64_t>(cfg.value()->max_workspace_size));
    node->SetAttr("use_fp16", static_cast<int64_t>(cfg.value()->use_fp16));
    node->SetAttr("use_uint8", static_cast<int64_t>(cfg.value()->use_uint8));
  }

 private:
  /*! \brief The bindings to look up composite functions. */
  ffi::Map<Var, Expr> bindings_;
};

void CollectFromCompositeFunctionBody::VisitExpr_(const ConstantNode* constant_node) {
  for (const auto& entry : serializer_->VisitExpr(ffi::GetRef<Constant>(constant_node))) {
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
ffi::Array<ffi::Module> TensorRTCompiler(ffi::Array<Function> functions,
                                         ffi::Map<ffi::String, ffi::Any> /*unused*/,
                                         ffi::Map<Constant, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;
  for (const auto& func : functions) {
    VLOG(1) << "TensorRT partition:" << std::endl << func;
    TensorRTJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    std::string graph_json = serializer.GetJSON();
    VLOG(1) << "TensorRT JSON:" << std::endl << graph_json;
    auto constant_names = serializer.GetConstantNames();
    const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.tensorrt_runtime_create");
    std::string func_name = GetExtSymbol(func);
    VLOG(1) << "Creating tensorrt ffi::Module for '" << func_name << "'";
    compiled_functions.push_back(pf(func_name, graph_json, constant_names).cast<ffi::Module>());
  }
  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.tensorrt", TensorRTCompiler);
}

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
ffi::Array<Integer> GetTensorRTVersion() {
#if TVM_GRAPH_EXECUTOR_TENSORRT
  return {Integer(NV_TENSORRT_MAJOR), Integer(NV_TENSORRT_MINOR), Integer(NV_TENSORRT_PATCH)};
#else
  return {};
#endif  // TVM_GRAPH_EXECUTOR_TENSORRT
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.is_tensorrt_runtime_enabled", IsTensorRTRuntimeEnabled)
      .def("relax.get_tensorrt_version", GetTensorRTVersion);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
