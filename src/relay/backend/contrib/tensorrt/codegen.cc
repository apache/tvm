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
 * \file src/relay/backend/contrib/tensorrt/codegen.cc
 * \brief Implementation of the TensorRT JSON serializer.
 */
#include <tvm/ir/module.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/type.h>

#include <memory>
#include <string>
#include <vector>

#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

#if TVM_GRAPH_EXECUTOR_TENSORRT
#include "NvInfer.h"
#endif

namespace tvm {
namespace relay {
namespace contrib {

/*! \brief Attributes to store the compiler options for TensorRT. */
struct TensorRTCompilerConfigNode : public tvm::AttrsNode<TensorRTCompilerConfigNode> {
  Array<Integer> tensorrt_version;
  bool use_implicit_batch;
  size_t max_workspace_size;
  bool remove_no_mac_subgraphs;

  TVM_DECLARE_ATTRS(TensorRTCompilerConfigNode, "ext.attrs.TensorRTCompilerConfigNode") {
    TVM_ATTR_FIELD(tensorrt_version)
        .describe("TensorRT version as (major, minor, patch).")
        .set_default(Array<Integer>({6, 0, 1}));
    TVM_ATTR_FIELD(use_implicit_batch).set_default(true);
    TVM_ATTR_FIELD(max_workspace_size).set_default(size_t(1) << 30);
    TVM_ATTR_FIELD(remove_no_mac_subgraphs).set_default(false);
  }
};

class TensorRTCompilerConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(TensorRTCompilerConfig, Attrs,
                                            TensorRTCompilerConfigNode);
};

TVM_REGISTER_NODE_TYPE(TensorRTCompilerConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.tensorrt.options", TensorRTCompilerConfig);

/*!
 * \brief Generates an TensorRTModule from a relay expression by serializing the expression to a
 * json representation. TensorRT is not required here because use of TensorRT APIs is deferred until
 * runtime.
 */
class TensorRTJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  TensorRTJSONSerializer(const std::string& symbol, const Expr& expr)
      : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) {
    std::string name;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else {
      return JSONSerializer::VisitExpr_(cn);
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    if (name == "nn.pad") {
      SetPadNodeAttribute(node, cn);
    } else if (name == "strided_slice") {
      SetStridedSliceNodeAttribute(node, cn);
    } else if (name == "split") {
      SetSplitNodeAttribute(node, cn);
    } else {
      SetCallNodeAttribute(node, cn);
    }
    // These attributes are global to the whole module.
    SaveGlobalAttributes(node);
    return AddNode(node, GetRef<Expr>(cn));
  }

  void SetPadNodeAttribute(std::shared_ptr<JSONGraphNode> node, const CallNode* cn) {
    const auto* pad_attr = cn->attrs.as<PadAttrs>();
    ICHECK(pad_attr);
    auto p = pad_attr->pad_width;
    const int dim_h = (p.size() == 5) ? 3 : 2;
    const int dim_w = (p.size() == 5) ? 4 : 3;
    std::vector<std::string> padding = {std::to_string(p[dim_h][0].as<IntImmNode>()->value),
                                        std::to_string(p[dim_w][0].as<IntImmNode>()->value),
                                        std::to_string(p[dim_h][1].as<IntImmNode>()->value),
                                        std::to_string(p[dim_w][1].as<IntImmNode>()->value)};
    std::vector<dmlc::any> padding_attr;
    padding_attr.emplace_back(padding);
    node->SetAttr("padding", padding_attr);
  }

  void SetStridedSliceNodeAttribute(std::shared_ptr<JSONGraphNode> node, const CallNode* cn) {
    const auto* attrs = cn->attrs.as<StridedSliceAttrs>();
    ICHECK(attrs && attrs->begin && attrs->end && attrs->strides)
        << "StridedSlice must have static begin, end, and strides.";
    const bool default_strides =
        !attrs->strides.value().defined() || attrs->strides.value().size() == 0;
    auto ishape = backend::GetShape(cn->args[0]->checked_type());

    auto process_slice_index = [](Integer x, int default_value, int dim_value) {
      if (!x.defined()) return default_value;
      int value = x.as<IntImmNode>()->value;
      if (value < 0) value += dim_value;
      return value;
    };

    std::vector<std::string> start, size, strides;
    for (size_t i = 0; i < attrs->begin.value().size(); ++i) {
      const int begin_value = process_slice_index(attrs->begin.value()[i], 0, ishape[i]);
      ICHECK_GE(begin_value, 0);
      start.push_back(std::to_string(begin_value));
      const int stride_value = (default_strides || i >= attrs->strides.value().size() ||
                                !attrs->strides.value()[i].defined())
                                   ? 1
                                   : attrs->strides.value()[i].as<IntImmNode>()->value;
      ICHECK_GT(stride_value, 0);
      strides.push_back(std::to_string(stride_value));
      int size_value;
      if (attrs->slice_mode == "end") {
        const int end_value = process_slice_index(attrs->end.value()[i], ishape[i], ishape[i]);
        size_value = (end_value - begin_value + stride_value - 1) / stride_value;
      } else if (attrs->slice_mode == "size") {
        // with slice_mode = "size", attrs->end_value mean the size of the slice
        int end_value = attrs->end.value()[i].as<IntImmNode>()->value;
        size_value = (end_value == -1) ? ishape[i] - begin_value : end_value;
      } else {
        LOG(FATAL) << "Unexpected slice_mode " << attrs->slice_mode << ", expected end or size";
        throw;
      }
      ICHECK_GT(size_value, 0);
      size.push_back(std::to_string(size_value));
    }
    std::vector<dmlc::any> start_attr, size_attr, strides_attr;
    start_attr.emplace_back(start);
    size_attr.emplace_back(size);
    strides_attr.emplace_back(strides);
    node->SetAttr("start", start_attr);
    node->SetAttr("size", size_attr);
    node->SetAttr("strides", strides_attr);
  }

  void SetSplitNodeAttribute(std::shared_ptr<JSONGraphNode> node, const CallNode* cn) {
    const auto* split_attr = cn->attrs.as<SplitAttrs>();
    ICHECK(split_attr);

    std::vector<std::string> indices_or_sections;
    std::vector<std::string> mode;
    std::vector<std::string> axis = {std::to_string(split_attr->axis)};
    if (const IntImmNode* sections = split_attr->indices_or_sections.as<IntImmNode>()) {
      mode.emplace_back("sections");
      indices_or_sections.emplace_back(std::to_string(sections->value));
    } else {
      mode.emplace_back("indices");
      auto indices = Downcast<tvm::Array<Integer>>(split_attr->indices_or_sections);
      for (const auto& i : indices) {
        indices_or_sections.emplace_back(std::to_string(i->value));
      }
    }

    std::vector<dmlc::any> indices_or_sections_attr;
    std::vector<dmlc::any> mode_attr;
    std::vector<dmlc::any> axis_attr;
    indices_or_sections_attr.emplace_back(indices_or_sections);
    mode_attr.emplace_back(mode);
    axis_attr.emplace_back(axis);
    node->SetAttr("indices_or_sections", indices_or_sections_attr);
    node->SetAttr("mode", mode_attr);
    node->SetAttr("axis", axis_attr);
  }

  void SaveGlobalAttributes(std::shared_ptr<JSONGraphNode> node) {
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfig<TensorRTCompilerConfig>("relay.ext.tensorrt.options");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<TensorRTCompilerConfig>();
    }
    ICHECK_EQ(cfg.value()->tensorrt_version.size(), 3);
    std::vector<std::string> tensorrt_version = {std::to_string(cfg.value()->tensorrt_version[0]),
                                                 std::to_string(cfg.value()->tensorrt_version[1]),
                                                 std::to_string(cfg.value()->tensorrt_version[2])};
    std::vector<std::string> use_implicit_batch = {std::to_string(cfg.value()->use_implicit_batch)};
    std::vector<std::string> max_workspace_size = {std::to_string(cfg.value()->max_workspace_size)};
    std::vector<dmlc::any> tensorrt_version_attr, use_implicit_batch_attr, max_workspace_size_attr;
    tensorrt_version_attr.emplace_back(tensorrt_version);
    use_implicit_batch_attr.emplace_back(use_implicit_batch);
    max_workspace_size_attr.emplace_back(max_workspace_size);
    node->SetAttr("tensorrt_version", tensorrt_version_attr);
    node->SetAttr("use_implicit_batch", use_implicit_batch_attr);
    node->SetAttr("max_workspace_size", max_workspace_size_attr);
  }
};

/*!
 * \brief Create a runtime module for TensorRT.
 * \param ref The ext_func Relay expression/module to be executed using extern ops.
 * \return A runtime module.
 */
runtime::Module TensorRTCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>()) << "The input ref is expected to be a Relay function.";
  Function func = Downcast<Function>(ref);
  std::string func_name = backend::GetExtSymbol(func);

  TensorRTJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto param_names = serializer.GetParams();
  const auto* pf = runtime::Registry::Get("runtime.tensorrt_runtime_create");
  ICHECK(pf != nullptr) << "Cannot find TensorRT runtime module create function.";
  runtime::Module lib = (*pf)(func_name, graph_json, param_names);
  return lib;
}

TVM_REGISTER_GLOBAL("relay.ext.tensorrt").set_body_typed(TensorRTCompiler);

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

TVM_REGISTER_GLOBAL("relay.op.is_tensorrt_runtime_enabled")
    .set_body_typed(IsTensorRTRuntimeEnabled);
TVM_REGISTER_GLOBAL("relay.op.get_tensorrt_version").set_body_typed(GetTensorRTVersion);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
