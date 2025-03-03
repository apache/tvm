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

#include <tvm/ir/module.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../transform/utils.h"
#include "../codegen_json/codegen_json.h"
#include "tvm/relax/attrs/manipulate.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONSerializer = backend::contrib::JSONSerializer;
using JSONGraphNode = backend::contrib::JSONGraphNode;
using JSONGraphNodeEntry = backend::contrib::JSONGraphNodeEntry;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using NodeEntries = backend::contrib::NodeEntries;

class NNAPIJSONSerializer;

class CollectFromCompositeFunctionBody : public ExprVisitor {
 public:
  explicit CollectFromCompositeFunctionBody(NNAPIJSONSerializer* serializer)
      : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const CallNode* call_node) override;

  void SetPermuteDimsAttribute(const CallNode* call_node) {
    const auto* permute_dims_attr = call_node->attrs.as<PermuteDimsAttrs>();
    ICHECK(permute_dims_attr);
    if (permute_dims_attr->axes) {
      std::vector<std::string> axes;
      for (auto axis : permute_dims_attr->axes.value()) {
        axes.push_back(std::to_string(axis.IntValue()));
      }

      std::vector<dmlc::any> axes_attr;
      axes_attr.emplace_back(axes);
      node_->SetAttr("axes", axes_attr);
    }
  }

  void SetAstypeAttribute(const CallNode* call_node) {
    const auto* astype_attrs = call_node->attrs.as<AstypeAttrs>();
    ICHECK(astype_attrs);

    std::vector<dmlc::any> dtype_attr;
    auto dtype_str = runtime::DLDataType2String(astype_attrs->dtype);
    dtype_attr.emplace_back(std::vector<std::string>{dtype_str});
    node_->SetAttr("astype_dtype", dtype_attr);
  }

  void SetMeanAttribute(const CallNode* call_node) {
    const auto* mean_attrs = call_node->attrs.as<StatisticalAttrs>();
    ICHECK(mean_attrs);
    ICHECK(mean_attrs->axis.defined());

    {
      std::vector<std::string> axis;
      for (auto dim : mean_attrs->axis.value()) {
        axis.push_back(std::to_string(dim->value));
      }

      std::vector<dmlc::any> axis_attr;
      axis_attr.emplace_back(axis);
      node_->SetAttr("axis", axis_attr);
    }

    {
      const std::vector<std::string> keepdims{mean_attrs->keepdims ? "1" : "0"};
      std::vector<dmlc::any> keepdims_attr;
      keepdims_attr.emplace_back(keepdims);
      node_->SetAttr("keepdims", keepdims_attr);
    }
  }

  void SetConv2dAttribute(const CallNode* call_node) {
    const auto* conv2d_attr = call_node->attrs.as<Conv2DAttrs>();
    ICHECK(conv2d_attr) << "didn't catch attributes";

    std::vector<std::string> strides;
    if (!conv2d_attr->strides.empty()) {
      for (auto stride : conv2d_attr->strides) {
        const auto* stride_val = stride.as<IntImmNode>();
        ICHECK(stride_val) << "convertion failed";

        strides.push_back(std::to_string(stride_val->value));
      }
    } else {
      strides = {"1", "1"};
    }

    std::vector<std::string> padding;
    for (auto pad : conv2d_attr->padding) {
      const auto* padding_val = pad.as<IntImmNode>();

      padding.push_back(std::to_string(padding_val->value));
    }

    std::vector<std::string> groups;
    const int group_val = conv2d_attr->groups;
    groups.push_back(std::to_string(group_val));

    std::vector<dmlc::any> strides_attr;
    strides_attr.emplace_back(strides);
    node_->SetAttr("strides", strides_attr);

    std::vector<dmlc::any> padding_attr;
    padding_attr.emplace_back(padding);
    node_->SetAttr("padding", padding_attr);

    std::vector<dmlc::any> group_attr;
    group_attr.emplace_back(groups);
    node_->SetAttr("group", group_attr);
  }

  void SetMaxPool2dAttribute(const CallNode* call_node) {
    const auto* max_pool_2d_attr = call_node->attrs.as<Pool2DAttrs>();
    ICHECK(max_pool_2d_attr) << "didn't catch attributes";

    std::vector<std::string> strides;
    if (!max_pool_2d_attr->strides.empty()) {
      for (auto stride : max_pool_2d_attr->strides) {
        const auto* stride_val = stride.as<IntImmNode>();
        ICHECK(stride_val) << "convertion failed";

        strides.push_back(std::to_string(stride_val->value));
      }
    } else {
      strides.push_back("1");
      strides.push_back("1");
    }

    std::vector<std::string> padding;
    for (auto pad : max_pool_2d_attr->padding) {
      const auto* padding_val = pad.as<IntImmNode>();

      padding.push_back(std::to_string(padding_val->value));
    }

    std::vector<std::string> pool_size;
    for (auto size : max_pool_2d_attr->pool_size) {
      const auto* pooling_val = size.as<IntImmNode>();

      pool_size.push_back(std::to_string(pooling_val->value));
    }

    std::vector<dmlc::any> strides_attr;
    strides_attr.emplace_back(strides);
    node_->SetAttr("strides", strides_attr);

    std::vector<dmlc::any> padding_attr;
    padding_attr.emplace_back(padding);
    node_->SetAttr("padding", padding_attr);

    std::vector<dmlc::any> pooling_attr;
    pooling_attr.emplace_back(pool_size);
    node_->SetAttr("pool_size", pooling_attr);
  }

  NNAPIJSONSerializer* serializer_;
  JSONGraphObjectPtr node_;
};

class NNAPIJSONSerializer : public JSONSerializer {
 public:
  explicit NNAPIJSONSerializer(Map<Constant, String> constant_names, Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}
  using JSONSerializer::VisitExpr_;

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* call_node) final {
    const auto* fn_var = call_node->op.as<VarNode>();
    ICHECK(fn_var);
    const auto fn = Downcast<Function>(bindings_[GetRef<Var>(fn_var)]);
    ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr<String>(attr::kComposite);
    ICHECK(composite_opt.defined()) << "Only composite functions are supported.";

    std::string composite_name = composite_opt.value();

    CollectFromCompositeFunctionBody collector(this);
    collector.VisitExpr(fn->body);

    NodeEntries inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, /* name_ */
                                                "kernel",       /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    node->CaptureAttrs(*collector.node_);

    VLOG(1) << "Adding node " << composite_name << " with " << node->GetInputs().size()
            << " inputs";
    return AddNode(node, GetRef<Expr>(call_node));
  }

 private:
  Map<Var, Expr> bindings_;
};

void CollectFromCompositeFunctionBody::VisitExpr_(const CallNode* call_node) {
  const auto* op_node = call_node->op.as<OpNode>();
  ICHECK(op_node != nullptr);
  std::string name = op_node->name;
  if (name == "relax.permute_dims") {
    SetPermuteDimsAttribute(call_node);
  } else if (name == "relax.astype") {
    SetAstypeAttribute(call_node);
  } else if (name == "relax.mean") {
    SetMeanAttribute(call_node);
  } else if (name == "relax.nn.conv2d") {
    SetConv2dAttribute(call_node);
  } else if (name == "relax.nn.max_pool2d") {
    SetMaxPool2dAttribute(call_node);
  } else {
  }
  ExprVisitor::VisitExpr_(call_node);
}

Array<runtime::Module> NNAPICompiler(Array<Function> functions, Map<String, ObjectRef> /*unused*/,
                                     Map<Constant, String> constant_names) {
  VLOG(1) << "NNAPI Compiler";

  Array<runtime::Module> compiled_functions;
  for (const auto& func : functions) {
    NNAPIJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto constant_names = serializer.GetConstantNames();
    const auto* pf = runtime::Registry::Get("runtime.nnapi_runtime_create");
    ICHECK(pf != nullptr) << "Cannot find NNAPI runtime module create function.";
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back((*pf)(func_name, graph_json, constant_names));
  }

  return compiled_functions;
}

TVM_REGISTER_GLOBAL("relax.ext.nnapi").set_body_typed(NNAPICompiler);

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
