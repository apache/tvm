/*!
 * \file src/relay/backend/contrib/tensorrt/codegen.cc
 * \brief Implementation of the TensorRT JSON serializer.
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

class NNAPIJSONSerializer;

class CollectFromCompositeFunctionBody : public ExprVisitor {
 public:
  explicit CollectFromCompositeFunctionBody(NNAPIJSONSerializer* serializer)
      : serializer_(serializer), node_(std::make_shared<JSONGraphNode>()) {}

  void VisitExpr_(const CallNode* call_node) final;

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
    const auto* max_pool_2d_attr = call_node->attrs.as<MaxPool2DAttrs>();
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

    // 改
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

class NNAPIJSONSerializer : public backend::contrib::JSONSerializer {
 public:
  NNAPIJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* call_node) override {
    std::string name;
    tvm::Array<Expr> args;

    const auto* fn = call_node->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "NNAPI JSON runtime only supports composite functions.";
    name = comp.value();
    args = call_node->args;

    CollectFromCompositeFunctionBody collector(this);
    collector.VisitExpr(fn->body);

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);

    node->CaptureAttrs(*collector.node_);

    return AddNode(node, GetRef<Expr>(call_node));
  }
};

void CollectFromCompositeFunctionBody::VisitExpr_(const CallNode* call_node) {
  const auto* op_node = call_node->op.as<OpNode>();
  ICHECK(op_node != nullptr);
  std::string name = op_node->name;
  if (name == "nn.conv2d") {
    SetConv2dAttribute(call_node);
  } else if (name == "nn.max_pool2d") {
    SetMaxPool2dAttribute(call_node);
  }
  ExprVisitor::VisitExpr_(call_node);
}

runtime::Module NNAPICompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  NNAPIJSONSerializer serializer(func_name, func);

  serializer.serialize();

  std::string graph_json = serializer.GetJSON();
  LOG(INFO) << graph_json;
  const auto* pf = runtime::Registry::Get("runtime.nnapi_runtime_create");
  ICHECK(pf != nullptr) << "Cannot find NNAPI runtime module create function.";
  auto mod = (*pf)(func_name, graph_json, serializer.const_names());
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.nnapi").set_body_typed(NNAPICompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
