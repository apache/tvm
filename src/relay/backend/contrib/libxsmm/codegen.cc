#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include "../../utils.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class LibxsmmJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  LibxsmmJSONSerializer(const std::string& symbol, const Expr& expr)
      : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* call_node) override {
    std::string name;
    const CallNode* call = call_node;
    if (const auto* op_node = call_node->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* function_node = call_node->op.as<FunctionNode>()) {
      auto comp = function_node->GetAttr<String>(attr::kComposite);
      name = comp.value();

      auto body = function_node->body.as<CallNode>();
      if (name == "libxsmm.dense_bias") {
        auto add_op_type = IsOp(body->args[0].as<CallNode>(), "add") ? "add" : "nn.bias_add";
        call = GetRootCall(body, 1, {"nn.dense", add_op_type});
      } else if (name == "libxsmm.dense_relu") {
        call = GetRootCall(body, 1, {"nn.dense", "nn.relu"});
      } else if (name == "libxsmm.dense_bias_relu") {
        auto add_op_type = IsOp(body->args[0].as<CallNode>(), "add") ? "add" : "nn.bias_add";
        call = GetRootCall(body, 2, {"nn.dense", add_op_type, "nn.relu"});
      } else if (name == "libxsmm.dense") {
        call = GetRootCall(body, 0, {"nn.dense"});
      } else {
        LOG(FATAL) << "Unrecognized LIBXSMM pattern: " << name;
      }
    } else {
      LOG(FATAL) << "LIBXSMM JSON runtime does not support call to " << call_node->op->GetTypeKey();
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);

    SetCallNodeAttribute(node, call);
    return AddNode(node, GetRef<Expr>(call_node));
  }
};

/*!
 * \brief The extrenal compiler/codegen tool. It takes a Relay expression/module and compile it into
 * a runtime module.
 */
runtime::Module LibxsmmCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = backend::GetExtSymbol(func);

  LibxsmmJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();
  const auto* pf = runtime::Registry::Get("runtime.LibxsmmJSONRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create.";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.libxsmm").set_body_typed(LibxsmmCompiler);
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
