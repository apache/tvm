#include <tvm/runtime/device_api.h>
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
      : JSONSerializer(symbol, expr), symbol_(symbol) {}

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
        std::vector<std::string> expected_op_names = {"nn.dense", add_op_type};
        call = GetRootCall(body, 1, expected_op_names);
      } else if (name == "libxsmm.dense_relu") {
        std::vector<std::string> expected_op_names = {"nn.dense", "nn.relu"};
        call = GetRootCall(body, 1, expected_op_names);
      } else if (name == "libxsmm.dense_bias_relu") {
        auto add_op_type = IsOp(body->args[0].as<CallNode>(), "add") ? "add" : "nn.bias_add";
        std::vector<std::string> expected_op_names = {"nn.dense", add_op_type, "nn.relu"};
        call = GetRootCall(body, 2, expected_op_names);
      } else if (name == "libxsmm.dense") {
        std::vector<std::string> expected_op_names = {"nn.dense"};
        call = GetRootCall(body, 0, expected_op_names);
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

    // Add the constant weight also as input node
    ICHECK(call->args.size() == 2);
    auto weights = VisitExpr(call->args[1]);
    inputs.insert(inputs.end(), weights.begin(), weights.end());

    auto node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);

    SetCallNodeAttribute(node, call);
    return AddNode(node, GetRef<Expr>(call_node));
  }

  /*!
   * \brief Visit call nodes and generate ordered params.
   *
   * \param cn The current constant node.
   * \return A list of graph entry nodes.
   */
  std::vector<JSONGraphNodeEntry> VisitExpr_(const ConstantNode* cn) override {
    std::string name = "libxsmm_" + symbol_ + "_const_" + std::to_string(params_.size());
    params_.push_back(name);
    auto node = std::make_shared<JSONGraphNode>(name, "const" /* op_type_ */);
    return AddNode(node, GetRef<Expr>(cn));
  }

  Array<String> GetParams() const { return params_; }

 private:
  std::string symbol_;
  Array<String> params_;
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

struct LibxsmmConstantUpdater : public ExprVisitor {
 public:
  LibxsmmConstantUpdater(const std::string& symbol,
                         std::unordered_map<std::string, runtime::NDArray>* params)
      : symbol_(symbol), params_(params) {}

  void VisitExpr_(const FunctionNode* op) final {
    // Set true if current op is a libxsmm composite function.
    is_pattern_match_ = match_pattern(op);

    ExprVisitor::VisitExpr_(op);

    // Reset.
    is_pattern_match_ = false;
  }

  void VisitExpr_(const CallNode* op) final {
    this->VisitSpan(op->span);
    this->VisitExpr(op->op);

    for (auto ty_arg : op->type_args) {
      this->VisitType(ty_arg);
    }

    for (auto arg : op->args) {
      // Transpose weight if current call is nested in a libxsmm composite function.
      // Note: libxsmm doesn't support FC, so we have to transpose the weight here to leverage its
      // gemm primitive.
      if (is_pattern_match_) {
        auto const_node = arg.as<ConstantNode>();
        if (const_node) {
          transpose_weight_tensor(const_node);
        }
      }

      this->VisitExpr(arg);
    }
  }

  /*!
   * \brief Visit call nodes and generate ordered params.
   *
   * \param cn The current constant node.
   * \return A list of graph entry nodes.
   */
  void VisitExpr_(const ConstantNode* cn) override {
    std::string name = "libxsmm_" + symbol_ + "_const_" + std::to_string(const_idx_++);
    (*params_)[name] = cn->data;
  }

 private:
  void transpose_weight_tensor(const ConstantNode* const_node) {
    // Get weight_tensor from const_node.
    const DLTensor* weight_tensor = const_node->data.operator->();
    runtime::NDArray ndarray = const_node->data;

    int ndim = weight_tensor->ndim;
    ICHECK(ndim == 2);

    int64_t* shape = weight_tensor->shape;

    // Weight tensor has a shape of (K, N).
    int64_t K = shape[0];
    int64_t N = shape[1];

    // Create a new tensor which has the same shape with weight_tensor.
    DLTensor transposed_tensor;
    transposed_tensor.data =
        runtime::DeviceAPI::Get(ndarray->device)
            ->AllocDataSpace(ndarray->device, ndim, weight_tensor->shape, ndarray->dtype);
    transposed_tensor.ndim = weight_tensor->ndim;
    transposed_tensor.dtype = weight_tensor->dtype;
    transposed_tensor.strides = weight_tensor->strides;
    transposed_tensor.shape = weight_tensor->shape;
    transposed_tensor.byte_offset = weight_tensor->byte_offset;
    transposed_tensor.device = weight_tensor->device;

    // Init this new tensor with the value of transposed weight_tensor.
    float* data = static_cast<float*>(transposed_tensor.data);
    for (int64_t k = 0; k < shape[0]; k++) {
      for (int64_t n = 0; n < shape[1]; n++) {
        data[n * K + k] = static_cast<float*>(weight_tensor->data)[k * N + n];
      }
    }

    // Copy transposed_tensor's value back to weight_tensor.
    ndarray.CopyFrom(&transposed_tensor);
  }

  // Return true if current node is a libxsmm composite function.
  bool match_pattern(const FunctionNode* op) {
    auto comp = op->GetAttr<String>(attr::kComposite);

    if (comp.defined()) {
      const std::string name = comp.value();
      for (auto pattern : patterns_) {
        if (name == pattern) {
          return true;
        }
      }
    }

    return false;
  }

  // Mark if current in a libxsmm composite operator call.
  bool is_pattern_match_{false};

  // Composite operator patterns libxsmm supports.
  std::vector<std::string> patterns_{"libxsmm.dense", "libxsmm.dense_relu", "libxsmm.dense_bias",
                                     "libxsmm.dense_bias_relu"};
  int const_idx_{0};
  std::string symbol_;
  std::unordered_map<std::string, runtime::NDArray>* params_;
};

Map<String, runtime::NDArray> LibxsmmConstantUpdaterFunc(Expr expr, std::string symbol) {
  std::unordered_map<std::string, runtime::NDArray> res;
  LibxsmmConstantUpdater const_updater(symbol, &res);
  const_updater(expr);

  Map<String, runtime::NDArray> ret;
  for (const auto& kvp : res) ret.Set(kvp.first, kvp.second);

  return ret;
}

TVM_REGISTER_GLOBAL("relay.ext.libxsmm").set_body_typed(LibxsmmCompiler);
TVM_REGISTER_GLOBAL("relay.ext.libxsmm.constant_updater")
    .set_body_typed(LibxsmmConstantUpdaterFunc);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
