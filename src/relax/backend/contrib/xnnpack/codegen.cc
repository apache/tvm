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
 * \file src/relax/backend/contrib/xnnpack/codegen.cc
 * \brief Minimal XNNPACK Relax external codegen.
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>

#include <limits>
#include <string>
#include <vector>

#include "../codegen_json/codegen_json.h"
#include "../utils.h"

namespace tvm {
namespace relax {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphObjectPtr = backend::contrib::JSONGraphObjectPtr;
using JSONSerializer = backend::contrib::JSONSerializer;
using backend::contrib::NodeEntries;

class XNNPACKJSONSerializer : public JSONSerializer {
 public:
  XNNPACKJSONSerializer(ffi::Map<Constant, ffi::String> constant_names,
                        ffi::Map<Var, Expr> bindings)
      : JSONSerializer(constant_names), bindings_(bindings) {}

  using JSONSerializer::VisitExpr_;

  NodeEntries VisitExpr_(const CallNode* call_node) final {
    const auto* fn_var = call_node->op.as<VarNode>();
    TVM_FFI_ICHECK(fn_var) << "XNNPACK codegen expects calls to composite functions.";

    const auto fn = Downcast<Function>(bindings_[ffi::GetRef<Var>(fn_var)]);
    TVM_FFI_ICHECK(fn.defined()) << "Expects the callee to be a function.";

    auto composite_opt = fn->GetAttr<ffi::String>(attr::kComposite);
    TVM_FFI_ICHECK(composite_opt.has_value()) << "Only composite functions are supported.";

    std::string composite_name = composite_opt.value();
    TVM_FFI_ICHECK(IsSupportedComposite(composite_name))
        << "Unsupported XNNPACK composite pattern: " << composite_name;

    NodeEntries inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    for (const auto& constant : CollectConstants(fn)) {
      auto res = VisitExpr(constant);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, "kernel", inputs, 1);
    SetCompositeAttrs(node, fn, composite_name, inputs.size());
    return AddNode(node, ffi::GetRef<Expr>(call_node));
  }

 private:
  static constexpr double kXNNPACKInfinity = 3.4028234663852886e38;

  static bool IsSupportedComposite(const std::string& name) {
    static const std::vector<std::string> supported = {
        "xnnpack.conv2d_bias_clip",
        "xnnpack.conv2d_bias_relu",
        "xnnpack.conv2d_clip",
        "xnnpack.conv2d_relu",
        "xnnpack.conv2d_bias",
        "xnnpack.conv2d",
        "xnnpack.max_pool2d",
        "xnnpack.avg_pool2d",
        "xnnpack.add",
        "xnnpack.clip",
        "xnnpack.relu",
        "xnnpack.sigmoid",
        "xnnpack.tanh",
    };
    return std::find(supported.begin(), supported.end(), name) != supported.end();
  }

  static std::string OpName(const CallNode* call) {
    const auto* op_node = call->op.as<OpNode>();
    TVM_FFI_ICHECK(op_node) << "XNNPACK composite functions must contain Relax op calls.";
    return op_node->name;
  }

  static std::vector<const CallNode*> CollectCalls(const Function& fn) {
    std::vector<const CallNode*> calls;
    PostOrderVisit(fn->body, [&calls](const Expr& expr) {
      if (const auto* call = expr.as<CallNode>()) {
        calls.push_back(call);
      }
    });
    return calls;
  }

  static std::vector<Constant> CollectConstants(const Function& fn) {
    std::vector<Constant> constants;
    PostOrderVisit(fn->body, [&constants](const Expr& expr) {
      if (expr.as<ConstantNode>()) {
        constants.push_back(Downcast<Constant>(expr));
      }
    });
    return constants;
  }

  static const CallNode* FindCall(const std::vector<const CallNode*>& calls,
                                  const std::string& op_name) {
    for (const CallNode* call : calls) {
      if (call->op.as<OpNode>() && OpName(call) == op_name) {
        return call;
      }
    }
    return nullptr;
  }

  static const CallNode* RootCall(const std::vector<const CallNode*>& calls) {
    TVM_FFI_ICHECK(!calls.empty()) << "XNNPACK composite function must contain at least one call.";
    return calls.back();
  }

  static double PrimValueToDouble(const Expr& expr) {
    const auto* prim = expr.as<PrimValueNode>();
    TVM_FFI_ICHECK(prim) << "Expected Relax PrimValue.";
    if (const auto* value = prim->value.as<FloatImmNode>()) {
      return value->value;
    }
    if (const auto* value = prim->value.as<IntImmNode>()) {
      return static_cast<double>(value->value);
    }
    TVM_FFI_THROW(InternalError) << "Unsupported PrimValue in XNNPACK composite.";
  }

  static ffi::Array<int64_t> AsIntArray(const ffi::Array<int64_t>& input) {
    ffi::Array<int64_t> result;
    for (int64_t value : input) {
      result.push_back(value);
    }
    return result;
  }

  static ffi::Array<int64_t> NormalizePadding(const ffi::Array<int64_t>& padding) {
    ffi::Array<int64_t> result;
    if (padding.size() == 1) {
      result.push_back(padding[0]);
      result.push_back(padding[0]);
      result.push_back(padding[0]);
      result.push_back(padding[0]);
    } else if (padding.size() == 2) {
      result.push_back(padding[0]);
      result.push_back(padding[1]);
      result.push_back(padding[0]);
      result.push_back(padding[1]);
    } else {
      TVM_FFI_ICHECK_EQ(padding.size(), 4U);
      result = AsIntArray(padding);
    }
    return result;
  }

  static void SetActivationAttrs(const JSONGraphObjectPtr& node, const std::string& activation,
                                 double min_value = -kXNNPACKInfinity,
                                 double max_value = kXNNPACKInfinity) {
    node->SetAttr("activation", ffi::String(activation));
    node->SetAttr("activation_min", min_value);
    node->SetAttr("activation_max", max_value);
  }

  static void SetConv2DAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                             const std::string& composite_name, size_t num_inputs) {
    const auto calls = CollectCalls(fn);
    const CallNode* conv_call = FindCall(calls, "relax.nn.conv2d");
    TVM_FFI_ICHECK(conv_call) << composite_name << " must contain relax.nn.conv2d.";
    const auto* attrs = conv_call->attrs.as<Conv2DAttrs>();
    TVM_FFI_ICHECK(attrs) << "relax.nn.conv2d is missing Conv2DAttrs.";

    const bool has_bias = composite_name.find("_bias") != std::string::npos;
    TVM_FFI_ICHECK_EQ(num_inputs, has_bias ? 3U : 2U)
        << composite_name << " expects data, weight, and optional bias inputs.";

    node->SetAttr("op_kind", ffi::String("conv2d"));
    node->SetAttr("strides", AsIntArray(attrs->strides));
    node->SetAttr("padding", NormalizePadding(attrs->padding));
    node->SetAttr("dilation", AsIntArray(attrs->dilation));
    node->SetAttr("groups", static_cast<int64_t>(attrs->groups));
    node->SetAttr("has_bias", static_cast<int64_t>(has_bias));

    if (composite_name.find("_relu") != std::string::npos) {
      SetActivationAttrs(node, "clamp", 0.0, kXNNPACKInfinity);
    } else if (composite_name.find("_clip") != std::string::npos) {
      const CallNode* root = RootCall(calls);
      TVM_FFI_ICHECK_EQ(OpName(root), "relax.clip");
      SetActivationAttrs(node, "clamp", PrimValueToDouble(root->args[1]),
                         PrimValueToDouble(root->args[2]));
    } else {
      SetActivationAttrs(node, "none");
    }
  }

  static void SetPool2DAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                             const std::string& composite_name, size_t num_inputs) {
    TVM_FFI_ICHECK_EQ(num_inputs, 1U) << composite_name << " expects one input.";
    const auto calls = CollectCalls(fn);
    const std::string op_name =
        composite_name == "xnnpack.max_pool2d" ? "relax.nn.max_pool2d" : "relax.nn.avg_pool2d";
    const CallNode* pool_call = FindCall(calls, op_name);
    TVM_FFI_ICHECK(pool_call) << composite_name << " must contain " << op_name << ".";
    const auto* attrs = pool_call->attrs.as<Pool2DAttrs>();
    TVM_FFI_ICHECK(attrs) << op_name << " is missing Pool2DAttrs.";

    node->SetAttr("op_kind", ffi::String(composite_name == "xnnpack.max_pool2d" ? "max_pool2d"
                                                                                : "avg_pool2d"));
    node->SetAttr("pool_size", AsIntArray(attrs->pool_size));
    node->SetAttr("strides", AsIntArray(attrs->strides));
    node->SetAttr("padding", NormalizePadding(attrs->padding));
    node->SetAttr("dilation", AsIntArray(attrs->dilation));
    SetActivationAttrs(node, "none");
  }

  static void SetUnaryAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                            const std::string& composite_name, size_t num_inputs) {
    TVM_FFI_ICHECK_EQ(num_inputs, 1U) << composite_name << " expects one input.";
    node->SetAttr("op_kind", ffi::String("unary"));
    if (composite_name == "xnnpack.relu") {
      node->SetAttr("unary_op", ffi::String("clamp"));
      SetActivationAttrs(node, "clamp", 0.0, kXNNPACKInfinity);
    } else if (composite_name == "xnnpack.clip") {
      const auto calls = CollectCalls(fn);
      const CallNode* root = RootCall(calls);
      TVM_FFI_ICHECK_EQ(OpName(root), "relax.clip");
      node->SetAttr("unary_op", ffi::String("clamp"));
      SetActivationAttrs(node, "clamp", PrimValueToDouble(root->args[1]),
                         PrimValueToDouble(root->args[2]));
    } else if (composite_name == "xnnpack.sigmoid") {
      node->SetAttr("unary_op", ffi::String("sigmoid"));
      SetActivationAttrs(node, "none");
    } else {
      TVM_FFI_ICHECK_EQ(composite_name, "xnnpack.tanh");
      node->SetAttr("unary_op", ffi::String("tanh"));
      SetActivationAttrs(node, "none");
    }
  }

  static void SetCompositeAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                                const std::string& composite_name, size_t num_inputs) {
    if (composite_name.find("xnnpack.conv2d") == 0) {
      SetConv2DAttrs(node, fn, composite_name, num_inputs);
    } else if (composite_name == "xnnpack.max_pool2d" || composite_name == "xnnpack.avg_pool2d") {
      SetPool2DAttrs(node, fn, composite_name, num_inputs);
    } else if (composite_name == "xnnpack.add") {
      TVM_FFI_ICHECK_EQ(num_inputs, 2U) << "xnnpack.add expects two inputs.";
      node->SetAttr("op_kind", ffi::String("add"));
      SetActivationAttrs(node, "none");
    } else {
      SetUnaryAttrs(node, fn, composite_name, num_inputs);
    }
  }

  ffi::Map<Var, Expr> bindings_;
};

ffi::Array<ffi::Module> XNNPACKCompiler(ffi::Array<Function> functions,
                                        ffi::Map<ffi::String, ffi::Any> /*options*/,
                                        ffi::Map<Constant, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;
  const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.XNNPACKJSONRuntimeCreate");

  for (const auto& func : functions) {
    XNNPACKJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto const_names = serializer.GetConstantNames();
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back(pf(func_name, graph_json, const_names).cast<ffi::Module>());
  }

  return compiled_functions;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.xnnpack", XNNPACKCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
