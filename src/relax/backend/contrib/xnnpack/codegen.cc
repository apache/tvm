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
#include <tvm/relax/attrs/qdq.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/struct_info.h>

#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
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

struct XNNPACKRuntimeOptions {
  bool use_weights_cache{false};
  bool use_workspace{false};
  bool profile{false};
  bool dont_spin_workers{false};
  bool transient_indirection_buffer{false};
  int64_t num_threads{1};
  std::string precision{"fp32"};

  std::string Serialize() const {
    std::ostringstream os;
    os << "use_weights_cache=" << (use_weights_cache ? 1 : 0) << ";";
    os << "use_workspace=" << (use_workspace ? 1 : 0) << ";";
    os << "profile=" << (profile ? 1 : 0) << ";";
    os << "dont_spin_workers=" << (dont_spin_workers ? 1 : 0) << ";";
    os << "transient_indirection_buffer=" << (transient_indirection_buffer ? 1 : 0) << ";";
    os << "num_threads=" << num_threads << ";";
    os << "precision=" << precision << ";";
    return os.str();
  }
};

bool GetBoolOption(const ffi::Map<ffi::String, ffi::Any>& options, const std::string& key,
                   bool default_value) {
  auto it = options.find(key);
  if (it == options.end()) return default_value;
  const ffi::Any& value = (*it).second;
  if (auto opt_bool = value.try_cast<bool>()) return opt_bool.value();
  if (auto opt_int = value.try_cast<int64_t>()) return opt_int.value() != 0;
  TVM_FFI_THROW(ValueError) << "XNNPACK RunCodegen option '" << key << "' must be a boolean value.";
}

int64_t GetIntOption(const ffi::Map<ffi::String, ffi::Any>& options, const std::string& key,
                     int64_t default_value) {
  auto it = options.find(key);
  if (it == options.end()) return default_value;
  const ffi::Any& value = (*it).second;
  if (auto opt_int = value.try_cast<int64_t>()) return opt_int.value();
  TVM_FFI_THROW(ValueError) << "XNNPACK RunCodegen option '" << key
                            << "' must be an integer value.";
}

ffi::Optional<ffi::String> GetStringOption(const ffi::Map<ffi::String, ffi::Any>& options,
                                           const std::string& key) {
  auto it = options.find(key);
  if (it == options.end()) return std::nullopt;
  const ffi::Any& value = (*it).second;
  if (auto opt_string = value.try_cast<ffi::String>()) return opt_string.value();
  TVM_FFI_THROW(ValueError) << "XNNPACK RunCodegen option '" << key << "' must be a string value.";
}

void ValidatePrecision(const std::string& precision) {
  static const std::unordered_set<std::string> supported = {"fp32", "fp16_hint", "fp16_force"};
  TVM_FFI_ICHECK(supported.count(precision)) << "Unsupported XNNPACK precision: " << precision;
}

XNNPACKRuntimeOptions ParseRuntimeOptions(const ffi::Map<ffi::String, ffi::Any>& options,
                                          const ffi::Optional<ffi::String>& annotated_precision) {
  static const std::unordered_set<std::string> supported = {
      "use_weights_cache",
      "use_workspace",
      "profile",
      "dont_spin_workers",
      "transient_indirection_buffer",
      "num_threads",
      "precision",
  };
  for (const auto& kv : options) {
    const std::string key = kv.first;
    TVM_FFI_ICHECK(supported.count(key)) << "Unsupported XNNPACK RunCodegen option: " << key;
  }

  XNNPACKRuntimeOptions parsed;
  parsed.use_weights_cache = GetBoolOption(options, "use_weights_cache", false);
  parsed.use_workspace = GetBoolOption(options, "use_workspace", false);
  parsed.profile = GetBoolOption(options, "profile", false);
  parsed.dont_spin_workers = GetBoolOption(options, "dont_spin_workers", false);
  parsed.transient_indirection_buffer =
      GetBoolOption(options, "transient_indirection_buffer", false);
  parsed.num_threads = GetIntOption(options, "num_threads", 1);
  if (annotated_precision.has_value()) {
    parsed.precision = annotated_precision.value();
  }
  if (auto option_precision = GetStringOption(options, "precision")) {
    ValidatePrecision(option_precision.value());
    if (annotated_precision.has_value()) {
      TVM_FFI_ICHECK_EQ(std::string(annotated_precision.value()),
                        std::string(option_precision.value()))
          << "XNNPACK precision from partition_for_xnnpack and RunCodegen options must match.";
    }
    parsed.precision = option_precision.value();
  }
  ValidatePrecision(parsed.precision);
  TVM_FFI_ICHECK_GE(parsed.num_threads, 1)
      << "XNNPACK RunCodegen option 'num_threads' must be >= 1.";
  return parsed;
}

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

    if (IsDynamicRangeComposite(composite_name)) {
      return VisitDynamicRangeComposite(call_node, fn, composite_name);
    }
    if (IsQuantizedComposite(composite_name)) {
      return VisitQuantizedComposite(call_node, fn, composite_name);
    }

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
        "xnnpack.dynamic_range_fully_connected_bias_clip",
        "xnnpack.dynamic_range_fully_connected_bias_relu",
        "xnnpack.dynamic_range_fully_connected_clip",
        "xnnpack.dynamic_range_fully_connected_relu",
        "xnnpack.dynamic_range_fully_connected_bias",
        "xnnpack.dynamic_range_fully_connected",
        "xnnpack.qs8_fully_connected_bias_clip",
        "xnnpack.qs8_fully_connected_bias_relu",
        "xnnpack.qs8_fully_connected_clip",
        "xnnpack.qs8_fully_connected_relu",
        "xnnpack.qs8_fully_connected_bias",
        "xnnpack.qs8_fully_connected",
        "xnnpack.qs8_conv2d_bias_clip",
        "xnnpack.qs8_conv2d_bias_relu",
        "xnnpack.qs8_conv2d_clip",
        "xnnpack.qs8_conv2d_relu",
        "xnnpack.qs8_conv2d_bias",
        "xnnpack.qs8_conv2d",
        "xnnpack.qs8_depthwise_conv2d_bias_clip",
        "xnnpack.qs8_depthwise_conv2d_bias_relu",
        "xnnpack.qs8_depthwise_conv2d_clip",
        "xnnpack.qs8_depthwise_conv2d_relu",
        "xnnpack.qs8_depthwise_conv2d_bias",
        "xnnpack.qs8_depthwise_conv2d",
        "xnnpack.qs8_reshape",
        "xnnpack.qs8_flatten",
        "xnnpack.qs8_copy",
        "xnnpack.qs8_max_pool2d",
        "xnnpack.qs8_avg_pool2d",
        "xnnpack.qs8_add_clip",
        "xnnpack.qs8_add_relu",
        "xnnpack.qs8_add",
    };
    return std::find(supported.begin(), supported.end(), name) != supported.end();
  }

  static bool IsQuantizedComposite(const std::string& name) {
    return name.find("xnnpack.qs8_") == 0;
  }

  static bool IsDynamicRangeComposite(const std::string& name) {
    return name.find("xnnpack.dynamic_range_") == 0;
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

  static const CallNode* AsCall(const Expr& expr, const char* name) {
    const auto* call = expr.as<CallNode>();
    TVM_FFI_ICHECK(call) << name << " must be a Relax call.";
    return call;
  }

  static Constant AsConstant(const Expr& expr, const char* name) {
    TVM_FFI_ICHECK(expr.as<ConstantNode>()) << name << " must be a Relax constant.";
    return Downcast<Constant>(expr);
  }

  static Expr ResolveExpr(const Expr& expr, const ffi::Map<Var, Expr>& local_bindings) {
    if (const auto* var = expr.as<VarNode>()) {
      Var ref = ffi::GetRef<Var>(var);
      auto it = local_bindings.find(ref);
      if (it != local_bindings.end()) return (*it).second;
    }
    return expr;
  }

  Expr ResolveCompositeArg(const Expr& expr, const Function& fn, const CallNode* call_node,
                           const ffi::Map<Var, Expr>& local_bindings) const {
    Expr resolved = ResolveExpr(expr, local_bindings);
    if (const auto* var = resolved.as<VarNode>()) {
      Var ref = ffi::GetRef<Var>(var);
      for (size_t i = 0; i < fn->params.size(); ++i) {
        if (fn->params[i].same_as(ref)) {
          TVM_FFI_ICHECK_LT(i, call_node->args.size());
          return ResolveExpr(call_node->args[i], bindings_);
        }
      }
    }
    return resolved;
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

  static ffi::Array<int64_t> StaticShape(const Expr& expr, const char* name) {
    const auto* expr_node = expr.as<ExprNode>();
    TVM_FFI_ICHECK(expr_node) << name << " must be a Relax expression.";
    auto sinfo = Downcast<TensorStructInfo>(expr_node->struct_info_);
    TVM_FFI_ICHECK(sinfo->shape.defined()) << name << " must have static shape.";
    auto shape = Downcast<ShapeExpr>(sinfo->shape.value());
    ffi::Array<int64_t> result;
    for (PrimExpr dim : shape->values) {
      const auto* int_dim = dim.as<IntImmNode>();
      TVM_FFI_ICHECK(int_dim) << name << " must have static integer shape.";
      TVM_FFI_ICHECK_GT(int_dim->value, 0) << name << " dimensions must be positive.";
      result.push_back(int_dim->value);
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

  static std::vector<double> ConstantFloatArray(const Expr& expr, const char* name) {
    const auto* constant = expr.as<ConstantNode>();
    TVM_FFI_ICHECK(constant) << name << " must be a constant.";
    auto sinfo = Downcast<TensorStructInfo>(constant->struct_info_);
    TVM_FFI_ICHECK(sinfo->dtype == DataType::Float(32)) << name << " must be float32.";
    size_t count = 1;
    if (sinfo->shape.defined()) {
      auto shape = Downcast<ShapeExpr>(sinfo->shape.value());
      count = 1;
      for (PrimExpr dim : shape->values) {
        const auto* int_dim = dim.as<IntImmNode>();
        TVM_FFI_ICHECK(int_dim) << name << " must have static shape.";
        count *= static_cast<size_t>(int_dim->value);
      }
    }
    const float* data = static_cast<const float*>(constant->data->data);
    std::vector<double> result;
    for (size_t i = 0; i < count; ++i) result.push_back(data[i]);
    return result;
  }

  static int64_t ConstantIntScalar(const Expr& expr, const char* name) {
    const auto* constant = expr.as<ConstantNode>();
    TVM_FFI_ICHECK(constant) << name << " must be a constant.";
    auto sinfo = Downcast<TensorStructInfo>(constant->struct_info_);
    size_t count = 1;
    if (sinfo->shape.defined()) {
      auto shape = Downcast<ShapeExpr>(sinfo->shape.value());
      for (PrimExpr dim : shape->values) {
        const auto* int_dim = dim.as<IntImmNode>();
        TVM_FFI_ICHECK(int_dim) << name << " must have static shape.";
        count *= static_cast<size_t>(int_dim->value);
      }
    }
    TVM_FFI_ICHECK_EQ(count, 1U) << name << " must be a scalar.";
    if (sinfo->dtype == DataType::Int(8)) {
      return static_cast<const int8_t*>(constant->data->data)[0];
    }
    if (sinfo->dtype == DataType::Int(32)) {
      return static_cast<const int32_t*>(constant->data->data)[0];
    }
    TVM_FFI_THROW(ValueError) << name << " must be int8 or int32.";
  }

  static ffi::String JoinFloats(const std::vector<double>& values) {
    std::ostringstream os;
    for (size_t i = 0; i < values.size(); ++i) {
      if (i != 0) os << ",";
      os << values[i];
    }
    return ffi::String(os.str());
  }

  static std::string QScheme(const std::vector<double>& scale) {
    return scale.size() == 1 ? "per_tensor" : "per_channel";
  }

  static void SetQParams(const JSONGraphObjectPtr& node, const std::string& prefix,
                         const CallNode* qdq_call, int64_t channel_dim) {
    const auto* attrs = qdq_call->attrs.as<QuantizeAttrs>();
    TVM_FFI_ICHECK(attrs) << "relax.quantize/dequantize is missing QuantizeAttrs.";
    const std::vector<double> scales = ConstantFloatArray(qdq_call->args[1], "qparam scale");
    node->SetAttr(prefix + "_qscheme", ffi::String(QScheme(scales)));
    node->SetAttr(prefix + "_scales", JoinFloats(scales));
    node->SetAttr(prefix + "_zero_point",
                  ConstantIntScalar(qdq_call->args[2], "qparam zero_point"));
    node->SetAttr(prefix + "_axis",
                  channel_dim >= 0 ? channel_dim : static_cast<int64_t>(attrs->axis));
    node->SetAttr(prefix + "_channel_dim", channel_dim);
  }

  static const CallNode* FindBiasDequantize(const std::vector<const CallNode*>& calls,
                                            const CallNode* weighted_call,
                                            const ffi::Map<Var, Expr>& local_bindings) {
    for (const CallNode* call : calls) {
      if (call->op.as<OpNode>() && OpName(call) == "relax.add") {
        Expr lhs_expr = ResolveExpr(call->args[0], local_bindings);
        Expr rhs_expr = ResolveExpr(call->args[1], local_bindings);
        const CallNode* lhs = lhs_expr.as<CallNode>();
        const CallNode* rhs = rhs_expr.as<CallNode>();
        if (lhs == weighted_call && rhs != nullptr && OpName(rhs) == "relax.dequantize") {
          return rhs;
        }
        if (rhs == weighted_call && lhs != nullptr && OpName(lhs) == "relax.dequantize") {
          return lhs;
        }
      }
    }
    return nullptr;
  }

  static void SetActivationAttrs(const JSONGraphObjectPtr& node, const std::string& activation,
                                 double min_value = -kXNNPACKInfinity,
                                 double max_value = kXNNPACKInfinity) {
    node->SetAttr("activation", ffi::String(activation));
    node->SetAttr("activation_min", min_value);
    node->SetAttr("activation_max", max_value);
  }

  NodeEntries VisitQuantizedComposite(const CallNode* call_node, const Function& fn,
                                      const std::string& composite_name) {
    if (composite_name == "xnnpack.qs8_reshape" ||
        composite_name == "xnnpack.qs8_flatten" ||
        composite_name == "xnnpack.qs8_copy" ||
        composite_name == "xnnpack.qs8_max_pool2d" ||
        composite_name == "xnnpack.qs8_avg_pool2d" ||
        composite_name.find("xnnpack.qs8_add") == 0) {
      return VisitQuantizedIslandComposite(call_node, fn, composite_name);
    }

    const auto calls = CollectCalls(fn);
    const auto local_bindings = AnalyzeVar2Value(fn);
    const CallNode* weighted_call = nullptr;
    if (composite_name.find("fully_connected") != std::string::npos) {
      weighted_call = FindCall(calls, "relax.matmul");
    } else {
      weighted_call = FindCall(calls, "relax.nn.conv2d");
    }
    TVM_FFI_ICHECK(weighted_call) << composite_name << " is missing its weighted op.";

    const CallNode* data_dq =
        AsCall(ResolveExpr(weighted_call->args[0], local_bindings), "quantized input dequantize");
    const CallNode* weight_dq =
        AsCall(ResolveExpr(weighted_call->args[1], local_bindings), "quantized weight dequantize");
    TVM_FFI_ICHECK_EQ(OpName(data_dq), "relax.dequantize");
    TVM_FFI_ICHECK_EQ(OpName(weight_dq), "relax.dequantize");
    const CallNode* bias_dq = FindBiasDequantize(calls, weighted_call, local_bindings);
    const bool has_bias = composite_name.find("_bias") != std::string::npos;
    TVM_FFI_ICHECK_EQ(has_bias, bias_dq != nullptr);

    NodeEntries inputs;
    TVM_FFI_ICHECK_GE(call_node->args.size(), 1U)
        << composite_name << " expects one external quantized input.";
    auto data_res = VisitExpr(call_node->args[0]);
    inputs.insert(inputs.end(), data_res.begin(), data_res.end());
    Expr weight_expr = ResolveExpr(weight_dq->args[0], local_bindings);
    if (!weight_expr.as<ConstantNode>() && call_node->args.size() > 1) {
      weight_expr = ResolveExpr(call_node->args[1], bindings_);
    }
    auto weight_res = weight_expr.as<ConstantNode>() ? VisitExpr(Downcast<Constant>(weight_expr))
                                                     : VisitExpr(weight_expr);
    inputs.insert(inputs.end(), weight_res.begin(), weight_res.end());
    if (has_bias) {
      Expr bias_expr = ResolveExpr(bias_dq->args[0], local_bindings);
      if (!bias_expr.as<ConstantNode>() && call_node->args.size() > 2) {
        bias_expr = ResolveExpr(call_node->args[2], bindings_);
      }
      auto bias_res = bias_expr.as<ConstantNode>() ? VisitExpr(Downcast<Constant>(bias_expr))
                                                   : VisitExpr(bias_expr);
      inputs.insert(inputs.end(), bias_res.begin(), bias_res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, "kernel", inputs, 1);
    SetQuantizedCompositeAttrs(node, fn, composite_name, inputs.size(), weighted_call, data_dq,
                               weight_dq, bias_dq);
    return AddNode(node, ffi::GetRef<Expr>(call_node));
  }

  NodeEntries VisitQuantizedIslandComposite(const CallNode* call_node, const Function& fn,
                                            const std::string& composite_name) {
    const auto calls = CollectCalls(fn);
    const auto local_bindings = AnalyzeVar2Value(fn);
    const CallNode* root = RootCall(calls);
    TVM_FFI_ICHECK_EQ(OpName(root), "relax.quantize");

    NodeEntries inputs;
    for (const auto& arg : call_node->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, "kernel", inputs, 1);
    SetQuantizedIslandAttrs(node, fn, composite_name, inputs.size(), root, local_bindings);
    return AddNode(node, ffi::GetRef<Expr>(call_node));
  }

  NodeEntries VisitDynamicRangeComposite(const CallNode* call_node, const Function& fn,
                                         const std::string& composite_name) {
    const auto calls = CollectCalls(fn);
    const auto local_bindings = AnalyzeVar2Value(fn);
    const CallNode* weighted_call = FindCall(calls, "relax.matmul");
    TVM_FFI_ICHECK(weighted_call)
        << composite_name << " must contain relax.matmul for dynamic-range fully_connected.";
    const CallNode* weight_dq =
        AsCall(ResolveExpr(weighted_call->args[1], local_bindings), "dynamic-range weight");
    TVM_FFI_ICHECK_EQ(OpName(weight_dq), "relax.dequantize");
    const bool has_bias = composite_name.find("_bias") != std::string::npos;

    NodeEntries inputs;
    TVM_FFI_ICHECK_GE(call_node->args.size(), 1U)
        << composite_name << " expects one external float32 input.";
    Expr data_expr = ResolveCompositeArg(weighted_call->args[0], fn, call_node, local_bindings);
    auto data_res = VisitExpr(data_expr);
    inputs.insert(inputs.end(), data_res.begin(), data_res.end());
    Expr weight_expr = ResolveCompositeArg(weight_dq->args[0], fn, call_node, local_bindings);
    auto weight_res = weight_expr.as<ConstantNode>() ? VisitExpr(Downcast<Constant>(weight_expr))
                                                     : VisitExpr(weight_expr);
    inputs.insert(inputs.end(), weight_res.begin(), weight_res.end());
    if (has_bias) {
      const CallNode* bias_add = FindCall(calls, "relax.add");
      TVM_FFI_ICHECK(bias_add) << composite_name << " must contain relax.add for bias.";
      Expr lhs = ResolveExpr(bias_add->args[0], local_bindings);
      Expr rhs = ResolveExpr(bias_add->args[1], local_bindings);
      Expr bias_expr = lhs.as<CallNode>() == weighted_call ? rhs : lhs;
      bias_expr = ResolveCompositeArg(bias_expr, fn, call_node, local_bindings);
      auto bias_res = bias_expr.as<ConstantNode>() ? VisitExpr(Downcast<Constant>(bias_expr))
                                                   : VisitExpr(bias_expr);
      inputs.insert(inputs.end(), bias_res.begin(), bias_res.end());
    }

    auto node = std::make_shared<JSONGraphNode>(composite_name, "kernel", inputs, 1);
    SetDynamicRangeCompositeAttrs(node, fn, composite_name, inputs.size(), weight_dq);
    return AddNode(node, ffi::GetRef<Expr>(call_node));
  }

  static void SetQuantizedActivationAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                                          const std::string& composite_name) {
    const auto calls = CollectCalls(fn);
    if (composite_name.find("_relu") != std::string::npos) {
      SetActivationAttrs(node, "clamp", 0.0, kXNNPACKInfinity);
    } else if (composite_name.find("_clip") != std::string::npos) {
      const CallNode* clip = FindCall(calls, "relax.clip");
      TVM_FFI_ICHECK(clip) << composite_name << " must contain relax.clip.";
      SetActivationAttrs(node, "clamp", PrimValueToDouble(clip->args[1]),
                         PrimValueToDouble(clip->args[2]));
    } else {
      SetActivationAttrs(node, "none");
    }
  }

  static void SetDynamicRangeCompositeAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                                            const std::string& composite_name, size_t num_inputs,
                                            const CallNode* weight_dq) {
    const bool has_bias = composite_name.find("_bias") != std::string::npos;
    TVM_FFI_ICHECK_EQ(num_inputs, has_bias ? 3U : 2U);
    node->SetAttr("quantized", static_cast<int64_t>(1));
    node->SetAttr("quantization", ffi::String("dynamic_range"));
    node->SetAttr("signedness", ffi::String("qd8_qc8w"));
    node->SetAttr("op_kind", ffi::String("dynamic_range_fully_connected"));
    node->SetAttr("has_bias", static_cast<int64_t>(has_bias));
    node->SetAttr("activation_dtype", ffi::String("float32"));
    node->SetAttr("output_dtype", ffi::String("float32"));
    SetQParams(node, "weight", weight_dq, 1);
    SetQuantizedActivationAttrs(node, fn, composite_name);
  }

  static void SetQuantizedCompositeAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                                         const std::string& composite_name, size_t num_inputs,
                                         const CallNode* weighted_call, const CallNode* data_dq,
                                         const CallNode* weight_dq, const CallNode* bias_dq) {
    const bool has_bias = composite_name.find("_bias") != std::string::npos;
    TVM_FFI_ICHECK_EQ(num_inputs, has_bias ? 3U : 2U);
    node->SetAttr("quantized", static_cast<int64_t>(1));
    node->SetAttr("signedness", ffi::String("qs8"));
    node->SetAttr("has_bias", static_cast<int64_t>(has_bias));
    SetQParams(node, "input", data_dq, -1);
    SetQParams(node, "output", RootCall(CollectCalls(fn)), -1);

    if (composite_name.find("fully_connected") != std::string::npos) {
      node->SetAttr("op_kind", ffi::String("qs8_fully_connected"));
      SetQParams(node, "weight", weight_dq, 1);
      if (has_bias) SetQParams(node, "bias", bias_dq, 0);
    } else if (composite_name.find("depthwise") != std::string::npos) {
      const auto* attrs = weighted_call->attrs.as<Conv2DAttrs>();
      TVM_FFI_ICHECK(attrs) << "relax.nn.conv2d is missing Conv2DAttrs.";
      node->SetAttr("op_kind", ffi::String("qs8_depthwise_conv2d"));
      node->SetAttr("strides", AsIntArray(attrs->strides));
      node->SetAttr("padding", NormalizePadding(attrs->padding));
      node->SetAttr("dilation", AsIntArray(attrs->dilation));
      node->SetAttr("groups", static_cast<int64_t>(attrs->groups));
      SetQParams(node, "weight", weight_dq, 3);
      if (has_bias) SetQParams(node, "bias", bias_dq, 0);
    } else {
      const auto* attrs = weighted_call->attrs.as<Conv2DAttrs>();
      TVM_FFI_ICHECK(attrs) << "relax.nn.conv2d is missing Conv2DAttrs.";
      node->SetAttr("op_kind", ffi::String("qs8_conv2d"));
      node->SetAttr("strides", AsIntArray(attrs->strides));
      node->SetAttr("padding", NormalizePadding(attrs->padding));
      node->SetAttr("dilation", AsIntArray(attrs->dilation));
      node->SetAttr("groups", static_cast<int64_t>(attrs->groups));
      SetQParams(node, "weight", weight_dq, 0);
      if (has_bias) SetQParams(node, "bias", bias_dq, 0);
    }
    SetQuantizedActivationAttrs(node, fn, composite_name);
  }

  static void SetQuantizedIslandAttrs(const JSONGraphObjectPtr& node, const Function& fn,
                                      const std::string& composite_name, size_t num_inputs,
                                      const CallNode* root,
                                      const ffi::Map<Var, Expr>& local_bindings) {
    node->SetAttr("quantized", static_cast<int64_t>(1));
    node->SetAttr("signedness", ffi::String("qs8"));
    SetQParams(node, "output", root, -1);

    if (composite_name == "xnnpack.qs8_reshape" ||
        composite_name == "xnnpack.qs8_flatten") {
      TVM_FFI_ICHECK_EQ(num_inputs, 1U) << composite_name << " expects one input.";
      const std::string op_name =
          composite_name == "xnnpack.qs8_reshape" ? "relax.reshape" : "relax.flatten";
      const CallNode* op_call = FindCall(CollectCalls(fn), op_name);
      TVM_FFI_ICHECK(op_call) << composite_name << " must contain " << op_name << ".";
      const CallNode* data_dq =
          AsCall(ResolveExpr(op_call->args[0], local_bindings), "quantized reshape input");
      TVM_FFI_ICHECK_EQ(OpName(data_dq), "relax.dequantize");
      node->SetAttr("op_kind", ffi::String("qs8_reshape"));
      node->SetAttr("new_shape", StaticShape(ffi::GetRef<Expr>(root), "quantized reshape output"));
      SetQParams(node, "input", data_dq, -1);
      SetActivationAttrs(node, "none");
      return;
    }

    if (composite_name == "xnnpack.qs8_copy") {
      TVM_FFI_ICHECK_EQ(num_inputs, 1U) << composite_name << " expects one input.";
      const CallNode* data_dq =
          AsCall(ResolveExpr(root->args[0], local_bindings), "quantized copy input");
      TVM_FFI_ICHECK_EQ(OpName(data_dq), "relax.dequantize");
      node->SetAttr("op_kind", ffi::String("qs8_copy"));
      SetQParams(node, "input", data_dq, -1);
      SetActivationAttrs(node, "none");
      return;
    }

    if (composite_name == "xnnpack.qs8_max_pool2d" ||
        composite_name == "xnnpack.qs8_avg_pool2d") {
      TVM_FFI_ICHECK_EQ(num_inputs, 1U) << composite_name << " expects one input.";
      const std::string op_name = composite_name == "xnnpack.qs8_max_pool2d"
                                      ? "relax.nn.max_pool2d"
                                      : "relax.nn.avg_pool2d";
      const auto calls = CollectCalls(fn);
      const CallNode* pool_call = FindCall(calls, op_name);
      TVM_FFI_ICHECK(pool_call) << composite_name << " must contain " << op_name << ".";
      const CallNode* data_dq =
          AsCall(ResolveExpr(pool_call->args[0], local_bindings), "quantized pool input");
      TVM_FFI_ICHECK_EQ(OpName(data_dq), "relax.dequantize");
      SetQParams(node, "input", data_dq, -1);
      SetPool2DAttrs(node, fn,
                     composite_name == "xnnpack.qs8_max_pool2d" ? "xnnpack.max_pool2d"
                                                                 : "xnnpack.avg_pool2d",
                     num_inputs);
      node->SetAttr("op_kind", ffi::String(composite_name == "xnnpack.qs8_max_pool2d"
                                               ? "qs8_max_pool2d"
                                               : "qs8_avg_pool2d"));
      return;
    }

    TVM_FFI_ICHECK(composite_name.find("xnnpack.qs8_add") == 0)
        << "Unsupported quantized island composite: " << composite_name;
    TVM_FFI_ICHECK_EQ(num_inputs, 2U) << composite_name << " expects two inputs.";
    const auto calls = CollectCalls(fn);
    const CallNode* add_call = FindCall(calls, "relax.add");
    TVM_FFI_ICHECK(add_call) << composite_name << " must contain relax.add.";
    const CallNode* lhs_dq =
        AsCall(ResolveExpr(add_call->args[0], local_bindings), "quantized add lhs");
    const CallNode* rhs_dq =
        AsCall(ResolveExpr(add_call->args[1], local_bindings), "quantized add rhs");
    TVM_FFI_ICHECK_EQ(OpName(lhs_dq), "relax.dequantize");
    TVM_FFI_ICHECK_EQ(OpName(rhs_dq), "relax.dequantize");
    node->SetAttr("op_kind", ffi::String("qs8_add"));
    SetQParams(node, "lhs", lhs_dq, -1);
    SetQParams(node, "rhs", rhs_dq, -1);
    SetQuantizedActivationAttrs(node, fn, composite_name);
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
                                        ffi::Map<ffi::String, ffi::Any> options,
                                        ffi::Map<Constant, ffi::String> constant_names) {
  ffi::Array<ffi::Module> compiled_functions;
  const auto pf = tvm::ffi::Function::GetGlobalRequired("runtime.XNNPACKJSONRuntimeCreate");

  for (const auto& func : functions) {
    const std::string runtime_options =
        ParseRuntimeOptions(options, func->GetAttr<ffi::String>("xnnpack_precision")).Serialize();
    XNNPACKJSONSerializer serializer(constant_names, AnalyzeVar2Value(func));
    serializer.serialize(func);
    auto graph_json = serializer.GetJSON();
    auto const_names = serializer.GetConstantNames();
    auto func_name = GetExtSymbol(func);
    compiled_functions.push_back(
        pf(func_name, graph_json, const_names, runtime_options).cast<ffi::Module>());
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
