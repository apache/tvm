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
 *
 * \file src/relay/op/annotation/annotation.cc
 * \brief Helpers for working with various 'annotations' attributes.
 */

#include "./annotation.h"

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/expr.h>
#include <tvm/topi/elemwise.h>

#include "../../transforms/infer_layout_utils.h"
#include "../type_relations.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(OnDeviceAttrs);

const Op& OnDeviceOp() {
  static const Op& op = Op::Get("on_device");
  return op;
}

Expr OnDevice(Expr expr, DLDeviceType device_type, bool is_fixed) {
  auto attrs = make_object<OnDeviceAttrs>();
  attrs->device_type = device_type;
  attrs->is_fixed = is_fixed;
  Span span = expr->span;
  return Call(OnDeviceOp(), {std::move(expr)}, Attrs(std::move(attrs)), /*type_args=*/{}, span);
}

Expr MaybeOnDevice(Expr expr, DLDeviceType device_type, bool is_fixed) {
  if (device_type == kInvalidDeviceType) {
    // Undefined signals no annotation is required.
    return expr;
  }
  if (expr->IsInstance<OpNode>() || expr->IsInstance<ConstructorNode>()) {
    // These operators are device polymorphic so no annotation is required.
    // TODO(mbs): The device planning pass does NOT currently support device polymorphism for
    // constructors, so we could remove them from this condition. However most constructors
    // accept type parameters, and it is not well-formed Relay to simply wrap such a
    // constructor in an "on_device" call. So we'll pretend they are device polymorphic to
    // avoid that difficultly. Overall ADTs need more work to be fully supported.
    return expr;
  }
  if (expr->IsInstance<GlobalVarNode>() || expr->IsInstance<VarNode>()) {
    // The device can be recovered from the binding site of the global or local variable.
    return expr;
  }
  if (const auto* function_node = expr.as<FunctionNode>()) {
    if (function_node->HasNonzeroAttr(attr::kPrimitive)) {
      // Primitive functions are device polymorphic, matching our interpretation for OpNode above.
      return expr;
    }
  }
  return OnDevice(expr, device_type, is_fixed);
}

TVM_REGISTER_GLOBAL("relay.op.annotation._make.on_device")
    .set_body_typed([](Expr expr, int device_type, bool is_fixed) {
      return OnDevice(expr, static_cast<DLDeviceType>(device_type), is_fixed);
    });

RELAY_REGISTER_OP("on_device")
    .describe(R"code(Annotate an expression with device type)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .set_support_level(10)
    .add_type_rel("Identity", IdentityRel)
    .set_attrs_type_key("relay.attrs.OnDeviceAttrs")
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<TNonComputational>("TNonComputational", true)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_type) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

OnDeviceProps GetOnDeviceProps(const CallNode* call_node) {
  if (call_node->op == OnDeviceOp()) {
    ICHECK_EQ(call_node->args.size(), 1) << "on_device expects one argument";
    ICHECK(call_node->attrs.defined()) << "on_device requires attributes";
    const auto* on_device_attrs = call_node->attrs.as<OnDeviceAttrs>();
    ICHECK(on_device_attrs != nullptr) << "on_device requires OnDeviceAttrs";
    auto device_type = static_cast<DLDeviceType>(on_device_attrs->device_type);
    // Follow nesting:
    //   on_device(on_device(expr, device_type=1), device_type=2) == {expr, 1}
    auto inner = GetOnDeviceProps(call_node->args[0]);
    if (inner.body.defined()) {
      return {inner.body, inner.device_type, on_device_attrs->is_fixed || inner.is_fixed};
    } else {
      return {call_node->args[0], device_type, on_device_attrs->is_fixed};
    }
  }
  return {};
}

OnDeviceProps GetOnDeviceProps(const Expr& expr) {
  if (const auto* call_node = expr.as<CallNode>()) {
    return GetOnDeviceProps(call_node);
  }
  return {};
}

Function FunctionOnDevice(Function function, Array<Integer> param_device_types,
                          Integer result_device_type) {
  return WithAttrs(std::move(function), {{tvm::attr::kParamDeviceTypes, param_device_types},
                                         {tvm::attr::kResultDeviceType, result_device_type}});
}

Function FunctionOnDevice(Function function, const std::vector<DLDeviceType>& param_device_types,
                          DLDeviceType result_device_type) {
  Array<Integer> arr;
  arr.reserve(param_device_types.size());
  for (const auto device_type : param_device_types) {
    arr.push_back(static_cast<int>(device_type));
  }
  return FunctionOnDevice(std::move(function), std::move(arr),
                          static_cast<int>(result_device_type));
}

Function OptFunctionOnDevice(Function function, const std::vector<DLDeviceType>& param_device_types,
                             DLDeviceType result_device_type) {
  if (std::all_of(param_device_types.begin(), param_device_types.end(),
                  [](DLDeviceType type) { return type == kInvalidDeviceType; }) &&
      result_device_type == kInvalidDeviceType) {
    return function;
  }
  return FunctionOnDevice(function, param_device_types, result_device_type);
}

TVM_REGISTER_GLOBAL("relay.op.annotation._make.function_on_device")
    .set_body_typed([](Function function, Array<Integer> param_device_types,
                       int result_device_type) {
      return FunctionOnDevice(function, param_device_types,
                              static_cast<DLDeviceType>(result_device_type));
    });

DLDeviceType GetFunctionResultDeviceType(const FunctionNode* function_node) {
  auto opt_integer = function_node->GetAttr<Integer>(tvm::attr::kResultDeviceType);
  if (!opt_integer) {
    // No annotation.
    return kInvalidDeviceType;
  }
  return static_cast<DLDeviceType>(opt_integer.value()->value);
}

DLDeviceType GetFunctionParamDeviceType(const FunctionNode* function_node, size_t i) {
  ICHECK_LT(i, function_node->params.size())
      << "param index " << i << " out of range for function of arity "
      << function_node->params.size();
  auto opt_array = function_node->GetAttr<Array<Integer>>(tvm::attr::kParamDeviceTypes);
  if (!opt_array) {
    // No annotation.
    return kInvalidDeviceType;
  }
  ICHECK_EQ(opt_array.value().size(), function_node->params.size())
      << "annotation parameters do not match function arity";
  return static_cast<DLDeviceType>(opt_array.value()[i]->value);
}

Expr StopFusion(Expr data) {
  static const Op& op = Op::Get("annotation.stop_fusion");
  return Call(op, {data}, Attrs{}, {});
}

TVM_REGISTER_GLOBAL("relay.op.annotation._make.stop_fusion").set_body_typed([](Expr data) {
  return StopFusion(data);
});

RELAY_REGISTER_OP("annotation.stop_fusion")
    .describe(
        R"code(Annotate an expression to prevent it being fused with following expressions.)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .add_type_rel("Identity", IdentityRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

// relay.annotation.cast_hint
TVM_REGISTER_NODE_TYPE(CastHintAttrs);

Expr CastHint(Expr data, DataType dtype) {
  auto attrs = make_object<CastHintAttrs>();
  attrs->dtype = dtype;
  static const Op& op = Op::Get("annotation.cast_hint");
  return Call(op, {data}, Attrs{attrs}, {});
}

RELAY_REGISTER_OP("annotation.cast_hint")
    .describe(
        R"code(Annotate an expression to be cast into specific data type.)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .add_type_rel("Identity", IdentityRel)
    .set_support_level(10)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

RELAY_REGISTER_OP("annotation.bitpack_start")
    .describe(R"code(
Mark the start of bitpacking.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .set_support_level(10)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

RELAY_REGISTER_OP("annotation.bitpack_end")
    .describe(R"code(
Mark the end of bitpacking.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .set_support_level(10)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

TVM_REGISTER_GLOBAL("relay.op.annotation._make.checkpoint").set_body_typed([](Expr data) {
  static const Op& op = Op::Get("annotation.checkpoint");
  return Call(op, {data}, Attrs{}, {});
});

RELAY_REGISTER_OP("annotation.checkpoint")
    .describe(R"code(
Mark a checkpoint for checkpointing memory optimization.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_support_level(10)
    .add_argument("data", "Tensor", "The input data.")
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             Array<te::Tensor> outputs;
                             for (size_t i = 0; i < inputs.size(); ++i) {
                               outputs.push_back(topi::identity(inputs[i]));
                             }
                             return outputs;
                           });

TVM_REGISTER_NODE_TYPE(CompilerAttrs);

RELAY_REGISTER_OP("annotation.compiler_begin")
    .describe(R"code(
Beginning of a region that is handled by a given compiler.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .set_support_level(10)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

TVM_REGISTER_GLOBAL("relay.op.annotation._make.compiler_begin")
    .set_body_typed([](Expr expr, String compiler) {
      auto attrs = make_object<CompilerAttrs>();
      attrs->compiler = compiler;
      static const Op& op = Op::Get("annotation.compiler_begin");
      return Call(op, {expr}, Attrs(attrs), {});
    });

RELAY_REGISTER_OP("annotation.compiler_end")
    .describe(R"code(
End of a region that is handled by a given compiler.
)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input data.")
    .set_support_level(10)
    .add_type_rel("Identity", IdentityRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", false)
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ElemwiseArbitraryLayout)
    .set_attr<FTVMCompute>("FTVMCompute",
                           [](const Attrs& attrs, const Array<te::Tensor>& inputs,
                              const Type& out_dtype) -> Array<te::Tensor> {
                             return {topi::identity(inputs[0])};
                           });

TVM_REGISTER_GLOBAL("relay.op.annotation._make.compiler_end")
    .set_body_typed([](Expr expr, String compiler) {
      auto attrs = make_object<CompilerAttrs>();
      attrs->compiler = compiler;
      static const Op& op = Op::Get("annotation.compiler_end");
      return Call(op, {expr}, Attrs(attrs), {});
    });

}  // namespace relay
}  // namespace tvm
