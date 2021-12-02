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
 * \file src/relay/op/memory/on_device.cc
 * \brief Helpers for working with the "on_device" 'annotation' call.
 */

#include "./on_device.h"

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

Expr OnDevice(Expr expr, SEScope se_scope, bool is_fixed) {
  ICHECK(!se_scope->IsFullyUnconstrained());
  auto attrs = make_object<OnDeviceAttrs>();
  attrs->se_scope = std::move(se_scope);
  attrs->is_fixed = is_fixed;
  Span span = expr->span;
  return Call(OnDeviceOp(), {std::move(expr)}, Attrs(std::move(attrs)), /*type_args=*/{},
              std::move(span));
}

TVM_REGISTER_GLOBAL("relay.op.annotation._make.OnDevice").set_body_typed(OnDevice);

Expr MaybeOnDevice(Expr expr, SEScope se_scope, bool is_fixed) {
  if (se_scope->IsFullyUnconstrained()) {
    // Nothing to annotate with.
    return expr;
  }
  if (expr->IsInstance<OpNode>() || expr->IsInstance<ConstructorNode>()) {
    // These operators are device polymorphic so no annotation is required.
    return expr;
  }
  if (expr->IsInstance<GlobalVarNode>() || expr->IsInstance<VarNode>()) {
    // The device can be recovered from the binding site of the global or local variable.
    return expr;
  }
  if (expr->IsInstance<FunctionNode>()) {
    // If a primitive function then it is device polymorphic. Otherwise the device is captured
    // by the function's "result_se_scope" attribute.
    return expr;
  }
  OnDeviceProps props = GetOnDeviceProps(expr);
  if (props.body.defined()) {
    // Don't nest on_devices.
    // If the inner and outer device types differ then we need to be careful:
    //  - If the inner on_device is_fixed then it disagrees with the outer.
    //  - If the outer on_device is_fixed then it implies a hidden device_copy
    // Otherwise just use the inner device type and ignore the outer.
    ICHECK(props.se_scope == se_scope || (!is_fixed && !props.is_fixed));
    return OnDevice(props.body, se_scope, is_fixed || props.is_fixed);
  }
  return OnDevice(expr, std::move(se_scope), is_fixed);
}

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
    .set_attr<TNonComputational>("TNonComputational", true);

OnDeviceProps GetOnDeviceProps(const CallNode* call_node) {
  if (call_node->op == OnDeviceOp()) {
    ICHECK_EQ(call_node->args.size(), 1) << "on_device expects one argument";
    ICHECK(call_node->attrs.defined()) << "on_device requires attributes";
    const auto* on_device_attrs = call_node->attrs.as<OnDeviceAttrs>();
    ICHECK(on_device_attrs != nullptr) << "on_device requires OnDeviceAttrs";
    // Follow nesting:
    //   on_device(on_device(expr, se_scope=S), se_scope=T) == {expr, S}
    auto inner = GetOnDeviceProps(call_node->args[0]);
    if (inner.body.defined()) {
      return {inner.body, inner.se_scope, on_device_attrs->is_fixed || inner.is_fixed};
    } else {
      return {call_node->args[0], on_device_attrs->se_scope, on_device_attrs->is_fixed};
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

Function FunctionOnDevice(Function function, Array<SEScope> param_se_scopes,
                          SEScope result_se_scope) {
  return WithAttrs(std::move(function), {{tvm::attr::kParamSEScopes, std::move(param_se_scopes)},
                                         {tvm::attr::kResultSEScope, std::move(result_se_scope)}});
}

TVM_REGISTER_GLOBAL("relay.op.annotation._make.FunctionOnDevice").set_body_typed(FunctionOnDevice);

Function MaybeFunctionOnDevice(Function function, Array<SEScope> param_se_scopes,
                               SEScope result_se_scope) {
  if (std::all_of(param_se_scopes.begin(), param_se_scopes.end(),
                  [](const SEScope& se_scope) { return se_scope->IsFullyUnconstrained(); }) &&
      result_se_scope->IsFullyUnconstrained()) {
    // Nothing to annotate.
    return function;
  }
  return FunctionOnDevice(function, std::move(param_se_scopes), std::move(result_se_scope));
}

SEScope GetFunctionResultSEScope(const FunctionNode* function_node) {
  auto opt_se_scope = function_node->GetAttr<SEScope>(tvm::attr::kResultSEScope);
  return opt_se_scope.value_or(SEScope::FullyUnconstrained());
}

SEScope GetFunctionParamSEScope(const FunctionNode* function_node, size_t i) {
  ICHECK_LT(i, function_node->params.size())
      << "param index " << i << " out of range for function of arity "
      << function_node->params.size();
  auto opt_array = function_node->GetAttr<Array<SEScope>>(tvm::attr::kParamSEScopes);
  if (!opt_array) {
    // No annotation.
    return SEScope::FullyUnconstrained();
  }
  ICHECK_EQ(opt_array.value().size(), function_node->params.size())
      << "annotation parameters do not match function arity";
  return opt_array.value()[i];
}

}  // namespace relay
}  // namespace tvm
