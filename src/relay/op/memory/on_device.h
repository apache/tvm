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
 * \file relay/op/memory/on_device.h
 * \brief Helpers for working with the "on_device" 'annotation' call.
 */
#ifndef TVM_RELAY_OP_MEMORY_ON_DEVICE_H_
#define TVM_RELAY_OP_MEMORY_ON_DEVICE_H_

#include <tvm/relay/attrs/on_device.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/runtime/ndarray.h>

#include <utility>
#include <vector>

namespace tvm {
namespace relay {

/*! \brief Returns the "on_device" operator. */
const Op& OnDeviceOp();

/*!
 * \brief Wraps \p body in an "on_device" CallNode for \p se_scope.
 *
 * See \p OnDeviceAttrs for an overview.
 */
Call OnDevice(Expr body, SEScope se_scope, bool constrain_result = false,
              bool constrain_body = true);

/*! \brief Result of \p GetOnDeviceProps. */
struct OnDeviceProps {
  Expr body;  // = null
  SEScope se_scope = SEScope::FullyUnconstrained();
  bool constrain_result = false;
  bool constrain_body = false;

  OnDeviceProps() = default;

  OnDeviceProps(Expr body, SEScope se_scope, bool constrain_result, bool constrain_body)
      : body(std::move(body)),
        se_scope(std::move(se_scope)),
        constrain_result(constrain_result),
        constrain_body(constrain_body) {}

  bool is_fixed() const { return constrain_result && constrain_body; }
  bool is_normal() const { return !constrain_result && constrain_body; }
};

/*!
 * \brief As for OnDevice, but taking all fields other than \p body from \p props.
 */
inline Call OnDeviceWithProps(Expr body, const OnDeviceProps& props) {
  return OnDevice(std::move(body), props.se_scope, props.constrain_result, props.constrain_body);
}

/*!
 * \brief As for OnDevice, but don't constrain the body or result to any particular virtual device.
 * This allows a "device_copy" when required.
 */
inline Call OnDeviceCopyOk(Expr body) {
  return OnDevice(std::move(body), SEScope::FullyUnconstrained(),
                  /*constrain_result=*/false, /*constrain_body=*/false);
}

/*!
 * \brief Wraps \p expr in an "on_device" CallNode for \p se_scope and \p constraint if the
 * \p SEScope for \p expr cannot otherwise be recovered by the lexical scoping convention.
 * This means we will NOT wrap if:
 *  - \p se_scope is full unconstrained, which signals there are no device annotations
 *    already in play.
 *  - \p expr is an operator or primitive function literal. These are device polymorphic.
 *  - \p expr is a non-primitive function literal. The device is captured by the
 *    "result_se_scope" attribute on the function itself.
 *  - \p expr is a global var. The device is on the function attributes the global is bound to.
 *  - \p expr is a local var. The device is tracked by the device aware visitors for us.
 *  - \p expr is a constructor. These are device polymorphic.
 * Nested on_device calls will never be constructed, they are instead merged on-the-fly.
 */
Expr MaybeOnDevice(Expr body, SEScope se_scope, bool constrain_result = false,
                   bool constrain_body = true);

/*! \brief As for MaybeOnDevice, but with both body and result constrained. */
inline Expr MaybeOnDeviceFixed(Expr body, SEScope se_scope) {
  return MaybeOnDevice(std::move(body), std::move(se_scope), /*constrain_result=*/true,
                       /*constrain_body=*/true);
}

/*! \brief As for MaybeOnDevice, but with fields other than body taken from \p props. */
inline Expr MaybeOnDeviceWithProps(Expr body, const OnDeviceProps& props) {
  return MaybeOnDevice(std::move(body), props.se_scope, props.constrain_result,
                       props.constrain_body);
}

/*!
 * \brief Returns the body expression, \p SEScope, and constraint field for \p call_node if it
 * is an "on_device" CallNode. Otherwise returns the null expression, the unconstrained
 * \p SEScope, and \p kBody.
 */
OnDeviceProps GetOnDeviceProps(const CallNode* call_node);

/*!
 * \brief Returns the body expression, \p SEScope, and constraint field for \p expr if it is an
 * "on_device" CallNode. Otherwise returns the null expression, the unconstrained \p SEScope,
 * and \p kBody.
 */
OnDeviceProps GetOnDeviceProps(const Expr& expr);

/*!
 * \brief Returns the body of \p expr if it is an "on_device" annotation, otherwise returns
 * \p expr directly.
 */
inline Expr IgnoreOnDevice(const Expr& expr) {
  OnDeviceProps props = GetOnDeviceProps(expr);
  return props.body.defined() ? props.body : expr;
}

/*!
 * \brief Returns \p expr as \p NodeType, or null if it is not of that type. Looks through
 * any "on_device" annotations.
 */
template <typename NodeType>
const NodeType* AsIgnoringOnDevice(const Expr& expr) {
  const auto* node = expr.as<NodeType>();
  if (node != nullptr) {
    return node;
  }
  OnDeviceProps props = GetOnDeviceProps(expr);
  if (!props.body.defined()) {
    return nullptr;
  }
  return props.body.as<NodeType>();
}

/*!
 * \brief Returns \p function annotated with "param_se_scopes" and "result_se_scope"
 * attributes capturing parameter and result \p SEScopes respectively.
 */
Function FunctionOnDevice(Function function, Array<SEScope> param_se_scopes, SEScope body_se_scope);

/*!
 * \brief As for \p FunctionOnDevice, but returns \p function unchanged if all parameters and
 * result \p SEScopes are unconstrained.
 */
Function MaybeFunctionOnDevice(Function function, Array<SEScope> param_se_scopes,
                               SEScope result_se_scope);

/*!
 * \brief Returns the \p SEScope for the resut of \p function_node, or the unconstrained
 * \p SEScope if function does not have the "result_se_scope" annotation.
 */
SEScope GetFunctionResultSEScope(const FunctionNode* function_node);

/*!
 * \brief Returns the \p SEScope for the \p i'th parameter of \p function_node, or
 * the unconstrained \p SEScope if function does not have the "param_se_scopes" annotation.
 */
SEScope GetFunctionParamSEScope(const FunctionNode* function_node, size_t i);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_MEMORY_ON_DEVICE_H_
