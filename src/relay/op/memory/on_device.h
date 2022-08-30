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
 * \brief Wraps \p body in an "on_device" CallNode for \p virtual_device.
 *
 * See \p OnDeviceAttrs for an overview.
 */
Call OnDevice(Expr body, VirtualDevice virtual_device, bool constrain_result = false,
              bool constrain_body = true);

/*! \brief Result of \p GetOnDeviceProps. */
struct OnDeviceProps {
  Expr body;  // = null
  VirtualDevice virtual_device = VirtualDevice::FullyUnconstrained();
  bool constrain_result = false;
  bool constrain_body = false;

  OnDeviceProps() = default;

  OnDeviceProps(Expr body, VirtualDevice virtual_device, bool constrain_result, bool constrain_body)
      : body(std::move(body)),
        virtual_device(std::move(virtual_device)),
        constrain_result(constrain_result),
        constrain_body(constrain_body) {}

  bool is_fixed() const { return constrain_result && constrain_body; }
  bool is_normal() const { return !constrain_result && constrain_body; }
};

/*!
 * \brief Wraps \p body in an "on_device" CallNode, taking all fields other than \p body from \p
 * props.
 */
inline Call OnDeviceWithProps(Expr body, const OnDeviceProps& props) {
  return OnDevice(std::move(body), props.virtual_device, props.constrain_result,
                  props.constrain_body);
}

/*!
 * \brief Wraps \p body in an "on_device" CallNode, but don't constrain the body or result to
 * any particular virtual device. This allows a "device_copy" to be inserted by PlanDevices
 * where required, while at the same time not introducing unnecessary freedom in the device
 * choices.
 */
inline Call OnDeviceCopyOk(Expr body) {
  return OnDevice(std::move(body), VirtualDevice::FullyUnconstrained(),
                  /*constrain_result=*/false, /*constrain_body=*/false);
}

/*!
 * \brief Wraps \p expr in an "on_device" CallNode for \p virtual_device and \p constraint if the
 * \p VirtualDevice for \p expr cannot otherwise be recovered by the lexical scoping convention.
 * This means we will NOT wrap if:
 *  - \p virtual_device is full unconstrained, which signals there are no device annotations
 *    already in play.
 *  - \p expr is an operator or primitive function literal. These are device polymorphic.
 *  - \p expr is a non-primitive function literal. The device is captured by the
 *    "result_virtual_device" attribute on the function itself.
 *  - \p expr is a global var. The device is on the function attributes the global is bound to.
 *  - \p expr is a local var. The device is tracked by the device aware visitors for us.
 *  - \p expr is a constructor. These are device polymorphic.
 * Nested on_device calls will never be constructed, they are instead merged on-the-fly.
 */
Expr MaybeOnDevice(Expr body, VirtualDevice virtual_device, bool constrain_result = false,
                   bool constrain_body = true);

/*! \brief As for MaybeOnDevice, but with both body and result constrained. */
inline Expr MaybeOnDeviceFixed(Expr body, VirtualDevice virtual_device) {
  return MaybeOnDevice(std::move(body), std::move(virtual_device), /*constrain_result=*/true,
                       /*constrain_body=*/true);
}

/*! \brief As for MaybeOnDevice, but with fields other than body taken from \p props. */
inline Expr MaybeOnDeviceWithProps(Expr body, const OnDeviceProps& props) {
  return MaybeOnDevice(std::move(body), props.virtual_device, props.constrain_result,
                       props.constrain_body);
}

/*!
 * \brief Returns the body expression, \p VirtualDevice, and constraint field for \p call_node if it
 * is an "on_device" CallNode. Otherwise returns the null expression, the unconstrained
 * \p VirtualDevice, and \p kBody.
 */
OnDeviceProps GetOnDeviceProps(const CallNode* call_node);

/*!
 * \brief Returns the body expression, \p VirtualDevice, and constraint field for \p expr if it is
 * an "on_device" CallNode. Otherwise returns the null expression, the unconstrained \p
 * VirtualDevice, and \p kBody.
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

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_MEMORY_ON_DEVICE_H_
