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
 * \file relay/attrs/annotation.h
 * \brief Helpers for working with various 'annotation' attributes.
 */
#ifndef TVM_RELAY_OP_ANNOTATION_ANNOTATION_H_
#define TVM_RELAY_OP_ANNOTATION_ANNOTATION_H_

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>
#include <tvm/runtime/ndarray.h>

#include <vector>

namespace tvm {
namespace relay {

/*! \brief Returns the "on_device" operator. */
const Op& OnDeviceOp();

/*!
 * \brief Wraps \p expr in an "on_device" CallNode for \p device_type and \p is_fixed.
 */
Expr OnDevice(Expr expr, DLDeviceType device_type, bool is_fixed);

/*!
 * \brief Wraps \p expr in an "on_device" CallNode for \p device_type and \p is_fixed. However
 * returns \p expr directly if:
 *  - \p device_type is \p kInvalidDeviceType, which signals there are no device annotations
 *    already in play.
 *  - \p expr is an operator or primitive function literal. These are device polymorphic.
 *  - \p expr is a global or local var. These already have an implied device.
 *  - \p expr is a constructor. There should probably be device polymorphic but are in an
 *    in-between state at the moment.
 */
Expr OptOnDevice(Expr expr, DLDeviceType device_type, bool is_fixed);

/*! \brief Result of \p GetOnDeviceProps. */
struct OnDeviceProps {
  Expr body;  // = null
  DLDeviceType device_type = kInvalidDeviceType;
  bool is_fixed = false;

  OnDeviceProps() = default;

  OnDeviceProps(const Expr& body, DLDeviceType deviceType, bool isFixed)
      : body(body), device_type(deviceType), is_fixed(isFixed) {}
};

/*!
 * \brief Returns the body expression, device type and is_fixed field for \p call_node if it is
 * an "on_device" CallNode. Otherwise returns the null expression, \p kInvalidDeviceType and \p
 * false.
 */
OnDeviceProps GetOnDeviceProps(const CallNode* call_node);

/*!
 * \brief Returns the body expression, device type and is_fixed field for \p expr if it is an
 * "on_device" CallNode. Otherwise returns the null expression, \p kInvalidDeviceType and \p false.
 */
OnDeviceProps GetOnDeviceProps(const Expr& expr);

/*! \brief Returns true if \p expr is an on_device CallNode. */
inline bool IsOnDeviceCall(const Expr& expr) { return GetOnDeviceProps(expr).body.defined(); }

/*!
 * \brief Returns \p function annotated with "on_device" attributes capturing parameter and result
 * devices types. However returns \p function directly if all device types are \p
 * kInvalidDeviceType.
 */
Function FunctionOnDevice(Function function, Array<Integer> param_device_types,
                          DLDeviceType body_device_type);
Function FunctionOnDevice(Function function, const std::vector<DLDeviceType>& param_device_types,
                          DLDeviceType body_device_type);

/*!
 * \brief Returns the device type for the resut of \p function_node, or \p kInvalidDeviceType
 * if function does not have "on_device" annotation.
 */
DLDeviceType GetFunctionResultDeviceType(const FunctionNode* function_node);

/*!
 * \brief Returns the device type for the \p i'th parameter of \p function_node, or
 * \p kInvalidDeviceType if function does not have "on_device" annotation.
 */
DLDeviceType GetFunctionParamDeviceType(const FunctionNode* function_node, size_t i);

/*! \brief Wraps \p data in a "stop_fusion" annotation. */
Expr StopFusion(Expr data);

/*! \brief Wraps \p data in a "cast_hint" annotation for \p dtype. */
Expr CastHint(Expr data, DataType dtype);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_ANNOTATION_ANNOTATION_H_
