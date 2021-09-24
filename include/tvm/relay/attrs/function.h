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
 * \file tvm/relay/attrs/function.h
 * \brief Attributes for Relay Functions which don't make sense on PrimFuncs.
 */
#ifndef TVM_RELAY_ATTRS_FUNCTION_H_
#define TVM_RELAY_ATTRS_FUNCTION_H_

namespace tvm {
namespace relay {
/*!
 * \brief Attributes for Relay function definitions which capture the devices for the
 * function parameters and result.
 *
 * See also OnDeviceAttrs in include/tvm/relay/attrs/annotation.h for the companion "on_device"
 * call attributes.
 */
struct FunctionOnDeviceAttrs : public tvm::AttrsNode<FunctionOnDeviceAttrs> {
  /*! \brief Device type on which each of the function's arguments already resides. */
  Array<Integer> param_device_types;
  // TODO(mbs): Replace device types with TargetDevice.
  /*! \brief Device type on which function body should be evaluated. */
  int result_device_type = kInvalidDeviceType;

  TVM_DECLARE_ATTRS(FunctionOnDeviceAttrs, "relay.attrs.FunctionOnDeviceAttrs") {
    TVM_ATTR_FIELD(param_device_types)
        .describe("The type of the virtual device which holds each function parameters.");
    TVM_ATTR_FIELD(result_device_type)
        .describe("The type of the virtual device which will hold the function's result.")
        .set_default(0);
  }
};

namespace attr {

/*!
 * \brief Device annotations for function parameters and results.
 *
 * Type: FunctionOnDeviceAttrs
 */
constexpr static const char* kFunctionAttrsKey = "on_device";

}  // namespace attr

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ATTRS_FUNCTION_H_
