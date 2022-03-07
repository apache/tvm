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
 * \file tvm/relay/attrs/on_device.h
 * \brief Attribute for the "on_device" annotation (ie operator).
 */
#ifndef TVM_RELAY_ATTRS_ON_DEVICE_H_
#define TVM_RELAY_ATTRS_ON_DEVICE_H_

#include <tvm/ir/attrs.h>
#include <tvm/target/virtual_device.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Attributes for the "on_device" annotation (ie operator).
 *
 * The Relay call:
 * \code
 *   on_device(sub_expr, virtual_device=S)
 * \endcode
 * constrains \p sub_expr to execute and store its result on the \p VirtualDevice \p S.
 * However the annotation itself may appear in an expression to be executed and stored on a
 * different \p VirtualDevice. If so the compiler will automatically insert a "device_copy" call to
 * mediate the transition between \p VirtualDevices.
 *
 * E.g.: Assuming %x and %y reside on the GPU and %z on the CPU then:
 * \code
 *   multiply(on_device(add(%x, %y), virtual_device=GPU), %z)
 * \endcode
 * indicates the \p add should execute on the GPU but the \p multiply should execute on the CPU.
 * The compiler will rewrite this to:
 * \code
 *   multiply(device_copy(add(%x, %y), src_virtual_device=GPU, dst_virtual_device=CPU), %z)
 * \endcode
 *
 * The \p constraint_body (default true) and \p constraint_result (default false) fields can be
 * used by passes for finer-grained control over how the \p VirtualDevice constraint should be
 * applied.
 */
struct OnDeviceAttrs : public tvm::AttrsNode<OnDeviceAttrs> {
  /*!
   * \brief The \p VirtualDevice to constraint to apply to the body, result, or both body and result
   * of the "on_device" call.
   */
  VirtualDevice virtual_device = VirtualDevice::FullyUnconstrained();

  /*!
   * \brief If false (the default), the result of the "on_device" call is not constrained to be
   * \p virtual_device.
   */
  bool constrain_result = false;

  /*!
   * \brief If true (the default), the body of the "on_device" call is constrained to be \p
   * virtual_device.
   */
  bool constrain_body = true;

  /*!
   * \brief Returns true if both the body and result are constrained.
   */
  bool is_fixed() const { return constrain_result && constrain_body; }

  /*!
   * \brief Returns true only the body is constrained (the 'normal' case).
   */
  bool is_normal() const { return !constrain_result && constrain_body; }

  TVM_DECLARE_ATTRS(OnDeviceAttrs, "relay.attrs.OnDeviceAttrs") {
    TVM_ATTR_FIELD(virtual_device)
        .describe("The (virtual) device to constrain to.")
        .set_default(VirtualDevice::FullyUnconstrained());
    TVM_ATTR_FIELD(constrain_result)
        .describe("Whether the constraint applies to the overall expression")
        .set_default(false);
    TVM_ATTR_FIELD(constrain_body)
        .describe("Whether the constraint applies to the body sub-expression.")
        .set_default(true);
  }
};

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_ATTRS_ON_DEVICE_H_
