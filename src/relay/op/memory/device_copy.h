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
 * \file relay/op/memory/device_copy.h
 * \brief Helpers for working with "device_copy" attributes.
 */

#ifndef TVM_RELAY_OP_MEMORY_DEVICE_COPY_H_
#define TVM_RELAY_OP_MEMORY_DEVICE_COPY_H_

#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>

#include <utility>

#include "../call/call.h"

namespace tvm {
namespace relay {

/*! \brief Returns the "device_copy" operator. */
const Op& DeviceCopyOp();

/*!
 * \brief Wraps \p expr in a "device_copy" CallNode indicating it should be evaluated and
 * stored at \p src_virtual_device but then copied to \p dst_virtual_device.
 */
Expr DeviceCopy(Expr expr, VirtualDevice src_virtual_device, VirtualDevice dst_virtual_device);

/*!
 * \brief Wraps \p expr in a "device_copy" CallNode indicating it should be evaluated and
 * stored at \p src_virtual_device but then copied to \p dst_virtual_device.However, return \p expr
 * directly if \p src_virtual_device and \p dst_virtual_device are (structurally) the same.
 */
Expr MaybeDeviceCopy(Expr expr, VirtualDevice src_virtual_device, VirtualDevice dst_virtual_device);

/*! \brief Result of \p GetDeviceCopyProps. */
struct DeviceCopyProps {
  Expr body;  // = null
  VirtualDevice src_virtual_device = VirtualDevice::FullyUnconstrained();
  VirtualDevice dst_virtual_device = VirtualDevice::FullyUnconstrained();

  DeviceCopyProps() = default;

  DeviceCopyProps(Expr body, VirtualDevice src_virtual_device, VirtualDevice dst_virtual_device)
      : body(std::move(body)),
        src_virtual_device(std::move(src_virtual_device)),
        dst_virtual_device(std::move(dst_virtual_device)) {}
};

/*!
 * \brief Returns the body expression, source, and destination \p VirtualDevices for \p call_node
 * if it is a "device_copy" CallNode. Otherwise returns the null expression and unconstrained
 * virtual device.
 */
DeviceCopyProps GetDeviceCopyProps(const CallNode* call_node);

/*!
 * \brief Returns the body expression, source, and destination \p VirtualDevices for \p expr if it
 * is a "device_copy" Call. Otherwise returns the null expression and unconstrained virtual device.
 */
DeviceCopyProps GetDeviceCopyProps(const Expr& expr);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_MEMORY_DEVICE_COPY_H_
