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
 * \file tvm/relay/attrs/device_copy.h
 * \brief Attribute for the device copy operator.
 */
#ifndef TVM_RELAY_ATTRS_DEVICE_COPY_H_
#define TVM_RELAY_ATTRS_DEVICE_COPY_H_

#include <tvm/ir/attrs.h>
#include <tvm/target/virtual_device.h>

#include <string>

namespace tvm {
namespace relay {

/*!
 * \brief Options for the device copy operators.
 */
struct DeviceCopyAttrs : public tvm::AttrsNode<DeviceCopyAttrs> {
  VirtualDevice src_virtual_device = VirtualDevice::FullyUnconstrained();
  VirtualDevice dst_virtual_device = VirtualDevice::FullyUnconstrained();

  TVM_DECLARE_ATTRS(DeviceCopyAttrs, "relay.attrs.DeviceCopyAttrs") {
    TVM_ATTR_FIELD(src_virtual_device)
        .describe("The (virtual) device and scope where the op copies data from.");
    TVM_ATTR_FIELD(dst_virtual_device)
        .describe("The (virtual) device and scope where the op copies data to.");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_DEVICE_COPY_H_
