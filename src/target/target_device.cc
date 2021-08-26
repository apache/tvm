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
 * \brief The implementation of the TargetDevice object for representing compilation target + virtual device.
 * \file src/target/target_device.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/tag.h>
#include <tvm/target/target_device.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stack>

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetDeviceNode);

TargetDevice::TargetDevice(Target target, Device virtual_device) {
  auto object = make_object<TargetDeviceNode>();
  object->target = target;
  object->virtual_device_id = virtual_device.device_id;
  object->device_type = virtual_device.device_type;
  data_ = std::move(object);
}

TargetDevice::operator Device() {
  return Device { .device_id = (*this)->virtual_device_id,
                  .device_type = (*this)->device_type };
}

}  // namespace tvm
