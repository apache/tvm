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

#include "vulkan_thread_entry.h"

#include "vulkan_buffer.h"
#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanThreadEntry::~VulkanThreadEntry() {}

VulkanThreadEntry* VulkanThreadEntry::ThreadLocal() { return VulkanThreadStore::Get(); }

VulkanThreadEntry::VulkanThreadEntry() {
  device.device_id = 0;
  device.device_type = static_cast<DLDeviceType>(kDLVulkan);
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
