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

#include "vulkan_resource.h"

#include <tvm/ffi/reflection/registry.h>

#include <string>
#include <utility>

#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanResource::VulkanResource(const VulkanDevice& device, std::optional<std::string> mem_scope,
                               std::shared_ptr<VulkanMemory> back_memory)
    : device_(device), layout(MemoryLayoutFromScope(mem_scope)), memory(back_memory) {}

VulkanResource::~VulkanResource() {}

VulkanResource::VulkanResource(VulkanResource&& other)
    : device_(other.device_), layout(other.layout), memory(other.memory) {
  other.device_ = VK_NULL_HANDLE;
  other.memory = VK_NULL_HANDLE;
}

VulkanResource& VulkanResource::operator=(VulkanResource&& other) {
  if (this != &other) {
    device_ = other.device_;
    layout = other.layout;
    memory = other.memory;
  }
  return *this;
}

VulkanResource::MemoryLayout VulkanResource::MemoryLayoutFromScope(
    std::optional<std::string> mem_scope) {
  if (!mem_scope) {
    return MemoryLayout::kBuffer1D;
  } else if (*mem_scope == "global") {
    return MemoryLayout::kBuffer1D;
  } else if (*mem_scope == "global.texture") {
    return MemoryLayout::kImage2DActivation;
  } else if (*mem_scope == "global.texture-weight") {
    return MemoryLayout::kImage2DWeight;
  } else if (*mem_scope == "global.texture-nhwc") {
    return MemoryLayout::kImage2DNHWC;
  }
  throw std::runtime_error("No memory layout defined for memory of scope: " + *mem_scope);
}

std::string VulkanResource::ScopeFromMemoryLayout(MemoryLayout layout) {
  switch (layout) {
    case MemoryLayout::kBuffer1D:
      return "global";
    case MemoryLayout::kImage2DActivation:
      return "global.texture";
    case MemoryLayout::kImage2DWeight:
      return "global.texture-weight";
    case MemoryLayout::kImage2DNHWC:
      return "global.texture-nhwc";
    default:
      throw std::runtime_error("No scope corresponding to the provided memory layout: " +
                               std::to_string(static_cast<int>(layout)));
  }
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
