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

#include "vulkan_instance.h"

#include <cstdlib>
#include <utility>

#include "../../support/utils.h"
#include "vulkan_common.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanInstance::VulkanInstance() {
  const auto layers = []() {
    std::vector<const char*> layers;

    if (support::BoolEnvironmentVar("TVM_VULKAN_ENABLE_VALIDATION_LAYERS")) {
      uint32_t inst_layer_prop_count;
      VULKAN_CALL(vkEnumerateInstanceLayerProperties(&inst_layer_prop_count, nullptr));
      std::vector<VkLayerProperties> inst_layer_prop(inst_layer_prop_count);
      VULKAN_CALL(
          vkEnumerateInstanceLayerProperties(&inst_layer_prop_count, inst_layer_prop.data()));

      for (const auto& lp : inst_layer_prop) {
        if (std::strcmp(lp.layerName, "VK_LAYER_LUNARG_standard_validation") == 0) {
          layers.push_back("VK_LAYER_LUNARG_standard_validation");
        }
        if (std::strcmp(lp.layerName, "VK_LAYER_LUNARG_parameter_validation") == 0) {
          layers.push_back("VK_LAYER_LUNARG_parameter_validation");
        }
        if (std::strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
          layers.push_back("VK_LAYER_KHRONOS_validation");
        }
      }
    }
    return layers;
  }();

  {
    std::vector<const char*> required_extensions{};
    std::vector<const char*> optional_extensions{"VK_KHR_get_physical_device_properties2"};

    // Check if RGP support is needed. If needed, enable VK_EXT_debug_utils extension for
    // inserting debug labels into the queue.
    if (support::BoolEnvironmentVar("TVM_USE_AMD_RGP")) {
      LOG(INFO) << "Push VK_EXT_debug_utils";
      required_extensions.push_back("VK_EXT_debug_utils");
    }

    uint32_t inst_extension_prop_count;
    VULKAN_CALL(
        vkEnumerateInstanceExtensionProperties(nullptr, &inst_extension_prop_count, nullptr));
    std::vector<VkExtensionProperties> inst_extension_prop(inst_extension_prop_count);
    VULKAN_CALL(vkEnumerateInstanceExtensionProperties(nullptr, &inst_extension_prop_count,
                                                       inst_extension_prop.data()));

    enabled_extensions_ =
        FindEnabledExtensions(inst_extension_prop, required_extensions, optional_extensions);
  }

  uint32_t api_version = VK_MAKE_VERSION(1, 0, 0);
  {
    // Result from vkGetInstanceProcAddr is NULL if driver only
    // supports vulkan 1.0.
    auto vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(
        nullptr, "vkEnumerateInstanceVersion");
    if (vkEnumerateInstanceVersion) {
      vkEnumerateInstanceVersion(&api_version);
    }
  }

  {
    VkApplicationInfo app_info;
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = nullptr;
    app_info.pApplicationName = "TVM";
    app_info.applicationVersion = 0;
    app_info.pEngineName = "";
    app_info.engineVersion = 0;
    app_info.apiVersion = api_version;

    VkInstanceCreateInfo inst_info;
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pNext = nullptr;
    inst_info.flags = 0;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledLayerCount = layers.size();
    inst_info.ppEnabledLayerNames = layers.data();
    inst_info.enabledExtensionCount = enabled_extensions_.size();
    inst_info.ppEnabledExtensionNames = enabled_extensions_.data();

    VULKAN_CALL(vkCreateInstance(&inst_info, nullptr, &instance_));
  }
}

VulkanInstance::~VulkanInstance() {
  if (instance_) {
    vkDestroyInstance(instance_, nullptr);
  }
}

VulkanInstance::VulkanInstance(VulkanInstance&& other) { do_swap(std::move(other)); }

VulkanInstance& VulkanInstance::operator=(VulkanInstance&& other) {
  do_swap(std::move(other));
  return *this;
}

void VulkanInstance::do_swap(VulkanInstance&& other) {
  if (this == &other) {
    return;
  }

  std::swap(enabled_extensions_, other.enabled_extensions_);
  std::swap(instance_, other.instance_);
}

bool VulkanInstance::HasExtension(const char* query) const {
  return std::any_of(enabled_extensions_.begin(), enabled_extensions_.end(),
                     [&](const char* extension) { return std::strcmp(query, extension) == 0; });
}

std::vector<VkPhysicalDevice> VulkanInstance::GetPhysicalDevices() const {
  uint32_t device_count = 0;
  VULKAN_CALL(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr));
  std::vector<VkPhysicalDevice> devices(device_count);
  VULKAN_CALL(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()));
  return devices;
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
