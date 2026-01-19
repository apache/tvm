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

#include "vulkan_image.h"

#include <cstdint>
#include <string>
#include <utility>

#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VkImageCreateInfo MakeImageCreateInfo(VkFormat format, uint32_t width, uint32_t height,
                                      uint32_t layers, VkImageUsageFlags usage) {
  VkImageCreateInfo info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  info.imageType = VK_IMAGE_TYPE_2D;
  info.flags = 0;
  info.format = format;
  info.extent.width = width;
  info.extent.height = height;
  info.extent.depth = 1;  // Must be 1 for 2d images
  info.mipLevels = 1;
  info.arrayLayers = layers;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.tiling = VK_IMAGE_TILING_LINEAR;
  info.usage = usage;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  return info;
}

VulkanImage::VulkanImage(const VulkanDevice& device, VkFormat format, uint32_t width,
                         uint32_t height, uint32_t layers, VkImageUsageFlags usage,
                         uint32_t mem_type_index, std::optional<std::string> mem_scope,
                         std::shared_ptr<VulkanMemory> back_memory)
    : VulkanResource(device, mem_scope, back_memory), width(width), height(height), layers(layers) {
  // Create an image
  VkImageCreateInfo image_info = MakeImageCreateInfo(format, width, height, layers, usage);
  VULKAN_CALL(vkCreateImage(device, &image_info, nullptr, &image));

  VkMemoryRequirements mem_reqs;
  vkGetImageMemoryRequirements(device, image, &mem_reqs);

  // Allocate new memory if no memory is passed in or if the existing memory is not compatible
  if (!memory) {
    AllocateMemory(mem_reqs, mem_type_index);
  }
  // Bind the image to the allocated memory
  VULKAN_CALL(vkBindImageMemory(device, image, memory->memory_, 0));
}

void VulkanImage::AllocateMemory(const VkMemoryRequirements& mem_reqs, uint32_t mem_type_index) {
  VkMemoryAllocateInfo mem_info = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mem_info.allocationSize = mem_reqs.size;
  mem_info.memoryTypeIndex = mem_type_index;

  // Allocate memory
  VkDeviceMemory raw_memory;
  VULKAN_CALL(vkAllocateMemory(device_, &mem_info, nullptr, &raw_memory));

  // Store the allocated memory along with its requirements
  memory = std::make_shared<VulkanMemory>(raw_memory, mem_reqs);
}

VulkanImage::~VulkanImage() {
  if (imageView) {
    vkDestroyImageView(device_, imageView, nullptr);
  }
  if (image) {
    vkDestroyImage(device_, image, nullptr);
  }
}

VulkanImage::VulkanImage(VulkanImage&& other)
    : VulkanResource(std::move(other)), image(other.image), imageView(other.imageView) {
  other.image = VK_NULL_HANDLE;
  other.imageView = VK_NULL_HANDLE;
}

uint32_t VulkanImage::FindMemoryTypeForImage(const VulkanDevice& device,
                                             VkMemoryPropertyFlags properties,
                                             uint32_t typeFilter) {
  VkPhysicalDeviceMemoryProperties memProperties;
  VkPhysicalDevice physicalDeviceHandle =
      device;  // Implicit conversion using operator VkPhysicalDevice()
  vkGetPhysicalDeviceMemoryProperties(physicalDeviceHandle, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("Failed to find suitable memory type!");
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) {
  std::swap(device_, other.device_);
  std::swap(image, other.image);
  std::swap(memory, other.memory);
  std::swap(imageView, other.imageView);
  return *this;
}

void VulkanImage::CreateImageView(VkFormat format) {
  VkImageViewCreateInfo view_info = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view_info.format = format;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = layers;

  VULKAN_CALL(vkCreateImageView(device_, &view_info, nullptr, &imageView));
}

bool VulkanImage::UseDedicatedAllocation(const VulkanDevice& device, VkImage image,
                                         VkDeviceSize* nbytes) {
  if (device.get_image_memory_requirements_2_functions) {
    // Which image to request information about
    VkImageMemoryRequirementsInfo2KHR req_info2 = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR};
    req_info2.image = image;

    // What information to request
    VkMemoryDedicatedRequirementsKHR dedicated_req;
    dedicated_req.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
    dedicated_req.pNext = nullptr;

    VkMemoryRequirements2KHR req2 = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR};
    req2.pNext = &dedicated_req;

    device.get_image_memory_requirements_2_functions->vkGetImageMemoryRequirements2KHR(
        device, &req_info2, &req2);
    if (dedicated_req.requiresDedicatedAllocation || dedicated_req.prefersDedicatedAllocation) {
      *nbytes = req2.memoryRequirements.size;
      return true;
    }
  }
  return false;
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
