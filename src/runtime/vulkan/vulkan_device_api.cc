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
 *  Copyright (c) 2017 by Contributors
 * \file vulkan_device_api.cc
 */
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>
#include <cstring>
#include "vulkan_common.h"

namespace tvm {
namespace runtime {
namespace vulkan {

VulkanWorkspace::~VulkanWorkspace() {
  for (VulkanContext& ctx : context_) {
    vkDestroyDevice(ctx.device, nullptr);
  }
  if (instance_ != nullptr) {
    vkDestroyInstance(instance_, nullptr);
  }
}

const std::shared_ptr<VulkanWorkspace>& VulkanWorkspace::Global() {
  static std::shared_ptr<VulkanWorkspace> inst = std::make_shared<VulkanWorkspace>();
  return inst;
}

void VulkanWorkspace::SetDevice(TVMContext ctx) {
  VulkanThreadEntry::ThreadLocal()->context.device_id = ctx.device_id;
}

void VulkanWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index< context_.size());
    return;
  }
  CHECK_LT(index, context_.size())
      << "Invalid device id " << index;
  switch (kind) {
    case kMaxThreadsPerBlock: {
      VkPhysicalDeviceProperties phy_prop;
      vkGetPhysicalDeviceProperties(context_[ctx.device_id].phy_device, &phy_prop);
      int64_t value = phy_prop.limits.maxComputeWorkGroupSize[0];
      *rv = value;
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      VkPhysicalDeviceProperties phy_prop;
      vkGetPhysicalDeviceProperties(context_[ctx.device_id].phy_device, &phy_prop);
      int64_t value = phy_prop.limits.maxComputeSharedMemorySize;
      *rv = value;
      break;
    }
    case kWarpSize: {
      *rv = 1;
      break;
    }
    case kComputeVersion: {
      VkPhysicalDeviceProperties phy_prop;
      vkGetPhysicalDeviceProperties(context_[ctx.device_id].phy_device, &phy_prop);
      int64_t value = phy_prop.apiVersion;
      std::ostringstream os;
      os << VK_VERSION_MAJOR(value)
         << "." << VK_VERSION_MINOR(value)
         << "." << VK_VERSION_PATCH(value);
      *rv = os.str();
      break;
    }
    case kDeviceName: return;
    case kMaxClockRate: return;
    case kMultiProcessorCount: return;
    case kExist: break;
    case kMaxThreadDimensions: break;
  }
}

void* VulkanWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment, TVMType type_hint) {
  this->Init();

  VulkanContext& vctx = context_[ctx.device_id];

  VkBufferCreateInfo info;
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0;
  info.size = size;
  info.queueFamilyIndexCount = 1;
  info.pQueueFamilyIndices = &(vctx.queue_family_index);
  info.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  // create buffer
  VkBuffer buffer;
  VULKAN_CALL(vkCreateBuffer(vctx.device, &info, nullptr, &buffer));
  // bind to memory
  VkMemoryAllocateInfo minfo;
  minfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  minfo.pNext = nullptr;
  minfo.allocationSize = size;
  minfo.memoryTypeIndex = vctx.compute_mtype_index;
  VkDeviceMemory memory;
  VULKAN_CALL(vkAllocateMemory(vctx.device, &minfo, nullptr, &memory));
  VULKAN_CALL(vkBindBufferMemory(vctx.device, buffer, memory, 0));

  VulkanBuffer* pbuf = new VulkanBuffer();
  pbuf->memory = memory;
  pbuf->buffer = buffer;
  return pbuf;
}

void VulkanWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  VulkanContext& vctx = context_[ctx.device_id];
  VulkanBuffer* pbuf = static_cast<VulkanBuffer*>(ptr);
  vkDestroyBuffer(vctx.device, pbuf->buffer, nullptr);
  vkFreeMemory(vctx.device, pbuf->memory, nullptr);
  delete pbuf;
}

void VulkanWorkspace::CopyDataFromTo(const void* from,
                                     size_t from_offset,
                                     void* to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMType type_hint,
                                     TVMStreamHandle stream) {
  this->Init();
  CHECK(stream == nullptr);
  TVMContext ctx = ctx_from;
  if (ctx_from.device_type == kDLCPU) ctx = ctx_to;
  VulkanThreadEntry* tls = VulkanThreadEntry::ThreadLocal();
  VulkanCommandBuffer* cmd = tls->CommandPool(ctx.device_id)->Alloc();

  VkCommandBufferBeginInfo cb_begin;
  cb_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cb_begin.pNext = nullptr;
  cb_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  cb_begin.pInheritanceInfo = 0;

  VkSubmitInfo cb_submit;
  cb_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  cb_submit.pNext = nullptr;
  cb_submit.waitSemaphoreCount = 0;
  cb_submit.pWaitSemaphores = nullptr;
  cb_submit.pWaitDstStageMask = 0;
  cb_submit.commandBufferCount = 1;
  cb_submit.pCommandBuffers = &(cmd->cmd_buffer);
  cb_submit.signalSemaphoreCount = 0;
  cb_submit.pSignalSemaphores = nullptr;


  int from_dev_type = static_cast<int>(ctx_from.device_type);
  int to_dev_type = static_cast<int>(ctx_to.device_type);

  if (from_dev_type == kDLVulkan && to_dev_type == kDLVulkan) {
    CHECK_EQ(ctx_from.device_id, ctx_to.device_id)
        << "Vulkan disallow cross device copy.";
    const VulkanContext& vctx = context_[ctx_from.device_id];
    const VulkanBuffer* from_buf = static_cast<const VulkanBuffer*>(from);
    VulkanBuffer* to_buf = static_cast<VulkanBuffer*>(to);
    // The assumption is that subsequence ops only perform compute/transfer
    // 0: begin
    VULKAN_CALL(vkBeginCommandBuffer(cmd->cmd_buffer, &cb_begin));
    // 1: copy
    VkBufferCopy copy_info;
    copy_info.srcOffset = from_offset;
    copy_info.dstOffset = to_offset;
    copy_info.size = size;
    vkCmdCopyBuffer(cmd->cmd_buffer, from_buf->buffer, to_buf->buffer, 1, &copy_info);
    // 2: barrier(transfer-> compute|transfer)
    VkMemoryBarrier barrier_info;
    barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier_info.pNext = nullptr;
    barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_info.dstAccessMask =
        (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(
        cmd->cmd_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier_info, 0, nullptr, 0, nullptr);
    // 3: end
    VULKAN_CALL(vkEndCommandBuffer(cmd->cmd_buffer));
    // 4: submit with cmd->fence
    VULKAN_CALL(vkQueueSubmit(vctx.queue, 1, &cb_submit, cmd->fence));
  } else if (from_dev_type == kDLVulkan && to_dev_type == kDLCPU) {
    const VulkanContext& vctx = context_[ctx_from.device_id];
    const VulkanBuffer* from_buf = static_cast<const VulkanBuffer*>(from);
    VulkanStagingBuffer* temp = tls->StagingBuffer(ctx_from.device_id, size);
    // 0: begin
    VULKAN_CALL(vkBeginCommandBuffer(cmd->cmd_buffer, &cb_begin));
    // 1: copy
    VkBufferCopy copy_info;
    copy_info.srcOffset = from_offset;
    copy_info.dstOffset = 0;
    copy_info.size = size;
    vkCmdCopyBuffer(cmd->cmd_buffer,
                    from_buf->buffer,
                    temp->buffer,
                    1, &copy_info);
    // 2: end
    VULKAN_CALL(vkEndCommandBuffer(cmd->cmd_buffer));
    // 4: submit with cmd->fence
    VULKAN_CALL(vkQueueSubmit(vctx.queue, 1, &cb_submit, cmd->fence));
    // Block until done, to make sure temp can be reused later.
    VULKAN_CALL(vkQueueWaitIdle(vctx.queue));
    // host side invalidation if access is not coherent.
    // so writes from GPU is visible to CPU
    if (!vctx.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = temp->memory;
      mrange.offset = 0;
      mrange.size = size;
      VULKAN_CALL(vkInvalidateMappedMemoryRanges(
          vctx.device, 1, &mrange));
    }
    memcpy(static_cast<char*>(to) + to_offset,
           static_cast<char*>(temp->host_addr),
           size);
  } else if (from_dev_type == kDLCPU && to_dev_type == kDLVulkan) {
    const VulkanContext& vctx = context_[ctx_to.device_id];
    const VulkanBuffer* to_buf = static_cast<const VulkanBuffer*>(to);
    VulkanStagingBuffer* temp = tls->StagingBuffer(ctx_to.device_id, size);
    memcpy(temp->host_addr,
           static_cast<const char*>(from) + from_offset,
           size);
    // host side flush if access is not coherent.
    // so writes from CPU is visible to GPU
    if (!vctx.coherent_staging) {
      VkMappedMemoryRange mrange;
      mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
      mrange.pNext = nullptr;
      mrange.memory = temp->memory;
      mrange.offset = 0;
      mrange.size = size;
      VULKAN_CALL(vkFlushMappedMemoryRanges(vctx.device, 1, &mrange));
    }
    VULKAN_CALL(vkBeginCommandBuffer(cmd->cmd_buffer, &cb_begin));
    // 0: barrier(host->transfer)
    VkMemoryBarrier barrier_info;
    barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier_info.pNext = nullptr;
    barrier_info.srcAccessMask = 0;
    barrier_info.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd->cmd_buffer,
                         VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier_info,
                         0, nullptr, 0, nullptr);
    // 1: copy
    VkBufferCopy copy_info;
    copy_info.srcOffset = 0;
    copy_info.dstOffset = to_offset;
    copy_info.size = size;
    vkCmdCopyBuffer(cmd->cmd_buffer,
                    temp->buffer,
                    to_buf->buffer,
                    1, &copy_info);
    // 2: end
    VULKAN_CALL(vkEndCommandBuffer(cmd->cmd_buffer));
    // 4: submit with cmd->fence
    VULKAN_CALL(vkQueueSubmit(vctx.queue, 1, &cb_submit, cmd->fence));
    // wait until copy finishes, so we can reuse temp next time.
    VULKAN_CALL(vkQueueWaitIdle(vctx.queue));
  } else {
    LOG(FATAL) << "Expect copy from/to Metal or between Metal"
               << ", from=" << from_dev_type
               << ", to=" << to_dev_type;
  }
}

void VulkanWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  VulkanContext& vctx = context_[ctx.device_id];
  VULKAN_CALL(vkQueueWaitIdle(vctx.queue));
}

void* VulkanWorkspace::AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) {
  return VulkanThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
}

void VulkanWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  VulkanThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
}

// VulkanCommandPool
VulkanCommandPool::VulkanCommandPool(const VulkanContext& vctx) {
  ring_.resize(kMaxPending, VulkanCommandBuffer());
  device_ = vctx.device;
  {
    // create command pool
    VkCommandPoolCreateInfo cmd_pool_cinfo;
    cmd_pool_cinfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_cinfo.pNext = nullptr;
    cmd_pool_cinfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cmd_pool_cinfo.queueFamilyIndex = vctx.queue_family_index;
    VULKAN_CALL(vkCreateCommandPool(device_, &cmd_pool_cinfo, nullptr, &cmd_pool_));
  }
  {
    // create descriptor pool
    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = kMaxPending * kMaxNumArgs;
    VkDescriptorPoolCreateInfo descrip_pool_cinfo;
    descrip_pool_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descrip_pool_cinfo.pNext = nullptr;
    descrip_pool_cinfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descrip_pool_cinfo.maxSets = kMaxPending + 2;
    descrip_pool_cinfo.poolSizeCount = 1;
    descrip_pool_cinfo.pPoolSizes = &pool_size;
    VULKAN_CALL(vkCreateDescriptorPool(
        device_, &descrip_pool_cinfo, nullptr, &descriptor_pool_));
  }
  VkCommandBufferAllocateInfo buffer_alloc_info;
  buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  buffer_alloc_info.pNext = nullptr;
  buffer_alloc_info.commandPool = cmd_pool_;
  buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  buffer_alloc_info.commandBufferCount = 1;

  VkFenceCreateInfo fence_cinfo;
  fence_cinfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_cinfo.pNext = nullptr;
  fence_cinfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < ring_.size(); ++i) {
    VULKAN_CALL(vkAllocateCommandBuffers(
        device_, &buffer_alloc_info, &(ring_[i].cmd_buffer)));
    VULKAN_CALL(vkCreateFence(
        device_, &fence_cinfo, nullptr, &(ring_[i].fence)));
  }
}

VulkanCommandPool::~VulkanCommandPool() {
  // wait device to be idle so we know we can recycle buffers
  VULKAN_CALL(vkDeviceWaitIdle(device_));
  // start recycling.
  for (size_t i = 0; i < ring_.size(); ++i) {
    if (ring_[i].cmd_buffer != nullptr) {
      vkFreeCommandBuffers(device_, cmd_pool_, 1, &(ring_[i].cmd_buffer));
      ring_[i].cmd_buffer = nullptr;
    }
    if (ring_[i].fence != VK_NULL_HANDLE) {
      vkDestroyFence(device_, ring_[i].fence, nullptr);
    }
  }
  // delete cmd_pool and descriptor pool
  vkDestroyCommandPool(device_, cmd_pool_, nullptr);
  vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
}

VulkanCommandBuffer* VulkanCommandPool::Alloc() {
  return Alloc(nullptr);
}

VulkanCommandBuffer* VulkanCommandPool::Alloc(
    const VkDescriptorSetLayout* dlayout) {
  // always allocate resource in round robin manner
  VulkanCommandBuffer* e = &(ring_[clock_ptr_]);
  clock_ptr_ = (clock_ptr_ + 1) % ring_.size();
  // Wait until previous usage of commad buffer is finished.
  uint64_t timeout = 1UL << 30UL;
  VkResult res;
  res = vkWaitForFences(device_, 1, &(e->fence), 0, timeout);
  while (res == VK_TIMEOUT) {
    res = vkWaitForFences(device_, 1, &(e->fence), 0, timeout);
  }
  VULKAN_CHECK_ERROR(res);
  vkResetFences(device_, 1, (&e->fence));
  if (e->descriptor_set != VK_NULL_HANDLE) {
    VULKAN_CALL(vkFreeDescriptorSets(
        device_, descriptor_pool_, 1, &(e->descriptor_set)));
    e->descriptor_set = VK_NULL_HANDLE;
  }
  if (dlayout != nullptr) {
    VkDescriptorSetAllocateInfo alloc_info;
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = dlayout;
    VULKAN_CALL(vkAllocateDescriptorSets(
        device_, &alloc_info, &(e->descriptor_set)));
  }
  return e;
}

// VulkanThreadEntry
typedef dmlc::ThreadLocalStore<VulkanThreadEntry> VulkanThreadStore;

VulkanThreadEntry* VulkanThreadEntry::ThreadLocal() {
  return VulkanThreadStore::Get();
}

VulkanCommandPool* VulkanThreadEntry::CommandPool(int device_id) {
  while (pool_.size() <= static_cast<size_t>(device_id)) {
    pool_.emplace_back(std::unique_ptr<VulkanCommandPool>());
  }
  if (pool_[device_id] == nullptr) {
    const VulkanContext& vctx =
        VulkanWorkspace::Global()->context_[device_id];
    pool_[device_id].reset(new VulkanCommandPool(vctx));
  }
  return pool_[device_id].get();
}

VulkanStagingBuffer*
VulkanThreadEntry::StagingBuffer(int device_id, size_t size) {
  if (staging_buffer_.size() <= static_cast<size_t>(device_id)) {
    staging_buffer_.resize(device_id + 1, VulkanStagingBuffer());
  }
  VulkanStagingBuffer& buf = staging_buffer_[device_id];

  if (buf.device != nullptr && buf.size < size) {
    // free previous buffer
    if (buf.host_addr != nullptr) {
      vkUnmapMemory(buf.device, buf.memory);
    }
    if (buf.memory != VK_NULL_HANDLE) {
      vkFreeMemory(buf.device, buf.memory, nullptr);
    }
    if (buf.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(buf.device, buf.buffer, nullptr);
    }
    buf.host_addr = nullptr;
    buf.memory = VK_NULL_HANDLE;
    buf.buffer = VK_NULL_HANDLE;
  }
  const VulkanContext& vctx =
      VulkanWorkspace::Global()->context_[device_id];

  if (buf.device == nullptr) {
    buf.device = vctx.device;
  }
  if (buf.memory == VK_NULL_HANDLE) {
    // allocate the stagging buffer memory if necessary
    VkBufferCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.size = size;
    info.queueFamilyIndexCount = 1;
    info.pQueueFamilyIndices = &(vctx.queue_family_index);
    info.usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VULKAN_CALL(vkCreateBuffer(vctx.device, &info, nullptr, &(buf.buffer)));
    VkMemoryAllocateInfo minfo;
    minfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    minfo.pNext = nullptr;
    minfo.allocationSize = size;
    minfo.memoryTypeIndex = vctx.staging_mtype_index;
    VULKAN_CALL(vkAllocateMemory(vctx.device, &minfo, nullptr, &(buf.memory)));
    VULKAN_CALL(vkBindBufferMemory(vctx.device, (buf.buffer), buf.memory, 0));
    VULKAN_CALL(vkMapMemory(vctx.device, buf.memory, 0, size, 0, &(buf.host_addr)));
    buf.size = size;
  }
  memset(buf.host_addr, 0, size);
  return &buf;
}

VulkanThreadEntry::~VulkanThreadEntry() {
  // Because the thread entry refers to Device API
  // The command buffer always will be destroyed before
  // the instance and device get destroyed.
  // The destruction need to be manually called
  // to ensure the destruction order.
  pool_.clear();
  for (VulkanStagingBuffer buf : staging_buffer_) {
    if (buf.host_addr != nullptr) {
      vkUnmapMemory(buf.device, buf.memory);
    }
    if (buf.memory != VK_NULL_HANDLE) {
      vkFreeMemory(buf.device, buf.memory, nullptr);
    }
    if (buf.buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(buf.device, buf.buffer, nullptr);
    }
  }
}

VkInstance CreateInstance() {
  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = nullptr;
  app_info.pApplicationName = "TVM";
  app_info.applicationVersion = 0;
  app_info.pEngineName = "";
  app_info.engineVersion = 0;
  app_info.apiVersion = VK_MAKE_VERSION(1, 0, 65);

  VkInstanceCreateInfo inst_info;
  inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  inst_info.pNext = nullptr;
  inst_info.flags = 0;
  inst_info.pApplicationInfo = &app_info;
  inst_info.enabledLayerCount = 0;
  inst_info.ppEnabledLayerNames = nullptr;
  inst_info.enabledExtensionCount = 0;
  inst_info.ppEnabledExtensionNames = nullptr;

  VkInstance inst;
  VULKAN_CALL(vkCreateInstance(&inst_info, nullptr, &inst));
  return inst;
}

// find suitable mem_type_index for staging and compute
void FindMemoryTypeIndex(VulkanContext* vctx) {
  // Find suitable compute index.
  VkBuffer buffer;
  VkMemoryRequirements req_staging, req_compute;
  VkBufferCreateInfo info;
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0;
  info.size = 1024;
  info.queueFamilyIndexCount = 1;
  info.pQueueFamilyIndices = &(vctx->queue_family_index);

  // get staging requirement
  info.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VULKAN_CALL(vkCreateBuffer(vctx->device, &info, nullptr, &buffer));
  vkGetBufferMemoryRequirements(vctx->device, buffer, &req_staging);
  vkDestroyBuffer(vctx->device, buffer, nullptr);
  // get compute requirement
  info.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  VULKAN_CALL(vkCreateBuffer(vctx->device, &info, nullptr, &buffer));
  vkGetBufferMemoryRequirements(vctx->device, buffer, &req_compute);
  vkDestroyBuffer(vctx->device, buffer, nullptr);

  // Query phyiscal device property
  // find a memory that is host visible, no need to be consistent
  int win_rank = -1;
  VkPhysicalDeviceMemoryProperties prop;
  vkGetPhysicalDeviceMemoryProperties(vctx->phy_device, &prop);

  for (uint32_t k = 0; k < prop.memoryTypeCount; ++k) {
    VkMemoryType ty = prop.memoryTypes[k];
    size_t heap_size = prop.memoryHeaps[ty.heapIndex].size;
    // host visible
    if (!(ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
    // match copy requirment
    if (!(req_staging.memoryTypeBits & (1 << k))) continue;
    if (heap_size < 1024) continue;
    int rank = 0;
    rank += ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    if (rank > win_rank) {
      win_rank = rank;
      vctx->staging_mtype_index = k;
      vctx->coherent_staging =
          ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
  }
  CHECK_GE(win_rank, 0) << "Cannot find suitable staging memory on device.";

  win_rank = -1;
  for (uint32_t k = 0; k < prop.memoryTypeCount; ++k) {
    VkMemoryType ty = prop.memoryTypes[k];
    size_t heap_size = prop.memoryHeaps[ty.heapIndex].size;
    // host visible
    if (!(ty.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
    // match copy requirment
    if (!(req_staging.memoryTypeBits & (1 << k))) continue;
    if (heap_size < 1024) continue;
    int rank = 0;
    // prefer not host visible
    rank += !(ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (rank > win_rank) {
      win_rank = rank;
      vctx->compute_mtype_index = k;
    }
  }
  CHECK_GE(win_rank, 0) << "Cannot find suitable staging memory on device.";
}

// Get all logic devices that support compute
std::vector<VulkanContext> GetContext(VkInstance instance) {
  std::vector<VulkanContext> result;
  uint32_t phy_dev_count = 0;
  VULKAN_CALL(vkEnumeratePhysicalDevices(
      instance, &phy_dev_count, nullptr));
  std::vector<VkPhysicalDevice> all_phy_devs(phy_dev_count);
  VULKAN_CALL(vkEnumeratePhysicalDevices(
      instance, &phy_dev_count, dmlc::BeginPtr(all_phy_devs)));
  for (VkPhysicalDevice phy_dev : all_phy_devs) {
    uint32_t queue_prop_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        phy_dev, &queue_prop_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_props(queue_prop_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        phy_dev, &queue_prop_count, dmlc::BeginPtr(queue_props));
    uint32_t queue_family_index = 0;
    std::vector<VkDeviceQueueCreateInfo> queue_create_info;

    for (uint32_t i = 0; i < queue_props.size(); i++) {
      // find queues that support compute
      if (VK_QUEUE_COMPUTE_BIT & queue_props[i].queueFlags) {
        float priority = 1.0f;

        VkDeviceQueueCreateInfo info;
        info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.pNext = nullptr;
        info.flags = 0;
        info.queueFamilyIndex = i;
        info.queueCount = 1;
        info.pQueuePriorities = &priority;

        queue_create_info.push_back(info);
        // only use the first available queue for now
        if (queue_create_info.size() == 0) {
          queue_family_index = i;
        }
      }
    }
    if (queue_create_info.size() == 0) continue;

    VkDeviceCreateInfo device_create_info;
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = nullptr;
    device_create_info.flags = 0;
    device_create_info.queueCreateInfoCount
        = static_cast<uint32_t>(queue_create_info.size());
    device_create_info.pQueueCreateInfos = queue_create_info.data();
    device_create_info.enabledLayerCount = 0;
    device_create_info.ppEnabledLayerNames = nullptr;
    device_create_info.enabledExtensionCount = 0;
    device_create_info.ppEnabledExtensionNames = nullptr;
    device_create_info.pEnabledFeatures = nullptr;

    VulkanContext ctx;
    // setup context
    ctx.phy_device = phy_dev;
    vkGetPhysicalDeviceProperties(ctx.phy_device, &(ctx.phy_device_prop));
    VULKAN_CALL(vkCreateDevice(
        phy_dev, &device_create_info, nullptr, &(ctx.device)));
    vkGetDeviceQueue(ctx.device, queue_family_index, 0, &(ctx.queue));
    ctx.queue_family_index = queue_family_index;
    FindMemoryTypeIndex(&ctx);
    // Find suitable memory type for staging and compute
    result.push_back(ctx);
  }
  return result;
}

void VulkanWorkspace::Init() {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  if (initialized_) return;
  initialized_ = true;
  try {
    instance_ = CreateInstance();
    context_ = GetContext(instance_);
    LOG(INFO) << "Initialize Vulkan with " << context_.size() << " devices..";
    for (size_t i = 0; i < context_.size(); ++i) {
      LOG(INFO) << "vulkan(" << i
                <<  ")=\'" << context_[i].phy_device_prop.deviceName
                << "\' phy_dev_id=" << context_[i].phy_device;
    }
  } catch (const dmlc::Error& err) {
    LOG(INFO) << "Cannot initialize vulkan: " << err.what() << "\n"
              << "You can still compile vulkan module but cannot run locally";
  }
}

bool InitVulkan(TVMArgs args, TVMRetValue* rv) {
  vulkan::VulkanWorkspace::Global()->Init();
  return true;
}

TVM_REGISTER_GLOBAL("device_api.vulkan")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = VulkanWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
