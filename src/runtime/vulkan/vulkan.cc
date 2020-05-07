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

#include <vulkan/vulkan.h>
#include <dmlc/memory_io.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <cstring>


#include "../file_util.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../workspace_pool.h"

#include "vulkan_common.h"
#include "vulkan_module.h"
#include "vulkan_shader.h"
#include "vulkan_stream.h"

namespace tvm {
namespace runtime {
namespace vulkan {

/*! \brief Maximum number of GPU supported in VulkanModule. */
static constexpr const int kVulkanMaxNumDevice = 8;

/*! \brief TVM Vulkan binary pack magic number */
static constexpr const int kVulkanModuleMagic = 0x02700027;

class VulkanThreadEntry {
 public:
  VulkanThreadEntry();
  static VulkanThreadEntry* ThreadLocal();

  ~VulkanThreadEntry() {
    // Because the thread entry refers to Device API
    // The command buffer always will be destroyed before
    // the instance and device get destroyed.
    // The destruction need to be manually called
    // to ensure the destruction order.
    streams_.clear();
    for (const auto& kv : staging_buffers_) {
      if (!kv.second) {
        continue;
      }
      auto& buf = *(kv.second);
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

  TVMContext ctx;
  WorkspacePool pool;
  VulkanStream* Stream(size_t device_id);
  VulkanStagingBuffer* StagingBuffer(int device_id, size_t size);

 private:
  std::unordered_map<size_t, std::unique_ptr<VulkanStream>> streams_;
  std::unordered_map<size_t, std::unique_ptr<VulkanStagingBuffer>> staging_buffers_;
};

struct VulkanBuffer {
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
};

struct VulkanPipeline {
  VulkanContext* vctx_{nullptr};
  VkShaderModule shader{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layout{VK_NULL_HANDLE};
  VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkDescriptorUpdateTemplateKHR descriptor_update_template{VK_NULL_HANDLE};
};

typedef dmlc::ThreadLocalStore<VulkanThreadEntry> VulkanThreadStore;

class VulkanDeviceAPI final : public DeviceAPI {
 public:
  VulkanDeviceAPI();
  ~VulkanDeviceAPI() {
    for (auto& vctx : context_) {
      vkDestroyDevice(vctx.device, nullptr);
    }
    if (instance_) {
      vkDestroyInstance(instance_, nullptr);
    }
  }
  void SetDevice(TVMContext ctx) final { VulkanThreadEntry::ThreadLocal()->ctx = ctx; }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final;
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       DLDataType type_hint) final {
    const auto& vctx = context(ctx.device_id);
    VkBufferCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.size = nbytes;
    info.queueFamilyIndexCount = 1;
    info.pQueueFamilyIndices = &(vctx.queue_family_index);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    // create buffer
    VkBuffer buffer;
    VULKAN_CALL(vkCreateBuffer(vctx.device, &info, nullptr, &buffer));
    // bind to memory
    VkBufferMemoryRequirementsInfo2KHR req_info2;
    req_info2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR;
    req_info2.pNext = 0;
    req_info2.buffer = buffer;

    VkMemoryRequirements2KHR req2;
    req2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
    req2.pNext = 0;

    VkMemoryDedicatedRequirementsKHR dedicated_req;
    dedicated_req.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
    dedicated_req.pNext = 0;
    req2.pNext = &dedicated_req;

    bool dedicated_allocation = false;
    if (vctx.get_buffer_memory_requirements_2_functions) {
      vctx.get_buffer_memory_requirements_2_functions->vkGetBufferMemoryRequirements2KHR(
          vctx.device, &req_info2, &req2);
      dedicated_allocation =
          dedicated_req.requiresDedicatedAllocation || dedicated_req.prefersDedicatedAllocation;
    }

    VkDeviceMemory memory;
    if (!dedicated_allocation) {
      VkMemoryAllocateInfo minfo;
      minfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      minfo.pNext = nullptr;
      minfo.allocationSize = nbytes;
      minfo.memoryTypeIndex = vctx.compute_mtype_index;
      VULKAN_CALL(vkAllocateMemory(vctx.device, &minfo, nullptr, &memory));
    } else {
      VkMemoryAllocateInfo minfo;
      minfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      minfo.pNext = nullptr;
      minfo.allocationSize = req2.memoryRequirements.size;
      minfo.memoryTypeIndex = vctx.compute_mtype_index;

      VkMemoryDedicatedAllocateInfoKHR mdinfo;
      mdinfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
      mdinfo.pNext = 0;
      mdinfo.image = 0;
      mdinfo.buffer = buffer;
      minfo.pNext = &mdinfo;
      VULKAN_CALL(vkAllocateMemory(vctx.device, &minfo, nullptr, &memory));
    }
    VULKAN_CALL(vkBindBufferMemory(vctx.device, buffer, memory, 0));
    VulkanBuffer* pbuf = new VulkanBuffer();
    pbuf->memory = memory;
    pbuf->buffer = buffer;
    return pbuf;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    // Before releasing the vkBuffer, call sync to
    // finish all the vulkan commands that reference the buffer.
    StreamSync(ctx, nullptr);

    const auto& vctx = context(ctx.device_id);
    auto* pbuf = static_cast<VulkanBuffer*>(ptr);
    vkDestroyBuffer(vctx.device, pbuf->buffer, nullptr);
    vkFreeMemory(vctx.device, pbuf->memory, nullptr);
    delete pbuf;
  }

  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    CHECK(stream == nullptr);
    TVMContext ctx = ctx_from;
    if (ctx_from.device_type == kDLCPU) {
      ctx = ctx_to;
    }

    int from_dev_type = static_cast<int>(ctx_from.device_type);
    int to_dev_type = static_cast<int>(ctx_to.device_type);
    if (from_dev_type == kDLVulkan && to_dev_type == kDLVulkan) {
      VulkanThreadEntry::ThreadLocal()
          ->Stream(ctx_from.device_id)
          ->Launch([=](VulkanStreamState* state) {
            // 1: copy
            const auto* from_buf = static_cast<const VulkanBuffer*>(from);
            auto* to_buf = static_cast<VulkanBuffer*>(to);
            VkBufferCopy copy_info;
            copy_info.srcOffset = from_offset;
            copy_info.dstOffset = to_offset;
            copy_info.size = size;
            vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, to_buf->buffer, 1, &copy_info);
            // 2: barrier(transfer-> compute|transfer)
            CHECK_EQ(ctx_from.device_id, ctx_to.device_id) << "Vulkan disallow cross device copy.";
            VkMemoryBarrier barrier_info;
            barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier_info.pNext = nullptr;
            barrier_info.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier_info.dstAccessMask =
                (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            vkCmdPipelineBarrier(
                state->cmd_buffer_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                &barrier_info, 0, nullptr, 0, nullptr);
          });

    } else if (from_dev_type == kDLVulkan && to_dev_type == kDLCPU) {
      const auto* from_buf = static_cast<const VulkanBuffer*>(from);
      const auto& vctx = context(ctx_from.device_id);
      auto* temp = VulkanThreadEntry::ThreadLocal()->StagingBuffer(ctx_from.device_id, size);
      VulkanThreadEntry::ThreadLocal()
          ->Stream(ctx_from.device_id)
          ->Launch([&](VulkanStreamState* state) {
            VkBufferCopy copy_info;
            copy_info.srcOffset = from_offset;
            copy_info.dstOffset = 0;
            copy_info.size = size;
            vkCmdCopyBuffer(state->cmd_buffer_, from_buf->buffer, temp->buffer, 1, &copy_info);
          });
      VulkanThreadEntry::ThreadLocal()->Stream(ctx_from.device_id)->Synchronize();
      if (!vctx.coherent_staging) {
        VkMappedMemoryRange mrange;
        mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mrange.pNext = nullptr;
        mrange.memory = temp->memory;
        mrange.offset = 0;
        mrange.size = VK_WHOLE_SIZE;  // size;
        VULKAN_CALL(vkInvalidateMappedMemoryRanges(vctx.device, 1, &mrange));
      }
      memcpy(static_cast<char*>(to) + to_offset, static_cast<char*>(temp->host_addr), size);
    } else if (from_dev_type == kDLCPU && to_dev_type == kDLVulkan) {
      const auto& vctx = context(ctx_to.device_id);
      const auto* to_buf = static_cast<const VulkanBuffer*>(to);
      VulkanStagingBuffer* temp =
          VulkanThreadEntry::ThreadLocal()->StagingBuffer(ctx_to.device_id, size);
      memcpy(temp->host_addr, static_cast<const char*>(from) + from_offset, size);
      // host side flush if access is not coherent.
      // so writes from CPU is visible to GPU
      if (!vctx.coherent_staging) {
        VkMappedMemoryRange mrange;
        mrange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mrange.pNext = nullptr;
        mrange.memory = temp->memory;
        mrange.offset = 0;
        mrange.size = VK_WHOLE_SIZE;  // size;
        VULKAN_CALL(vkFlushMappedMemoryRanges(vctx.device, 1, &mrange));
      }

      VulkanThreadEntry::ThreadLocal()
          ->Stream(ctx_from.device_id)
          ->Launch([&](VulkanStreamState* state) {
            // 0: barrier(host->transfer)
            VkMemoryBarrier barrier_info;
            barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier_info.pNext = nullptr;
            barrier_info.srcAccessMask = 0;
            barrier_info.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(state->cmd_buffer_, VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier_info, 0, nullptr, 0,
                                 nullptr);
            // 1: copy
            VkBufferCopy copy_info;
            copy_info.srcOffset = 0;
            copy_info.dstOffset = to_offset;
            copy_info.size = size;
            vkCmdCopyBuffer(state->cmd_buffer_, temp->buffer, to_buf->buffer, 1, &copy_info);
          });
      // TODO(tulloch): should we instead make the staging buffer a property of the
      // Stream? This would allow us to elide synchronizations here.
      VulkanThreadEntry::ThreadLocal()->Stream(ctx_from.device_id)->Synchronize();
    } else {
      LOG(FATAL) << "Expect copy from/to Vulkan or between Vulkan"
                 << ", from=" << from_dev_type << ", to=" << to_dev_type;
    }
  }

  // Always use the default stream
  TVMStreamHandle CreateStream(TVMContext ctx) {
    LOG(FATAL) << "Not implemented";
    return nullptr;
  }

  void FreeStream(TVMContext ctx, TVMStreamHandle stream) {
    LOG(FATAL) << "Not implemented";
    return;
  }

  void SyncStreamFromTo(TVMContext ctx, TVMStreamHandle event_src, TVMStreamHandle event_dst) {
    LOG(FATAL) << "Not implemented";
    return;
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    CHECK(stream == nullptr);
    VulkanThreadEntry::ThreadLocal()->Stream(ctx.device_id)->Synchronize();
  }

  void SetStream(TVMContext ctx, TVMStreamHandle stream) final {
    LOG(FATAL) << "Not implemented";
    return;
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) final {
    return VulkanThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
  }

  void FreeWorkspace(TVMContext ctx, void* data) final {
    VulkanThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
  }

  static const std::shared_ptr<VulkanDeviceAPI>& Global() {
    static std::shared_ptr<VulkanDeviceAPI> inst = std::make_shared<VulkanDeviceAPI>();
    return inst;
  }

  const VulkanContext& context(size_t device_id) const {
    CHECK_LT(device_id, context_.size());
    return context_[device_id];
  }

 private:
  VkInstance instance_{nullptr};
  // The physical devices, have 1 to 1 mapping to devices
  std::vector<VulkanContext> context_;
};

void VulkanDeviceAPI::GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < context_.size());
    return;
  }
  CHECK_LT(index, context_.size()) << "Invalid device id " << index;
  const auto& vctx = context(index);
  switch (kind) {
    case kMaxThreadsPerBlock: {
      VkPhysicalDeviceProperties phy_prop;
      vkGetPhysicalDeviceProperties(vctx.phy_device, &phy_prop);
      int64_t value = phy_prop.limits.maxComputeWorkGroupSize[0];
      *rv = value;
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      VkPhysicalDeviceProperties phy_prop;
      vkGetPhysicalDeviceProperties(vctx.phy_device, &phy_prop);
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
      vkGetPhysicalDeviceProperties(vctx.phy_device, &phy_prop);
      int64_t value = phy_prop.apiVersion;
      std::ostringstream os;
      os << VK_VERSION_MAJOR(value) << "." << VK_VERSION_MINOR(value) << "."
         << VK_VERSION_PATCH(value);
      *rv = os.str();
      break;
    }
    case kDeviceName:
      return;
    case kMaxClockRate:
      return;
    case kMultiProcessorCount:
      return;
    case kExist:
      break;
    case kMaxThreadDimensions:
      break;
    case kGcnArch:
      return;
  }
}

VulkanDeviceAPI::VulkanDeviceAPI() {
  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = nullptr;
  app_info.pApplicationName = "TVM";
  app_info.applicationVersion = 0;
  app_info.pEngineName = "";
  app_info.engineVersion = 0;
  app_info.apiVersion = VK_MAKE_VERSION(1, 0, 0);

  VkInstanceCreateInfo inst_info;
  inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  inst_info.pNext = nullptr;
  inst_info.flags = 0;

  const auto layers = []() -> std::vector<const char*> {
    uint32_t inst_layer_prop_count;
    VULKAN_CALL(vkEnumerateInstanceLayerProperties(&inst_layer_prop_count, nullptr));
    std::vector<VkLayerProperties> inst_layer_prop(inst_layer_prop_count);
    VULKAN_CALL(vkEnumerateInstanceLayerProperties(&inst_layer_prop_count, inst_layer_prop.data()));
    std::vector<const char*> l;
    for (const auto& lp : inst_layer_prop) {
      // TODO(tulloch): add CMAKE options.
      (void)lp;  // suppress unused variable warning.
#ifdef USE_VULKAN_VALIDATION
      if (std::strcmp(lp.layerName, "VK_LAYER_LUNARG_standard_validation") == 0) {
        l.push_back("VK_LAYER_LUNARG_standard_validation");
      }
      if (std::strcmp(lp.layerName, "VK_LAYER_LUNARG_parameter_validation") == 0) {
        l.push_back("VK_LAYER_LUNARG_parameter_validation");
      }
      if (std::strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
        l.push_back("VK_LAYER_KHRONOS_validation");
      }
#endif
    }
    return l;
  }();

  const auto instance_extensions = []() -> std::vector<const char*> {
    uint32_t inst_extension_prop_count;
    VULKAN_CALL(
        vkEnumerateInstanceExtensionProperties(nullptr, &inst_extension_prop_count, nullptr));
    std::vector<VkExtensionProperties> inst_extension_prop(inst_extension_prop_count);
    VULKAN_CALL(vkEnumerateInstanceExtensionProperties(nullptr, &inst_extension_prop_count,
                                                       inst_extension_prop.data()));
    std::vector<const char*> extensions;
    for (const auto& ip : inst_extension_prop) {
      if (std::strcmp(ip.extensionName, "VK_KHR_get_physical_device_properties2") == 0) {
        extensions.push_back("VK_KHR_get_physical_device_properties2");
      }
    }
    return extensions;
  }();

  inst_info.pApplicationInfo = &app_info;
  inst_info.enabledLayerCount = layers.size();
  inst_info.ppEnabledLayerNames = layers.data();
  inst_info.enabledExtensionCount = instance_extensions.size();
  inst_info.ppEnabledExtensionNames = instance_extensions.data();

  VULKAN_CALL(vkCreateInstance(&inst_info, nullptr, &instance_));

  uint32_t phy_dev_count = 0;
  VULKAN_CALL(vkEnumeratePhysicalDevices(instance_, &phy_dev_count, nullptr));
  std::vector<VkPhysicalDevice> all_phy_devs(phy_dev_count);
  VULKAN_CALL(vkEnumeratePhysicalDevices(instance_, &phy_dev_count, dmlc::BeginPtr(all_phy_devs)));
  for (VkPhysicalDevice phy_dev : all_phy_devs) {
    uint32_t queue_prop_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queue_prop_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_props(queue_prop_count);
    vkGetPhysicalDeviceQueueFamilyProperties(phy_dev, &queue_prop_count,
                                             dmlc::BeginPtr(queue_props));
    uint32_t queue_family_index = 0;
    std::vector<VkDeviceQueueCreateInfo> queue_create_info;
    float priority = 1.0f;
    for (uint32_t i = 0; i < queue_props.size(); i++) {
      // find queues that support compute
      if (VK_QUEUE_COMPUTE_BIT & queue_props[i].queueFlags) {
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

    VulkanContext ctx;
    // setup context
    ctx.phy_device = phy_dev;
    vkGetPhysicalDeviceProperties(ctx.phy_device, &(ctx.phy_device_prop));

    const auto extensions = [&]() {
      uint32_t device_extension_prop_count;
      VULKAN_CALL(vkEnumerateDeviceExtensionProperties(ctx.phy_device, nullptr,
                                                       &device_extension_prop_count, nullptr));
      std::vector<VkExtensionProperties> device_extension_prop(device_extension_prop_count);
      VULKAN_CALL(vkEnumerateDeviceExtensionProperties(
          ctx.phy_device, nullptr, &device_extension_prop_count, device_extension_prop.data()));
      std::vector<const char*> extensions;
      for (const auto& dp : device_extension_prop) {
        if ((std::strcmp(dp.extensionName, "VK_KHR_push_descriptor") == 0) && dp.specVersion > 0) {
          extensions.push_back("VK_KHR_push_descriptor");
        }
        if ((std::strcmp(dp.extensionName, "VK_KHR_descriptor_update_template") == 0) &&
            dp.specVersion > 0) {
          extensions.push_back("VK_KHR_descriptor_update_template");
        }
        if ((std::strcmp(dp.extensionName, "VK_KHR_get_memory_requirements2") == 0) &&
            dp.specVersion > 0) {
          extensions.push_back("VK_KHR_get_memory_requirements2");
        }
        if ((std::strcmp(dp.extensionName, "VK_KHR_dedicated_allocation") == 0) &&
            dp.specVersion > 0) {
          extensions.push_back("VK_KHR_dedicated_allocation");
        }
      }
      return extensions;
    }();
    VkDeviceCreateInfo device_create_info;
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = nullptr;
    device_create_info.flags = 0;
    device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_info.size());
    device_create_info.pQueueCreateInfos = queue_create_info.data();
    device_create_info.enabledLayerCount = 0;
    device_create_info.ppEnabledLayerNames = nullptr;
    device_create_info.enabledExtensionCount = extensions.size();
    device_create_info.ppEnabledExtensionNames = extensions.data();
    device_create_info.pEnabledFeatures = nullptr;
    VULKAN_CALL(vkCreateDevice(phy_dev, &device_create_info, nullptr, &(ctx.device)));
    ctx.queue_mutex.reset(new std::mutex());
    vkGetDeviceQueue(ctx.device, queue_family_index, 0, &(ctx.queue));
    ctx.queue_family_index = queue_family_index;
    // Find suitable memory type for staging and compute
    // Find suitable compute index.
    VkBuffer buffer;
    VkMemoryRequirements req_staging, req_compute;
    VkBufferCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext = nullptr;
    info.flags = 0;
    info.size = 1024;
    info.queueFamilyIndexCount = 1;
    info.pQueueFamilyIndices = &(ctx.queue_family_index);
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // get staging requirement
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VULKAN_CALL(vkCreateBuffer(ctx.device, &info, nullptr, &buffer));
    vkGetBufferMemoryRequirements(ctx.device, buffer, &req_staging);
    vkDestroyBuffer(ctx.device, buffer, nullptr);
    // get compute requirement
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VULKAN_CALL(vkCreateBuffer(ctx.device, &info, nullptr, &buffer));
    vkGetBufferMemoryRequirements(ctx.device, buffer, &req_compute);
    vkDestroyBuffer(ctx.device, buffer, nullptr);

    // Query phyiscal device property
    // find a memory that is host visible, no need to be consistent
    int win_rank = -1;
    VkPhysicalDeviceMemoryProperties prop;
    vkGetPhysicalDeviceMemoryProperties(ctx.phy_device, &prop);

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
        ctx.staging_mtype_index = k;
        ctx.coherent_staging = ty.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
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
        ctx.compute_mtype_index = k;
      }
    }
    CHECK_GE(win_rank, 0) << "Cannot find suitable local memory on device.";
    auto has_extension = [&extensions](const char* query) {
      return std::any_of(extensions.begin(), extensions.end(),
                         [&](const char* extension) { return std::strcmp(query, extension) == 0; });
    };

#ifdef USE_VULKAN_IMMEDIATE_MODE
    if (has_extension("VK_KHR_push_descriptor") &&
        has_extension("VK_KHR_descriptor_update_template")) {
      ctx.descriptor_template_khr_functions =
          std::unique_ptr<VulkanDescriptorTemplateKHRFunctions>(
              new VulkanDescriptorTemplateKHRFunctions());
      ctx.descriptor_template_khr_functions->vkCreateDescriptorUpdateTemplateKHR =
          CHECK_NOTNULL((PFN_vkCreateDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(
              ctx.device, "vkCreateDescriptorUpdateTemplateKHR"));
      ctx.descriptor_template_khr_functions->vkDestroyDescriptorUpdateTemplateKHR =
          CHECK_NOTNULL((PFN_vkDestroyDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(
              ctx.device, "vkDestroyDescriptorUpdateTemplateKHR"));
      ctx.descriptor_template_khr_functions->vkUpdateDescriptorSetWithTemplateKHR =
          CHECK_NOTNULL((PFN_vkUpdateDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(
              ctx.device, "vkUpdateDescriptorSetWithTemplateKHR"));
      ctx.descriptor_template_khr_functions->vkCmdPushDescriptorSetWithTemplateKHR =
          CHECK_NOTNULL((PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(
              ctx.device, "vkCmdPushDescriptorSetWithTemplateKHR"));
    }
#endif

#ifdef USE_VULKAN_DEDICATED_ALLOCATION
    if (has_extension("VK_KHR_get_memory_requirements2") &&
        has_extension("VK_KHR_dedicated_allocation")) {
      ctx.get_buffer_memory_requirements_2_functions =
          std::unique_ptr<VulkanGetBufferMemoryRequirements2Functions>(
              new VulkanGetBufferMemoryRequirements2Functions());
      ctx.get_buffer_memory_requirements_2_functions->vkGetBufferMemoryRequirements2KHR =
          CHECK_NOTNULL((PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(
              ctx.device, "vkGetBufferMemoryRequirements2KHR"));
    }
#endif
    context_.push_back(std::move(ctx));
  }

  LOG(INFO) << "Initialize Vulkan with " << context_.size() << " devices..";
  for (size_t i = 0; i < context_.size(); ++i) {
    LOG(INFO) << "vulkan(" << i << ")=\'" << context_[i].phy_device_prop.deviceName
              << "\' phy_dev_id=" << context_[i].phy_device
              << " use_immediate=" << context_[i].UseImmediate();
  }
}  // namespace vulkan
class VulkanModuleNode;

// a wrapped function class to get packed func.
class VulkanWrappedFunc {
 public:
  void Init(VulkanModuleNode* m,
            ObjectPtr<Object> sptr,
            const std::string& func_name,
            size_t num_buffer_args, size_t num_pack_args,
            const std::vector<std::string>& thread_axis_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    num_buffer_args_ = num_buffer_args;
    num_pack_args_ = num_pack_args;
    thread_axis_cfg_.Init(num_buffer_args + num_pack_args, thread_axis_tags);
  }

  void operator()(TVMArgs args, TVMRetValue* rv, const ArgUnion* pack_args) const;

 private:
  // internal module
  VulkanModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // v The name of the function.
  std::string func_name_;
  // Number of buffer arguments
  size_t num_buffer_args_;
  // number of packed arguments.
  size_t num_pack_args_;
  // Device state cache per device.
  // mark as mutable, to enable lazy initialization
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;

  mutable std::array<std::shared_ptr<VulkanPipeline>, kVulkanMaxNumDevice> scache_;
};

// Multi-device enabled module.
class VulkanModuleNode final : public runtime::ModuleNode {
 public:
  explicit VulkanModuleNode(std::unordered_map<std::string, VulkanShader> smap,
                             std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : smap_(smap), fmap_(fmap), source_(source) {}

  const char* type_key() const final { return "vulkan"; }

  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr_to_self) final {
    CHECK_EQ(sptr_to_self.get(), this);
    CHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
    auto it = fmap_.find(name);
    if (it == fmap_.end()) return PackedFunc();
    const FunctionInfo& info = it->second;
    VulkanWrappedFunc f;
    size_t num_buffer_args = NumBufferArgs(info.arg_types);
    f.Init(this, sptr_to_self, name, num_buffer_args, info.arg_types.size() - num_buffer_args,
           info.thread_axis_tags);
    return PackFuncNonBufferArg(std::move(f), info.arg_types);
  }

  ~VulkanModuleNode() {
    // cleanup vulkan related caches.
    for (size_t device_id = 0; device_id < ecache_.size(); ++device_id) {
      for (auto& kv : ecache_[device_id]) {
        auto& pe = kv.second;
        CHECK(pe);
        const auto& vctx = VulkanDeviceAPI::Global()->context(device_id);

        if (pe->descriptor_update_template != VK_NULL_HANDLE) {
          vctx.descriptor_template_khr_functions->vkDestroyDescriptorUpdateTemplateKHR(
              vctx.device, pe->descriptor_update_template, nullptr);
        }
        vkDestroyPipeline(vctx.device, pe->pipeline, nullptr);
        vkDestroyPipelineLayout(vctx.device, pe->pipeline_layout, nullptr);
        vkDestroyDescriptorPool(vctx.device, pe->descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(vctx.device, pe->descriptor_set_layout, nullptr);
        vkDestroyShaderModule(vctx.device, pe->shader, nullptr);
      }
    }
  }

  std::shared_ptr<VulkanPipeline> GetPipeline(size_t device_id, const std::string& func_name,
                                               size_t num_pack_args) {
    const auto& vctx = VulkanDeviceAPI::Global()->context(device_id);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& cp = ecache_[device_id][func_name];
    if (cp) {
      return cp;
    }
    // Create new pipeline
    auto pe = std::shared_ptr<VulkanPipeline>(new VulkanPipeline());
    {
      // create shader
      auto sit = smap_.find(func_name);
      CHECK(sit != smap_.end());
      const std::vector<uint32_t>& data = sit->second.data;
      VkShaderModuleCreateInfo shader_cinfo;
      shader_cinfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      shader_cinfo.pNext = nullptr;
      shader_cinfo.flags = 0;
      shader_cinfo.codeSize = data.size() * sizeof(uint32_t);
      shader_cinfo.pCode = data.data();
      VULKAN_CALL(vkCreateShaderModule(vctx.device, &shader_cinfo, nullptr, &(pe->shader)));
    }
    std::vector<VkDescriptorSetLayoutBinding> arg_binding;
    std::vector<VkDescriptorUpdateTemplateEntryKHR> arg_template;
    uint32_t num_pod = 0, num_buffer = 0;
    {
      auto fit = fmap_.find(func_name);
      CHECK(fit != fmap_.end());
      for (DLDataType arg_type : fit->second.arg_types) {
        if (arg_type.code == kTVMOpaqueHandle) {
          {
            VkDescriptorSetLayoutBinding bd;
            bd.binding = num_buffer;
            bd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bd.descriptorCount = 1;
            bd.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bd.pImmutableSamplers = nullptr;
            arg_binding.push_back(bd);
          }
          {
            VkDescriptorUpdateTemplateEntryKHR tpl;
            tpl.dstBinding = num_buffer;
            tpl.dstArrayElement = 0;
            tpl.descriptorCount = 1;
            tpl.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            tpl.offset = num_buffer * sizeof(VkDescriptorBufferInfo);
            tpl.stride = sizeof(VkDescriptorBufferInfo);
            arg_template.push_back(tpl);
          }
          ++num_buffer;
        } else {
          ++num_pod;
        }
      }
    }

    {
      VkDescriptorSetLayoutCreateInfo descrip_cinfo;
      descrip_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      descrip_cinfo.pNext = nullptr;
      descrip_cinfo.flags = 0;
      if (vctx.UseImmediate()) {
        descrip_cinfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
      }
      descrip_cinfo.bindingCount = arg_binding.size();
      descrip_cinfo.pBindings = arg_binding.data();
      VULKAN_CALL(vkCreateDescriptorSetLayout(vctx.device, &descrip_cinfo, nullptr,
                                              &(pe->descriptor_set_layout)));
    }

    {
      VkDescriptorPoolSize pool_size;
      pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      pool_size.descriptorCount = arg_binding.size();
      VkDescriptorPoolCreateInfo descrip_pool_cinfo;
      descrip_pool_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      descrip_pool_cinfo.pNext = nullptr;
      descrip_pool_cinfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
      descrip_pool_cinfo.maxSets = 1;
      descrip_pool_cinfo.poolSizeCount = 1;
      descrip_pool_cinfo.pPoolSizes = &pool_size;
      VULKAN_CALL(vkCreateDescriptorPool(vctx.device, &descrip_pool_cinfo, nullptr,
                                         &(pe->descriptor_pool)));
    }

    if (!vctx.UseImmediate()) {
      VkDescriptorSetAllocateInfo alloc_info;
      alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      alloc_info.pNext = nullptr;
      alloc_info.descriptorPool = pe->descriptor_pool;
      alloc_info.descriptorSetCount = 1;
      alloc_info.pSetLayouts = &(pe->descriptor_set_layout);
      VULKAN_CALL(vkAllocateDescriptorSets(vctx.device, &alloc_info, &(pe->descriptor_set)));
    }

    VkPushConstantRange crange;
    crange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    crange.offset = 0;
    crange.size = sizeof(ArgUnion) * num_pack_args;

    VkPipelineLayoutCreateInfo playout_cinfo;
    playout_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    playout_cinfo.pNext = nullptr;
    playout_cinfo.flags = 0;
    playout_cinfo.setLayoutCount = 1;
    playout_cinfo.pSetLayouts = &(pe->descriptor_set_layout);

    if (num_pack_args != 0) {
      playout_cinfo.pushConstantRangeCount = 1;
      playout_cinfo.pPushConstantRanges = &crange;
      CHECK_LE(crange.size, vctx.phy_device_prop.limits.maxPushConstantsSize);
    } else {
      playout_cinfo.pushConstantRangeCount = 0;
      playout_cinfo.pPushConstantRanges = nullptr;
    }

    VULKAN_CALL(
        vkCreatePipelineLayout(vctx.device, &playout_cinfo, nullptr, &(pe->pipeline_layout)));

    VkComputePipelineCreateInfo pipeline_cinfo;
    pipeline_cinfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_cinfo.pNext = nullptr;
    pipeline_cinfo.flags = 0;
    pipeline_cinfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_cinfo.stage.pNext = nullptr;
    pipeline_cinfo.stage.flags = 0;
    pipeline_cinfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_cinfo.stage.module = pe->shader;
    pipeline_cinfo.stage.pName = func_name.c_str();
    pipeline_cinfo.stage.pSpecializationInfo = nullptr;
    pipeline_cinfo.layout = pe->pipeline_layout;
    pipeline_cinfo.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_cinfo.basePipelineIndex = 0;
    VULKAN_CALL(vkCreateComputePipelines(vctx.device, VK_NULL_HANDLE, 1, &pipeline_cinfo, nullptr,
                                         &(pe->pipeline)));

    if (vctx.UseImmediate()) {
      VkDescriptorUpdateTemplateCreateInfoKHR descrip_template_cinfo;
      descrip_template_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
      descrip_template_cinfo.pNext = 0;
      descrip_template_cinfo.flags = 0;
      descrip_template_cinfo.descriptorUpdateEntryCount = arg_template.size();
      descrip_template_cinfo.pDescriptorUpdateEntries = arg_template.data();
      descrip_template_cinfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
      descrip_template_cinfo.descriptorSetLayout = pe->descriptor_set_layout;
      descrip_template_cinfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
      descrip_template_cinfo.pipelineLayout = pe->pipeline_layout;
      descrip_template_cinfo.set = 0;
      VULKAN_CALL(vctx.descriptor_template_khr_functions->vkCreateDescriptorUpdateTemplateKHR(
          vctx.device, &descrip_template_cinfo, 0, &(pe->descriptor_update_template)));
    }
    ecache_[device_id][func_name] = pe;
    return pe;
  }

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, fmt_) << "Can only save to customized format vulkan";
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    std::string data_bin;
    dmlc::MemoryStringStream fs(&data_bin);
    dmlc::Stream* stream = &fs;
    uint32_t magic = kVulkanModuleMagic;
    stream->Write(magic);
    stream->Write(smap_);
    SaveBinaryToFile(file_name, data_bin);
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(smap_);
  }
  std::string GetSource(const std::string& format) final {
    // can only return source code.
    return source_;
  }

 private:
  // the binary data
  std::vector<uint32_t> data_;
  // function information table.
  std::unordered_map<std::string, VulkanShader> smap_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The format
  std::string fmt_{"vulkan"};
  // The source
  std::string source_;

  // Guards accesses to `ecache_`
  std::mutex mutex_;
  std::array<std::unordered_map<std::string, std::shared_ptr<VulkanPipeline>>, kVulkanMaxNumDevice>
      ecache_;
};

Module VulkanModuleCreate(std::unordered_map<std::string, VulkanShader> smap,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<VulkanModuleNode>(smap, fmap, source);
  return Module(n);
}

VulkanThreadEntry* VulkanThreadEntry::ThreadLocal() { return VulkanThreadStore::Get(); }

VulkanStagingBuffer* VulkanThreadEntry::StagingBuffer(int device_id, size_t size) {
  if (!staging_buffers_[device_id]) {
    staging_buffers_[device_id] = std::unique_ptr<VulkanStagingBuffer>(new VulkanStagingBuffer());
  }
  auto& buf = *(staging_buffers_[device_id]);
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
  const auto& vctx = VulkanDeviceAPI::Global()->context(device_id);

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
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
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

VulkanThreadEntry::VulkanThreadEntry()
    : pool(static_cast<DLDeviceType>(kDLVulkan), VulkanDeviceAPI::Global()) {
  ctx.device_id = 0;
  ctx.device_type = static_cast<DLDeviceType>(kDLVulkan);
}

VulkanStream* VulkanThreadEntry::Stream(size_t device_id) {
  if (!streams_[device_id]) {
    streams_[device_id] = std::unique_ptr<VulkanStream>(
        new VulkanStream(&VulkanDeviceAPI::Global()->context(device_id)));
  }
  return streams_[device_id].get();
}

void VulkanWrappedFunc::operator()(TVMArgs args, TVMRetValue* rv,
                                    const ArgUnion* pack_args) const {
  int device_id = VulkanThreadEntry::ThreadLocal()->ctx.device_id;
  CHECK_LT(device_id, kVulkanMaxNumDevice);
  const auto& vctx = VulkanDeviceAPI::Global()->context(device_id);
  if (!scache_[device_id]) {
    scache_[device_id] = m_->GetPipeline(device_id, func_name_, num_pack_args_);
  }
  const auto& pipeline = scache_[device_id];
  ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
  std::vector<VkDescriptorBufferInfo> descriptor_buffers;
  descriptor_buffers.resize(num_buffer_args_);
  for (size_t i = 0; i < num_buffer_args_; ++i) {
    void* buf = args[static_cast<int>(i)];
    VkDescriptorBufferInfo binfo;
    binfo.buffer = static_cast<VulkanBuffer*>(buf)->buffer;
    binfo.offset = 0;
    binfo.range = VK_WHOLE_SIZE;
    descriptor_buffers[i] = binfo;
  }
  if (vctx.UseImmediate()) {
    // Can safely capture by reference as this lambda is immediately executed on the calling thread.
    VulkanThreadEntry::ThreadLocal()->Stream(device_id)->Launch([&](VulkanStreamState* state) {
      vkCmdBindPipeline(state->cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
      CHECK(pipeline->descriptor_update_template != VK_NULL_HANDLE);
      vctx.descriptor_template_khr_functions->vkCmdPushDescriptorSetWithTemplateKHR(
          state->cmd_buffer_, pipeline->descriptor_update_template, pipeline->pipeline_layout, 0,
          descriptor_buffers.data());
      if (num_pack_args_ != 0) {
        vkCmdPushConstants(state->cmd_buffer_, pipeline->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, num_pack_args_ * sizeof(ArgUnion),
                           pack_args);
      }
      vkCmdDispatch(state->cmd_buffer_, wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
      VkMemoryBarrier barrier_info;
      barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier_info.pNext = nullptr;
      barrier_info.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
      barrier_info.dstAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
      vkCmdPipelineBarrier(state->cmd_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           1, &barrier_info, 0, nullptr, 0, nullptr);
    });
    return;
  }

  // Otherwise, the more expensive deferred path.
  std::vector<ArgUnion> pack_args_storage(pack_args, pack_args + num_pack_args_);
  const auto& deferred_initializer = [&vctx, pipeline, descriptor_buffers]() {
    std::vector<VkWriteDescriptorSet> write_descriptor_sets;
    write_descriptor_sets.resize(descriptor_buffers.size());
    for (size_t i = 0; i < write_descriptor_sets.size(); i++) {
      write_descriptor_sets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor_sets[i].pNext = 0;
      write_descriptor_sets[i].dstSet = pipeline->descriptor_set;
      write_descriptor_sets[i].dstBinding = i;
      write_descriptor_sets[i].dstArrayElement = 0;
      write_descriptor_sets[i].descriptorCount = 1;
      write_descriptor_sets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      write_descriptor_sets[i].pImageInfo = 0;
      write_descriptor_sets[i].pBufferInfo = &(descriptor_buffers[i]);
      write_descriptor_sets[i].pTexelBufferView = 0;
    }
    vkUpdateDescriptorSets(vctx.device, write_descriptor_sets.size(), write_descriptor_sets.data(),
                           0, 0);
  };
  const auto& deferred_kernel = [pipeline, wl, pack_args_storage](VulkanStreamState* state) {
    vkCmdBindPipeline(state->cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(state->cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->pipeline_layout, 0, 1, &(pipeline->descriptor_set), 0,
                            nullptr);
    if (pack_args_storage.size() != 0) {
      vkCmdPushConstants(state->cmd_buffer_, pipeline->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, pack_args_storage.size() * sizeof(ArgUnion), pack_args_storage.data());
    }
    vkCmdDispatch(state->cmd_buffer_, wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
    VkMemoryBarrier barrier_info;
    barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier_info.pNext = nullptr;
    barrier_info.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier_info.dstAccessMask = (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(state->cmd_buffer_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &barrier_info, 0, nullptr, 0, nullptr);
  };
  VulkanStreamToken deferred_token;
  deferred_token.descriptor_set_ = pipeline->descriptor_set;
  deferred_token.buffers_.resize(descriptor_buffers.size());
  for (size_t i = 0; i < descriptor_buffers.size(); ++i) {
    deferred_token.buffers_[i] = descriptor_buffers[i].buffer;
  }
  VulkanThreadEntry::ThreadLocal()->Stream(device_id)->LaunchDeferred(
      deferred_initializer, deferred_kernel, deferred_token);
}

Module VulkanModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, VulkanShader> smap;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  dmlc::MemoryStringStream fs(&data);
  dmlc::Stream* stream = &fs;
  uint32_t magic;
  stream->Read(&magic);
  CHECK_EQ(magic, kVulkanModuleMagic) << "VulkanModule Magic mismatch";
  stream->Read(&smap);
  return VulkanModuleCreate(smap, fmap, "");
}

Module VulkanModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::unordered_map<std::string, VulkanShader> smap;
  std::unordered_map<std::string, FunctionInfo> fmap;

  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&smap);
  return VulkanModuleCreate(smap, fmap, "");
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_vulkan").set_body_typed(VulkanModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_vulkan").set_body_typed(VulkanModuleLoadBinary);

TVM_REGISTER_GLOBAL("device_api.vulkan").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = VulkanDeviceAPI::Global().get();
  *rv = static_cast<void*>(ptr);
});

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
