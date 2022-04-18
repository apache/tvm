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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_STREAM_H_
#define TVM_RUNTIME_VULKAN_VULKAN_STREAM_H_

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "vulkan_amdrgp.h"
#include "vulkan_common.h"

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanDevice;

class VulkanStreamState {
 public:
  VkCommandBuffer cmd_buffer_;
  VkFence fence_;
};

// Used to identify state that should only be used once-per-stream.
struct VulkanStreamToken {
  VkDescriptorSet descriptor_set_{VK_NULL_HANDLE};
  std::vector<VkBuffer> buffers_;
};

/*!
 *  \brief Wrapper around a vulkan command buffer
 *
 *  The VulkanStream collects commands into a VkCommandBuffer.  When a
 *  newly submitted command requires resources reserved by an
 *  already-submitted command, all of the queued commands are
 *  submitted to the GPU, and the CPU waits for all queued commands to
 *  finish.  The queued commands can also be explicitly pushed/waited
 *  on by calling VulkanStream::Synchronize.
 *
 *  Currently, there exists one VulkanStream for each GPU device, for
 *  each CPU thread.  Each time a VulkanWrappedFunc is called, it is
 *  submitted to the VulkanStream associated with the submitting CPU
 *  thread, and associated the thread-specific active device set by
 *  `DeviceAPI::SetDevice`.
 */
class VulkanStream {
 public:
  explicit VulkanStream(const VulkanDevice* device);

  ~VulkanStream();

  /*! \brief Push the kernel onto the stream's command buffer.
   *
   * If device.UseImmediate() is true, the kernel is executed
   * immediately to update the command buffer.  Otherwise, it is added
   * to the list of deferred updates to be pushed onto the command
   * buffer.
   *
   * Assumes that there are no descriptor sets or buffers accessed by this kernel.
   *
   */
  void Launch(const std::function<void(VulkanStreamState*)>& kernel);

  /*! \brief Push the kernel onto the stream's command buffer.
   *
   * Can only be called if device.UseImmediate() is false.  The
   * kernel is delayed, and isn't pushed to the command buffer until
   * all kernels are collected.
   *
   * \param deferred_initializer Updates the descriptor set.  Only
   * called if the deferred_token has differences from
   *
   * \param deferred_kernel Submits updates to the command buffer.
   *
   * \param deferred_token Indicates which descriptor set and buffers
   * are accessed by this kernel.  No two kernels in the command
   * buffer can use the same descriptor set.
   *
   */
  void LaunchDeferred(const std::function<void()>& deferred_initializer,
                      const std::function<void(VulkanStreamState*)>& deferred_kernel,
                      const VulkanStreamToken& deferred_token);

  // reset profiler state
  void ProfilerReset() {
    if (profiler_) {
      profiler_->reset();
    }
  }

  // set profiler to READY state after reset
  void ProfilerReady() {
    if (profiler_) {
      profiler_->ready();
    }
  }

  // Synchronize the current stream `state_` with respect to the host.
  void Synchronize();

 private:
  const VulkanDevice* device_;
  std::unique_ptr<VulkanStreamState> state_;
  // An index of deferred tokens, allowing us to efficiently detect duplicated
  // deferred_initializer blocks.
  std::unordered_map<VkDescriptorSet, std::vector<VulkanStreamToken>> deferred_tokens_;
  std::vector<std::function<void(VulkanStreamState*)>> deferred_kernels_;
  VkCommandPool cmd_pool_;
  VulkanStreamProfiler* profiler_ = nullptr;
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_STREAM_H_
