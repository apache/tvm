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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_TIMER_H_
#define TVM_RUNTIME_VULKAN_VULKAN_TIMER_H_

#include <tvm/runtime/profiling.h>
#include <vulkan/vulkan.h>

#include "vulkan_device.h"
#include "vulkan_stream.h"

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanDevice;

/*!
 * \brief Timer node for measuring GPU execution time using Vulkan.
 *
 * This class uses Vulkan timestamp queries to measure the time taken
 * by GPU operations between `Start()` and `Stop()` calls.
 */
class VulkanTimerNode : public TimerNode {
 public:
  /*!
   * \brief Constructs a VulkanTimerNode for the specified device.
   * \param dev The TVM device to be used for timing.
   */
  explicit VulkanTimerNode(Device dev);

  /*!
   * \brief Destructor to clean up Vulkan resources.
   */
  ~VulkanTimerNode() override;

  /*!
   * \brief Starts the timer by recording a timestamp.
   */
  void Start() override;

  /*!
   * \brief Stops the timer by recording another timestamp.
   */
  void Stop() override;

  /*!
   * \brief Retrieves the elapsed time in nanoseconds.
   * \return The elapsed time in nanoseconds between Start and Stop.
   */
  int64_t SyncAndGetElapsedNanos() override;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.vulkan.VulkanTimerNode", VulkanTimerNode, TimerNode);

 private:
  Device dev_;                              ///< The TVM device being used.
  VkDevice device_{VK_NULL_HANDLE};         ///< The Vulkan device handle.
  VulkanStream* stream_{nullptr};           ///< The Vulkan stream for command buffer management.
  VkQueryPool query_pool_{VK_NULL_HANDLE};  ///< The Vulkan query pool for timestamp queries.
  float timestamp_period_;    ///< The period (in nanoseconds) for each timestamp tick.
  uint32_t start_query_ = 0;  ///< The index for the start timestamp query.
  uint32_t end_query_ = 1;    ///< The index for the end timestamp query.
  int64_t duration_ = 0;      ///< The measured duration in nanoseconds.

  /*!
   * \brief Creates a Vulkan query pool for timestamp queries.
   */
  void CreateQueryPool();

  /*!
   * \brief Collects timestamps and calculates the duration.
   */
  void CollectTimestamps();

  /*!
   * \brief Cleans up the Vulkan query pool.
   */
  void Cleanup();
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_TIMER_H_
