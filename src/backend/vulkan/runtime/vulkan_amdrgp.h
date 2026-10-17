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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_AMDRGP_H_
#define TVM_RUNTIME_VULKAN_VULKAN_AMDRGP_H_

namespace tvm {
namespace runtime {
namespace vulkan {

class VulkanDevice;

class VulkanStreamProfiler {
 public:
  enum state { READY = 0, RUNNING, RESET };

  explicit VulkanStreamProfiler(const VulkanDevice* device);

  virtual ~VulkanStreamProfiler() {}

  virtual void reset() { curr_state_ = RESET; }

  virtual void ready() {
    if (curr_state_ == RESET) {
      curr_state_ = READY;
    }
  }

  virtual void capture() = 0;

 protected:
  const VulkanDevice* device_;
  state curr_state_;
  bool available_;
};

class AmdRgpProfiler : public VulkanStreamProfiler {
 public:
  explicit AmdRgpProfiler(const VulkanDevice* device) : VulkanStreamProfiler(device) {}

  void capture();
};

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VULKAN_VULKAN_AMDRGP_H_
