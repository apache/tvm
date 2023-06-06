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

#ifndef TVM_RUNTIME_VULKAN_VULKAN_WRAPPED_FUNC_H_
#define TVM_RUNTIME_VULKAN_VULKAN_WRAPPED_FUNC_H_

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../meta_data.h"
#include "../pack_args.h"
#include "../spirv/spirv_shader.h"
#include "../thread_storage_scope.h"
#include "vulkan/vulkan_core.h"
#include "vulkan_common.h"
#include "vulkan_device.h"

namespace tvm {
namespace runtime {
namespace vulkan {

struct VulkanPipeline {
  VulkanDevice* device{nullptr};
  VkShaderModule shader{VK_NULL_HANDLE};
  VkDescriptorSetLayout descriptor_set_layout{VK_NULL_HANDLE};
  VkDescriptorPool descriptor_pool{VK_NULL_HANDLE};
  VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
  VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
  VkPipeline pipeline{VK_NULL_HANDLE};
  VkDescriptorUpdateTemplateKHR descriptor_update_template{VK_NULL_HANDLE};
  bool use_ubo{false};
};

class VulkanModuleNode;

// a wrapped function class to get packed func.
class VulkanWrappedFunc {
 public:
  void Init(VulkanModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_buffer_args, size_t num_pack_args,
            const std::vector<std::string>& launch_param_tags);

  void operator()(TVMArgs args, TVMRetValue* rv, const ArgUnion64* pack_args) const;

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
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
  // Device state cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<std::shared_ptr<VulkanPipeline>, kVulkanMaxNumDevice> scache_;
};

class VulkanModuleNode final : public runtime::ModuleNode {
 public:
  explicit VulkanModuleNode(std::unordered_map<std::string, SPIRVShader> smap,
                            std::unordered_map<std::string, FunctionInfo> fmap, std::string source)
      : smap_(smap), fmap_(fmap), source_(source) {}
  ~VulkanModuleNode();

  const char* type_key() const final { return "vulkan"; }

  /*! \brief Get the property of the runtime module. */
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final;

  std::shared_ptr<VulkanPipeline> GetPipeline(size_t device_id, const std::string& func_name,
                                              size_t num_pack_args);

  void SaveToFile(const String& file_name, const String& format) final;

  void SaveToBinary(dmlc::Stream* stream) final;
  String GetSource(const String& format) final;

 private:
  // function information table.
  std::unordered_map<std::string, SPIRVShader> smap_;
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

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VULKAN_VULKAN_WRAPPED_FUNC_H_
