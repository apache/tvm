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

#include "vulkan_wrapped_func.h"

#include <dmlc/memory_io.h>

#include <utility>

#include "../file_utils.h"
#include "vulkan_device_api.h"

namespace tvm {
namespace runtime {
namespace vulkan {

void VulkanWrappedFunc::Init(VulkanModuleNode* m, ObjectPtr<Object> sptr,
                             const std::string& func_name, size_t num_buffer_args,
                             size_t num_pack_args,
                             const std::vector<std::string>& launch_param_tags) {
  m_ = m;
  sptr_ = sptr;
  func_name_ = func_name;
  num_buffer_args_ = num_buffer_args;
  num_pack_args_ = num_pack_args;
  launch_param_config_.Init(num_buffer_args + num_pack_args, launch_param_tags);
}

void VulkanWrappedFunc::operator()(TVMArgs args, TVMRetValue* rv,
                                   const ArgUnion64* pack_args) const {
  int device_id = VulkanDeviceAPI::Global()->GetActiveDeviceID();
  auto& device = VulkanDeviceAPI::Global()->device(device_id);
  if (!scache_[device_id]) {
    scache_[device_id] = m_->GetPipeline(device_id, func_name_, num_pack_args_);
  }
  const auto& pipeline = scache_[device_id];
  ThreadWorkLoad wl = launch_param_config_.Extract(args);
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
  const size_t nbytes_scalars = num_pack_args_ * sizeof(ArgUnion64);
  if (pipeline->use_ubo) {
    auto& ubo = device.ThreadLocalUniformBuffer(nbytes_scalars);
    VkDescriptorBufferInfo binfo;
    binfo.buffer = ubo.vk_buf.buffer;
    binfo.offset = 0;
    binfo.range = VK_WHOLE_SIZE;
    descriptor_buffers.push_back(binfo);
  }
  if (device.UseImmediate()) {
    // Can safely capture by reference as this lambda is immediately executed on the calling thread.
    device.ThreadLocalStream().Launch([&](VulkanStreamState* state) {
      vkCmdBindPipeline(state->cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
      ICHECK(pipeline->descriptor_update_template != VK_NULL_HANDLE);
      device.descriptor_template_khr_functions->vkCmdPushDescriptorSetWithTemplateKHR(
          state->cmd_buffer_, pipeline->descriptor_update_template, pipeline->pipeline_layout, 0,
          descriptor_buffers.data());

      if (pipeline->use_ubo) {
        auto& ubo = device.ThreadLocalUniformBuffer(nbytes_scalars);
        memcpy(ubo.host_addr, pack_args, nbytes_scalars);
      } else if (num_pack_args_ > 0) {
        vkCmdPushConstants(state->cmd_buffer_, pipeline->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, num_pack_args_ * sizeof(ArgUnion64),
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

      if (device.UseDebugUtilsLabel()) {
        VkDebugUtilsLabelEXT dispatch_label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                               nullptr,
                                               func_name_.c_str(),
                                               {0.0f, 0.0f, 0.0f, 0.0f}};
        device.queue_insert_debug_utils_label_functions->vkQueueInsertDebugUtilsLabelEXT(
            device.Queue(), &dispatch_label);
      }
    });
    return;
  }

  // Otherwise, the more expensive deferred path.
  std::vector<ArgUnion64> pack_args_storage(pack_args, pack_args + num_pack_args_);
  const auto& deferred_initializer = [&device, pipeline, descriptor_buffers]() {
    std::vector<VkWriteDescriptorSet> write_descriptor_sets;
    write_descriptor_sets.resize(descriptor_buffers.size());
    for (size_t i = 0; i < write_descriptor_sets.size(); i++) {
      write_descriptor_sets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write_descriptor_sets[i].pNext = nullptr;
      write_descriptor_sets[i].dstSet = pipeline->descriptor_set;
      write_descriptor_sets[i].dstBinding = i;
      write_descriptor_sets[i].dstArrayElement = 0;
      write_descriptor_sets[i].descriptorCount = 1;
      write_descriptor_sets[i].pImageInfo = nullptr;
      write_descriptor_sets[i].pBufferInfo = &(descriptor_buffers[i]);
      write_descriptor_sets[i].pTexelBufferView = nullptr;

      if (pipeline->use_ubo && i == write_descriptor_sets.size() - 1) {
        // The last binding is for UBO
        write_descriptor_sets[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      } else {
        write_descriptor_sets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      }
    }
    vkUpdateDescriptorSets(device, write_descriptor_sets.size(), write_descriptor_sets.data(), 0,
                           nullptr);
  };
  const auto& deferred_kernel = [this, pipeline, wl, pack_args_storage, nbytes_scalars,
                                 device_id](VulkanStreamState* state) {
    auto& device = VulkanDeviceAPI::Global()->device(device_id);

    vkCmdBindPipeline(state->cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
    vkCmdBindDescriptorSets(state->cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline->pipeline_layout, 0, 1, &(pipeline->descriptor_set), 0,
                            nullptr);

    if (pipeline->use_ubo) {
      auto& ubo = device.ThreadLocalUniformBuffer(nbytes_scalars);
      memcpy(ubo.host_addr, pack_args_storage.data(), nbytes_scalars);
    } else if (num_pack_args_ > 0) {
      vkCmdPushConstants(state->cmd_buffer_, pipeline->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, pack_args_storage.size() * sizeof(ArgUnion64),
                         pack_args_storage.data());
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
  device.ThreadLocalStream().LaunchDeferred(deferred_initializer, deferred_kernel, deferred_token);

  if (device.UseDebugUtilsLabel()) {
    VkDebugUtilsLabelEXT dispatch_label = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                           nullptr,
                                           func_name_.c_str(),
                                           {0.0f, 0.0f, 0.0f, 0.0f}};
    device.queue_insert_debug_utils_label_functions->vkQueueInsertDebugUtilsLabelEXT(
        device.Queue(), &dispatch_label);
  }
}

VulkanModuleNode::~VulkanModuleNode() {
  // cleanup vulkan related caches.
  for (size_t device_id = 0; device_id < ecache_.size(); ++device_id) {
    for (auto& kv : ecache_[device_id]) {
      auto& pe = kv.second;
      ICHECK(pe);
      const auto& device = VulkanDeviceAPI::Global()->device(device_id);

      if (pe->descriptor_update_template != VK_NULL_HANDLE) {
        device.descriptor_template_khr_functions->vkDestroyDescriptorUpdateTemplateKHR(
            device, pe->descriptor_update_template, nullptr);
      }
      vkDestroyPipeline(device, pe->pipeline, nullptr);
      vkDestroyPipelineLayout(device, pe->pipeline_layout, nullptr);
      vkDestroyDescriptorPool(device, pe->descriptor_pool, nullptr);
      vkDestroyDescriptorSetLayout(device, pe->descriptor_set_layout, nullptr);
      vkDestroyShaderModule(device, pe->shader, nullptr);
    }
  }
}

PackedFunc VulkanModuleNode::GetFunction(const String& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  VulkanWrappedFunc f;
  size_t num_buffer_args = NumBufferArgs(info.arg_types);
  f.Init(this, sptr_to_self, name, num_buffer_args, info.arg_types.size() - num_buffer_args,
         info.launch_param_tags);
  return PackFuncNonBufferArg(std::move(f), info.arg_types);
}

std::shared_ptr<VulkanPipeline> VulkanModuleNode::GetPipeline(size_t device_id,
                                                              const std::string& func_name,
                                                              size_t num_pack_args) {
  auto& device = VulkanDeviceAPI::Global()->device(device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& cp = ecache_[device_id][func_name];
  if (cp) {
    return cp;
  }
  // Create new pipeline
  auto pe = std::make_shared<VulkanPipeline>();
  {
    // create shader
    auto sit = smap_.find(func_name);
    ICHECK(sit != smap_.end());
    pe->use_ubo = sit->second.flag & (1 << ShaderMetaDataFlagMask::kUseUBO);
    const std::vector<uint32_t>& data = sit->second.data;
    VkShaderModuleCreateInfo shader_cinfo;
    shader_cinfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_cinfo.pNext = nullptr;
    shader_cinfo.flags = 0;
    shader_cinfo.codeSize = data.size() * sizeof(uint32_t);
    shader_cinfo.pCode = data.data();
    VULKAN_CALL(vkCreateShaderModule(device, &shader_cinfo, nullptr, &(pe->shader)));
  }
  std::vector<VkDescriptorSetLayoutBinding> arg_binding;
  std::vector<VkDescriptorUpdateTemplateEntryKHR> arg_template;
  std::vector<VkDescriptorPoolSize> descriptor_set_pool_sizes;
  uint32_t num_pod = 0, num_buffer = 0;

  auto push_arg_info = [&arg_binding, &arg_template, &descriptor_set_pool_sizes](
                           uint32_t binding, VkDescriptorType desc_type) {
    {
      auto result = std::find_if(descriptor_set_pool_sizes.begin(), descriptor_set_pool_sizes.end(),
                                 [&](const auto& psize) { return psize.type == desc_type; });
      if (result == descriptor_set_pool_sizes.end()) {
        VkDescriptorPoolSize new_size;
        new_size.type = desc_type;
        new_size.descriptorCount = 1;
        descriptor_set_pool_sizes.push_back(new_size);
      } else {
        result->descriptorCount++;
      }
    }

    {
      VkDescriptorSetLayoutBinding bd;
      bd.binding = binding;
      bd.descriptorType = desc_type;
      bd.descriptorCount = 1;
      bd.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      bd.pImmutableSamplers = nullptr;
      arg_binding.push_back(bd);
    }
    {
      VkDescriptorUpdateTemplateEntryKHR tpl;
      tpl.dstBinding = binding;
      tpl.dstArrayElement = 0;
      tpl.descriptorCount = 1;
      tpl.descriptorType = desc_type;
      tpl.offset = binding * sizeof(VkDescriptorBufferInfo);
      tpl.stride = sizeof(VkDescriptorBufferInfo);
      arg_template.push_back(tpl);
    }
  };

  {
    auto fit = fmap_.find(func_name);
    ICHECK(fit != fmap_.end());
    for (DLDataType arg_type : fit->second.arg_types) {
      if (arg_type.code == kTVMOpaqueHandle) {
        push_arg_info(num_buffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        ++num_buffer;
      } else {
        ++num_pod;
      }
    }
  }

  size_t nbytes_scalars = num_pod * sizeof(ArgUnion64);
  if (pe->use_ubo) {
    // Use UBO instead of push constants
    push_arg_info(num_buffer, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    device.AllocateThreadLocalUniformBuffer(nbytes_scalars);
  }

  {
    VkDescriptorSetLayoutCreateInfo descrip_cinfo;
    descrip_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descrip_cinfo.pNext = nullptr;
    descrip_cinfo.flags = 0;
    if (device.UseImmediate()) {
      descrip_cinfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }
    descrip_cinfo.bindingCount = arg_binding.size();
    descrip_cinfo.pBindings = arg_binding.data();
    VULKAN_CALL(
        vkCreateDescriptorSetLayout(device, &descrip_cinfo, nullptr, &(pe->descriptor_set_layout)));
  }

  if (!device.UseImmediate()) {
    VkDescriptorPoolCreateInfo descrip_pool_cinfo;
    descrip_pool_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descrip_pool_cinfo.pNext = nullptr;
    descrip_pool_cinfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descrip_pool_cinfo.maxSets = 1;
    descrip_pool_cinfo.poolSizeCount = descriptor_set_pool_sizes.size();
    descrip_pool_cinfo.pPoolSizes = descriptor_set_pool_sizes.data();
    VULKAN_CALL(
        vkCreateDescriptorPool(device, &descrip_pool_cinfo, nullptr, &(pe->descriptor_pool)));

    VkDescriptorSetAllocateInfo alloc_info;
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.descriptorPool = pe->descriptor_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &(pe->descriptor_set_layout);
    VULKAN_CALL(vkAllocateDescriptorSets(device, &alloc_info, &(pe->descriptor_set)));
  }

  VkPushConstantRange crange;
  crange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  crange.offset = 0;
  crange.size = sizeof(ArgUnion64) * num_pack_args;

  VkPipelineLayoutCreateInfo playout_cinfo;
  playout_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  playout_cinfo.pNext = nullptr;
  playout_cinfo.flags = 0;
  playout_cinfo.setLayoutCount = 1;
  playout_cinfo.pSetLayouts = &(pe->descriptor_set_layout);

  if (0 < nbytes_scalars && !pe->use_ubo) {
    playout_cinfo.pushConstantRangeCount = 1;
    playout_cinfo.pPushConstantRanges = &crange;
    ICHECK_LE(crange.size, device.device_properties.max_push_constants_size)
        << "The Vulkan shader uses " << crange.size
        << " bytes of push constants, but the device only supports "
        << device.device_properties.max_push_constants_size << "bytes. "
        << "Please rebuild the shader using a smaller limit on push constants size "
        << "by passing -max_push_constants_size=N in the Target string, "
        << "or pass -from_device=0 to query all device parameters.";
  } else {
    playout_cinfo.pushConstantRangeCount = 0;
    playout_cinfo.pPushConstantRanges = nullptr;
  }

  VULKAN_CALL(vkCreatePipelineLayout(device, &playout_cinfo, nullptr, &(pe->pipeline_layout)));

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
  VULKAN_CALL(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_cinfo, nullptr,
                                       &(pe->pipeline)));

  if (device.UseImmediate()) {
    VkDescriptorUpdateTemplateCreateInfoKHR descrip_template_cinfo;
    descrip_template_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
    descrip_template_cinfo.pNext = nullptr;
    descrip_template_cinfo.flags = 0;
    descrip_template_cinfo.descriptorUpdateEntryCount = arg_template.size();
    descrip_template_cinfo.pDescriptorUpdateEntries = arg_template.data();
    descrip_template_cinfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    descrip_template_cinfo.descriptorSetLayout = pe->descriptor_set_layout;
    descrip_template_cinfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    descrip_template_cinfo.pipelineLayout = pe->pipeline_layout;
    descrip_template_cinfo.set = 0;
    VULKAN_CALL(device.descriptor_template_khr_functions->vkCreateDescriptorUpdateTemplateKHR(
        device, &descrip_template_cinfo, nullptr, &(pe->descriptor_update_template)));
  }
  ecache_[device_id][func_name] = pe;
  return pe;
}

void VulkanModuleNode::SaveToFile(const String& file_name, const String& format) {
  std::string fmt = GetFileFormat(file_name, format);
  ICHECK_EQ(fmt, fmt_) << "Can only save to customized format vulkan";
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

void VulkanModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(smap_);
}

String VulkanModuleNode::GetSource(const String& format) {
  // can only return disassembly code.
  return source_;
}

}  // namespace vulkan
}  // namespace runtime
}  // namespace tvm
