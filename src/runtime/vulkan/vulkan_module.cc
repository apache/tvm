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
 *  Copyright (c) 2018 by Contributors
 * \file vulkan_module.cc
 */
#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
#include <array>
#include <string>
#include <mutex>
#include "vulkan_common.h"
#include "vulkan_module.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"


namespace tvm {
namespace runtime {

void VulkanShader::Save(dmlc::Stream* writer) const {
  writer->Write(flag);
  writer->Write(data);
}

bool VulkanShader::Load(dmlc::Stream* reader) {
  if (!reader->Read(&flag)) return false;
  if (!reader->Read(&data)) return false;
  return true;
}

// Multi-device enabled module.
class VulkanModuleNode final :public runtime::ModuleNode {
 public:
  // Pipeline cache states
  struct PipelineEntry {
    VkShaderModule shader{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptor_layout{VK_NULL_HANDLE};
    VkPipeline pipeline{VK_NULL_HANDLE};
  };
  // constructor
  explicit VulkanModuleNode(std::unordered_map<std::string, VulkanShader> smap,
                            std::unordered_map<std::string, FunctionInfo> fmap,
                            std::string source)
      : smap_(smap), fmap_(fmap), source_(source) {
  }

  ~VulkanModuleNode() {
    // cleanup vulkan related caches.
    for (DeviceEntry& e : finfo_) {
      if (e.device == nullptr) continue;
      for (auto &kv : e.smap) {
        PipelineEntry& pe = kv.second;
        vkDestroyShaderModule(e.device, pe.shader, nullptr);
        vkDestroyDescriptorSetLayout(e.device, pe.descriptor_layout, nullptr);
        vkDestroyPipelineLayout(e.device, pe.pipeline_layout, nullptr);
        vkDestroyPipeline(e.device, pe.pipeline, nullptr);
      }
    }
  }
  const char* type_key() const final {
    return "vulkan";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, fmt_)
        << "Can only save to customized format vulkan";
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

  // get a from primary context in device_id
  PipelineEntry GetPipeline(size_t device_id,
                            const std::string& func_name,
                            size_t num_pack_args) {
    vulkan::VulkanWorkspace* w = vulkan::VulkanWorkspace::Global().get();
    CHECK_LT(device_id, w->context_.size());
    // start lock scope.
    std::lock_guard<std::mutex> lock(mutex_);
    if (finfo_.size() <= device_id) {
      finfo_.resize(device_id + 1, DeviceEntry());
    }
    DeviceEntry& e = finfo_[device_id];
    auto it = e.smap.find(func_name);
    if (it != e.smap.end()) return it->second;
    PipelineEntry pe;
    if (e.device == nullptr) {
      e.device = w->context_[device_id].device;
    }
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
      VULKAN_CALL(vkCreateShaderModule(
          e.device, &shader_cinfo, nullptr, &(pe.shader)));
    }
    std::vector<VkDescriptorSetLayoutBinding> arg_binding;
    uint32_t num_pod = 0, num_buffer = 0;
    {
      auto fit = fmap_.find(func_name);
      CHECK(fit != fmap_.end());
      for (TVMType arg_type : fit->second.arg_types) {
        if (arg_type.code == kHandle) {
          VkDescriptorSetLayoutBinding bd;
          bd.binding = num_buffer;
          bd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
          bd.descriptorCount = 1;
        bd.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bd.pImmutableSamplers = nullptr;
        arg_binding.push_back(bd);
        ++num_buffer;
        } else {
          ++num_pod;
        }
      }
    }

    VkDescriptorSetLayoutCreateInfo descrip_cinfo;
    descrip_cinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descrip_cinfo.pNext = nullptr;
    descrip_cinfo.flags = 0;
    descrip_cinfo.bindingCount = arg_binding.size();
    descrip_cinfo.pBindings = arg_binding.data();
    VULKAN_CALL(vkCreateDescriptorSetLayout(
        e.device, &descrip_cinfo, nullptr, &(pe.descriptor_layout)));

    VkPushConstantRange crange;
    crange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    crange.offset = 0;
    crange.size = sizeof(ArgUnion) * num_pack_args;

    VkPipelineLayoutCreateInfo playout_cinfo;
    playout_cinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    playout_cinfo.pNext = nullptr;
    playout_cinfo.flags = 0;
    playout_cinfo.setLayoutCount = 1;
    playout_cinfo.pSetLayouts = &(pe.descriptor_layout);

    if (num_pack_args != 0) {
      playout_cinfo.pushConstantRangeCount = 1;
      playout_cinfo.pPushConstantRanges = &crange;
      CHECK_LE(crange.size,
               w->context_[device_id].phy_device_prop.limits.maxPushConstantsSize);
    } else {
      playout_cinfo.pushConstantRangeCount = 0;
      playout_cinfo.pPushConstantRanges = nullptr;
    }

    VULKAN_CALL(vkCreatePipelineLayout(
        e.device, &playout_cinfo, nullptr, &(pe.pipeline_layout)));
    VkComputePipelineCreateInfo pipeline_cinfo;
    pipeline_cinfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_cinfo.pNext = nullptr;
    pipeline_cinfo.flags = 0;
    pipeline_cinfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_cinfo.stage.pNext = nullptr;
    pipeline_cinfo.stage.flags = 0;
    pipeline_cinfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_cinfo.stage.module = pe.shader;
    pipeline_cinfo.stage.pName = func_name.c_str();
    pipeline_cinfo.stage.pSpecializationInfo = nullptr;
    pipeline_cinfo.layout = pe.pipeline_layout;
    pipeline_cinfo.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_cinfo.basePipelineIndex = 0;
    VULKAN_CALL(vkCreateComputePipelines(
        e.device, VK_NULL_HANDLE, 1, &pipeline_cinfo, nullptr, &(pe.pipeline)));
    e.smap[func_name] = pe;
    return pe;
  }

 private:
  // device specific entry
  struct DeviceEntry {
    VkDevice device{nullptr};
    std::unordered_map<std::string, PipelineEntry> smap;
  };
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
  // device local pipeline information.
  std::vector<DeviceEntry> finfo_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed func.
class VulkanWrappedFunc {
 public:
  // initialize the VULKAN function.
  void Init(VulkanModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            const std::string& func_name,
            size_t num_buffer_args,
            size_t num_pack_args,
            const std::vector<std::string>& thread_axis_tags) {
    w_ = vulkan::VulkanWorkspace::Global().get();
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    num_buffer_args_ = num_buffer_args;
    num_pack_args_ = num_pack_args;
    thread_axis_cfg_.Init(num_buffer_args + num_pack_args, thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  const ArgUnion* pack_args) const {
    vulkan::VulkanThreadEntry* tls = vulkan::VulkanThreadEntry::ThreadLocal();
    int device_id = tls->context.device_id;
    CHECK_LT(device_id, kVulkanMaxNumDevice);
    const vulkan::VulkanContext& vctx = w_->context_[device_id];
    VulkanModuleNode::PipelineEntry& pe = scache_[device_id];
    if (pe.pipeline == VK_NULL_HANDLE) {
      pe = m_->GetPipeline(device_id, func_name_, num_pack_args_);
    }
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    vulkan::VulkanCommandBuffer* cmd = tls->CommandPool(device_id)->Alloc(
        &(pe.descriptor_layout));

    cmd->write_descriptor_set.dstSet = cmd->descriptor_set;

    // setup descriptors
    for (uint32_t i = 0; i < num_buffer_args_; ++i) {
      void* buf = args[static_cast<int>(i)];
      VkDescriptorBufferInfo binfo;
      binfo.buffer = static_cast<vulkan::VulkanBuffer*>(buf)->buffer;
      binfo.offset = 0;
      binfo.range = VK_WHOLE_SIZE;
      cmd->write_descriptor_set.dstBinding = i;
      cmd->write_descriptor_set.pBufferInfo = &binfo;
      vkUpdateDescriptorSets(
          vctx.device, 1, &(cmd->write_descriptor_set), 0, nullptr);
    }

    // dispatch
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
    // 0: begin
    VULKAN_CALL(vkBeginCommandBuffer(cmd->cmd_buffer, &cb_begin));
    // 1: dispatch
    vkCmdBindPipeline(
        cmd->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pe.pipeline);
    vkCmdBindDescriptorSets(
        cmd->cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        pe.pipeline_layout, 0, 1, &(cmd->descriptor_set), 0, nullptr);
    // bind push constant if necessary
    if (num_pack_args_ != 0) {
      vkCmdPushConstants(
          cmd->cmd_buffer,
          pe.pipeline_layout,
          VK_SHADER_STAGE_COMPUTE_BIT,
          0, num_pack_args_ * sizeof(ArgUnion),
          pack_args);
    }
    vkCmdDispatch(
        cmd->cmd_buffer, wl.grid_dim(0), wl.grid_dim(1), wl.grid_dim(2));
    // 2: barrier(compute->compute|transfer)
    VkMemoryBarrier barrier_info;
    barrier_info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier_info.pNext = nullptr;
    barrier_info.srcAccessMask =
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    barrier_info.dstAccessMask =
        (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(
        cmd->cmd_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier_info, 0, nullptr, 0, nullptr);
    // 3: end
    VULKAN_CALL(vkEndCommandBuffer(cmd->cmd_buffer));
    // 4: submit with cmd->fence
    VULKAN_CALL(vkQueueSubmit(vctx.queue, 1, &cb_submit, cmd->fence));
  }

 private:
  // Reference to global workspace.
  vulkan::VulkanWorkspace* w_;
  // internal module
  VulkanModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // The name of the function.
  std::string func_name_;
  // Number of buffer arguments
  size_t num_buffer_args_;
  // number of packed arguments.
  size_t num_pack_args_;
  // Device state cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<VulkanModuleNode::PipelineEntry, kVulkanMaxNumDevice> scache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc VulkanModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
      << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  VulkanWrappedFunc f;
  size_t num_buffer_args = NumBufferArgs(info.arg_types);
  f.Init(this, sptr_to_self, name,
         num_buffer_args, info.arg_types.size() - num_buffer_args,
         info.thread_axis_tags);
  return PackFuncNonBufferArg(f, info.arg_types);
}

Module VulkanModuleCreate(
    std::unordered_map<std::string, VulkanShader> smap,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source) {
  vulkan::VulkanWorkspace::Global()->Init();
  std::shared_ptr<VulkanModuleNode> n =
      std::make_shared<VulkanModuleNode>(smap, fmap, source);
  return Module(n);
}

// Load module from module.
Module VulkanModuleLoadFile(const std::string& file_name,
                            const std::string& format) {
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
  CHECK_EQ(magic, kVulkanModuleMagic)
      << "VulkanModule Magic mismatch";
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

TVM_REGISTER_GLOBAL("module.loadfile_vulkan")
.set_body_typed(VulkanModuleLoadFile);

TVM_REGISTER_GLOBAL("module.loadbinary_vulkan")
.set_body_typed(VulkanModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
