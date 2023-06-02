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

#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../source_utils.h"
#include "../spirv/spirv_shader.h"
#include "opencl_common.h"
#include "opencl_module.h"

namespace tvm {
namespace runtime {

class OpenCLSPIRVModuleNode : public OpenCLModuleNodeBase {
 public:
  explicit OpenCLSPIRVModuleNode(const std::unordered_map<std::string, SPIRVShader>& shaders,
                                 const std::string& spirv_text,
                                 std::unordered_map<std::string, FunctionInfo> fmap)
      : OpenCLModuleNodeBase(fmap), shaders_(shaders), spirv_text_(spirv_text) {}

  void SaveToFile(const String& file_name, const String& format) final;
  void SaveToBinary(dmlc::Stream* stream) final;
  String GetSource(const String&) final { return spirv_text_; }

  void Init() override;
  cl_kernel InstallKernel(cl::OpenCLWorkspace* w, cl::OpenCLThreadEntry* t,
                          const std::string& func_name, const KTRefEntry& e) override;

 private:
  std::unordered_map<std::string, SPIRVShader> shaders_;
  std::string spirv_text_;
};

void OpenCLSPIRVModuleNode::SaveToFile(const String& file_name, const String& format) {
  // TODO(masahi): How SPIRV binaries should be save to a file?
  LOG(FATAL) << "Not implemented.";
}

void OpenCLSPIRVModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmap_);
  stream->Write(shaders_);
}

void OpenCLSPIRVModuleNode::Init() {
  workspace_ = GetGlobalWorkspace();
  workspace_->Init();
  // initialize the kernel id, need to lock global table.
  std::lock_guard<std::mutex> lock(workspace_->mu);
  for (const auto& kv : fmap_) {
    const std::string& key = kv.first;
    KTRefEntry e;
    if (workspace_->free_kernel_ids.size() != 0) {
      e.kernel_id = workspace_->free_kernel_ids.back();
      workspace_->free_kernel_ids.pop_back();
    } else {
      e.kernel_id = workspace_->num_registered_kernels++;
    }
    e.version = workspace_->timestamp++;
    kid_map_[key] = e;
  }

  // zero initialize cl_program pointers for each device kernel
  for (auto& kv : shaders_) {
    programs_.insert({kv.first, std::vector<cl_program>(workspace_->devices.size(), nullptr)});
  }
}

cl_kernel OpenCLSPIRVModuleNode::InstallKernel(cl::OpenCLWorkspace* w, cl::OpenCLThreadEntry* t,
                                               const std::string& func_name, const KTRefEntry& e) {
  std::lock_guard<std::mutex> lock(build_lock_);
  int device_id = t->device.device_id;
  if (programs_[func_name][device_id] == nullptr) {
    auto it = shaders_.find(func_name);
    const unsigned char* s = (const unsigned char*)it->second.data.data();
    size_t len = it->second.data.size() * sizeof(uint32_t);
    cl_int err;
    cl_device_id dev = w->devices[device_id];
    auto platform = w->device_to_platform[dev];
    programs_[func_name][device_id] =
        clCreateProgramWithBinary(w->contexts[platform], 1, &dev, &len, &s, nullptr, &err);
    OPENCL_CHECK_ERROR(err);

    // build program
    err = clBuildProgram(programs_[func_name][device_id], 1, &dev, nullptr, nullptr, nullptr);

    if (err != CL_SUCCESS) {
      size_t len;
      std::string log;
      clGetProgramBuildInfo(programs_[func_name][device_id], dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &len);
      log.resize(len);
      clGetProgramBuildInfo(programs_[func_name][device_id], dev, CL_PROGRAM_BUILD_LOG, len,
                            &log[0], nullptr);
      LOG(FATAL) << "OpenCL build error for device=" << dev << "\n" << log;
    }
  }
  // build kernel
  cl_int err;
  cl_kernel kernel = clCreateKernel(programs_[func_name][device_id], func_name.c_str(), &err);
  OPENCL_CHECK_ERROR(err);
  t->kernel_table[e.kernel_id].kernel = kernel;
  t->kernel_table[e.kernel_id].version = e.version;
  kernels_.push_back(kernel);
  return kernel;
}

Module OpenCLModuleCreate(const std::unordered_map<std::string, SPIRVShader>& shaders,
                          const std::string& spirv_text,
                          std::unordered_map<std::string, FunctionInfo> fmap) {
  auto n = make_object<OpenCLSPIRVModuleNode>(shaders, spirv_text, fmap);
  n->Init();
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm
