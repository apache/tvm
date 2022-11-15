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
 * \file opencl_module.cc
 */
#include "opencl_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../source_utils.h"
#include "opencl_common.h"

namespace tvm {
namespace runtime {

class OpenCLWrappedFunc {
 public:
  // initialize the OpenCL function.
  void Init(OpenCLModuleNode* m, ObjectPtr<Object> sptr, OpenCLModuleNode::KTRefEntry entry,
            std::string func_name, std::vector<size_t> arg_size,
            const std::vector<std::string>& launch_param_tags) {
    w_ = m->GetGlobalWorkspace();
    m_ = m;
    sptr_ = sptr;
    entry_ = entry;
    func_name_ = func_name;
    arg_size_ = arg_size;
    launch_param_config_.Init(arg_size.size(), launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    ICHECK(w_->context != nullptr) << "No OpenCL device";
    cl::OpenCLThreadEntry* t = w_->GetThreadEntry();
    // get the kernel from thread local kernel table.
    if (entry_.kernel_id >= t->kernel_table.size()) {
      t->kernel_table.resize(entry_.kernel_id + 1);
    }
    const auto& e = t->kernel_table[entry_.kernel_id];
    cl_kernel kernel = e.kernel;
    if (kernel == nullptr || e.version != entry_.version) {
      kernel = m_->InstallKernel(w_, t, func_name_, entry_);
    }
    // setup arguments.
    for (cl_uint i = 0; i < arg_size_.size(); ++i) {
      void* arg = nullptr;
      if (args.type_codes[i] == DLDataTypeCode::kDLOpaqueHandle) {
        arg = static_cast<cl::BufferDescriptor*>(void_args[i])->buffer;
      } else {
        arg = void_args[i];
      }
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], arg));
    }
    cl_command_queue queue = w_->GetQueue(t->device);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    cl_uint work_dim = static_cast<cl_uint>(launch_param_config_.work_dim());
    for (cl_uint i = 0; i < work_dim; ++i) {
      wl.work_size[i] *= wl.work_size[i + 3];
    }
    // launch kernel

    if (w_->IsProfiling(t->device)) {
      w_->GetEventQueue(t->device).resize(w_->GetEventQueue(t->device).size() + 1);
      OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, work_dim, nullptr, wl.work_size,
                                         wl.work_size + 3, 0, nullptr,
                                         &(w_->GetEventQueue(t->device).back())));
    } else {
      OPENCL_CALL(clEnqueueNDRangeKernel(queue, kernel, work_dim, nullptr, wl.work_size,
                                         wl.work_size + 3, 0, nullptr, nullptr));
    }
  }

 private:
  // global workspace.
  cl::OpenCLWorkspace* w_;
  // The module
  OpenCLModuleNode* m_;
  // resource handle
  ObjectPtr<Object> sptr_;
  // global kernel id in the kernel table.
  OpenCLModuleNode::KTRefEntry entry_;
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // launch parameters config
  LaunchParamConfig launch_param_config_;
};

OpenCLModuleNode::~OpenCLModuleNode() {
  {
    // free the kernel ids in global table.
    std::lock_guard<std::mutex> lock(workspace_->mu);
    for (auto& kv : kid_map_) {
      workspace_->free_kernel_ids.push_back(kv.second.kernel_id);
    }
  }
  // free the kernels
  for (cl_kernel k : kernels_) {
    OPENCL_CALL(clReleaseKernel(k));
  }
  // free the programs
  for (auto& kv : programs_) {
    for (auto& program : kv.second) {
      if (program) {
        OPENCL_CALL(clReleaseProgram(program));
      }
    }
  }
}

cl::OpenCLWorkspace* OpenCLModuleNode::GetGlobalWorkspace() {
  return cl::OpenCLWorkspace::Global();
}

PackedFunc OpenCLModuleNode::GetFunction(const std::string& name,
                                         const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  OpenCLWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    DLDataType t = info.arg_types[i];
    ICHECK_EQ(t.lanes, 1U);
    if (t.code == kTVMOpaqueHandle) {
      // specially store pointer type size in OpenCL driver
      arg_size[i] = sizeof(void*);
    } else {
      uint32_t bits = t.bits;
      ICHECK_EQ(bits % 8, 0U);
      arg_size[i] = bits / 8;
    }
  }
  // initialize the wrapped func.
  f.Init(this, sptr_to_self, kid_map_.at(name), name, arg_size, info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

void OpenCLModuleNode::SaveToFile(const std::string& file_name, const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
  std::string meta_file = GetMetaFilePath(file_name);
  SaveMetaDataToFile(meta_file, fmap_);
  SaveBinaryToFile(file_name, data_);
}

void OpenCLModuleNode::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(fmt_);
  stream->Write(fmap_);
  stream->Write(data_);
}

std::string OpenCLModuleNode::GetSource(const std::string& format) {
  if (format == fmt_) return data_;
  if (fmt_ == "cl") {
    return data_;
  } else {
    return source_;
  }
}

void OpenCLModuleNode::Init() {
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

  // split into source artifacts for each kernel
  parsed_kernels_ = SplitKernels(GetSource("cl"));
  ICHECK(!parsed_kernels_.empty()) << "The OpenCL module expects a kernel delimited "
                                   << "source from code generation, but no kernel "
                                   << "delimiter was found.";
  ICHECK_EQ(fmap_.size(), parsed_kernels_.size())
      << "The number of parsed kernel sources does not match the number of kernel functions";
  // zero initialize cl_program pointers for each device kernel
  for (auto& kv : parsed_kernels_) {
    programs_.insert({kv.first, std::vector<cl_program>(workspace_->devices.size(), nullptr)});
  }
}

cl_kernel OpenCLModuleNode::InstallKernel(cl::OpenCLWorkspace* w, cl::OpenCLThreadEntry* t,
                                          const std::string& func_name, const KTRefEntry& e) {
  std::lock_guard<std::mutex> lock(build_lock_);
  int device_id = t->device.device_id;
  if (programs_[func_name][device_id] == nullptr) {
    // create program
    if (fmt_ == "cl") {
      const char* s = parsed_kernels_[func_name].c_str();
      size_t len = parsed_kernels_[func_name].length();
      cl_int err;
      programs_[func_name][device_id] = clCreateProgramWithSource(w->context, 1, &s, &len, &err);
      OPENCL_CHECK_ERROR(err);
    } else if (fmt_ == "xclbin" || fmt_ == "awsxclbin" || fmt_ == "aocx") {
      const unsigned char* s = (const unsigned char*)data_.c_str();
      size_t len = data_.length();
      cl_int err;
      cl_device_id dev = w->devices[device_id];
      programs_[func_name][device_id] =
          clCreateProgramWithBinary(w->context, 1, &dev, &len, &s, nullptr, &err);
      OPENCL_CHECK_ERROR(err);
    } else {
      LOG(FATAL) << "Unknown OpenCL format " << fmt_;
    }
    // build program
    cl_int err;
    cl_device_id dev = w->devices[device_id];
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

Module OpenCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  auto n = make_object<OpenCLModuleNode>(data, fmt, fmap, source);
  n->Init();
  return Module(n);
}

// Load module from module.
Module OpenCLModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return OpenCLModuleCreate(data, fmt, fmap, std::string());
}

Module OpenCLModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return OpenCLModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_cl").set_body_typed(OpenCLModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadfile_clbin").set_body_typed(OpenCLModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_opencl").set_body_typed(OpenCLModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm
