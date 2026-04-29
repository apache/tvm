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
 * \brief Plugin-only OpenCL runtime module.  Built only when
 *        USE_OPENCL=ON.  No exported header — codegen-side construction
 *        goes through src/target/opencl/opencl_fallback_module.h:OpenCLModuleCreateWithFallback,
 *        which dispatches to "ffi.Module.create.opencl" registered
 *        below when this file is linked into the build.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/support/io.h>

#include <string>
#include <utility>
#include <vector>

#include "../../support/bytes_io.h"
#include "opencl_common.h"
#include "source_utils.h"

namespace tvm {
namespace runtime {

class OpenCLWrappedFunc {
 public:
  // initialize the OpenCL function.
  void Init(OpenCLModuleNodeBase* m, ObjectPtr<Object> sptr, OpenCLModuleNode::KTRefEntry entry,
            std::string func_name, std::vector<size_t> arg_size,
            const ffi::Array<ffi::String>& launch_param_tags) {
    w_ = m->GetGlobalWorkspace();
    m_ = m;
    sptr_ = sptr;
    entry_ = entry;
    func_name_ = func_name;
    arg_size_ = arg_size;
    launch_param_config_.Init(arg_size.size(), launch_param_tags);
  }

  // invoke the function with void arguments
  void operator()(ffi::PackedArgs args, ffi::Any* rv, void** void_args) const {
    TVM_FFI_ICHECK(w_->devices.size() > 0) << "No OpenCL device";
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
    ThreadWorkLoad wl = launch_param_config_.Extract(args);
    cl_uint work_dim = static_cast<cl_uint>(launch_param_config_.work_dim());
    // setup arguments.
    for (cl_uint i = 0; i < arg_size_.size(); ++i) {
      void* arg = nullptr;
      if (args[i].as<void*>()) {
        arg = static_cast<cl::BufferDescriptor*>(void_args[i])->buffer;
      } else {
        arg = void_args[i];
      }
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], arg));
    }
    cl_command_queue queue = w_->GetQueue(t->device);
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
  OpenCLModuleNodeBase* m_;
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

OpenCLModuleNodeBase::~OpenCLModuleNodeBase() {
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

cl::OpenCLWorkspace* OpenCLModuleNodeBase::GetGlobalWorkspace() {
  return cl::OpenCLWorkspace::Global();
}

ffi::Optional<ffi::Function> OpenCLModuleNodeBase::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  auto opt_info = fmap_.Get(name);
  if (!opt_info.has_value()) return std::nullopt;
  FunctionInfo info = opt_info.value();
  OpenCLWrappedFunc f;
  std::vector<size_t> arg_size(info->arg_types.size());
  for (size_t i = 0; i < info->arg_types.size(); ++i) {
    DLDataType t = info->arg_types[i];
    TVM_FFI_ICHECK_EQ(t.lanes, 1U);
    if (t.code == kDLOpaqueHandle) {
      // specially store pointer type size in OpenCL driver
      arg_size[i] = sizeof(void*);
    } else {
      uint32_t bits = t.bits;
      TVM_FFI_ICHECK_EQ(bits % 8, 0U);
      arg_size[i] = bits / 8;
    }
  }
  // initialize the wrapped func.
  f.Init(this, sptr_to_self, kid_map_.at(name), name, arg_size, info->launch_param_tags);
  return PackFuncVoidAddr(f, info->arg_types);
}

ffi::Bytes OpenCLModuleNode::SaveToBytes() const {
  // NOTE: serialization format MUST remain byte-identical to
  // target::OpenCLFallbackModuleNode::SaveToBytes in
  // src/target/opencl/opencl_fallback_module.cc.  This file is the
  // source of truth; the fallback follows.
  // 3 fields only — the source map is in-memory inspection material
  // and is NEVER serialized (matches upstream behavior for all
  // backends).
  std::string result;
  support::BytesOutStream stream(&result);
  stream.Write(fmt_);
  stream.Write(fmap_);
  stream.Write(code_);
  return ffi::Bytes(std::move(result));
}

ffi::String OpenCLModuleNode::InspectSource(const ffi::String& format) const {
  if (auto it = source_.find(format); it != source_.end()) {
    return (*it).second;
  }
  if (format.empty()) {
    // Default: aggregated OpenCL C source dump (key "cl").
    if (auto it = source_.find("cl"); it != source_.end()) {
      return (*it).second;
    }
    if (fmt_ == "cl") {
      return ffi::String(code_.data(), code_.size());
    }
  }
  return ffi::String();
}

void OpenCLModuleNode::Init() {
  workspace_ = GetGlobalWorkspace();
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

  // split into source artifacts for each kernel.  For fmt=="cl" the
  // code_ bytes are the OpenCL C source; for the binary formats
  // (xclbin/awsxclbin/aocx) parsing is skipped — the binary blob is
  // passed directly to clCreateProgramWithBinary in InstallKernel.
  if (fmt_ == "cl") {
    parsed_kernels_ = SplitKernels(std::string(code_.data(), code_.size()));
    TVM_FFI_ICHECK(!parsed_kernels_.empty()) << "The OpenCL module expects a kernel delimited "
                                             << "source from code generation, but no kernel "
                                             << "delimiter was found.";
    TVM_FFI_ICHECK_EQ(fmap_.size(), parsed_kernels_.size())
        << "The number of parsed kernel sources does not match the number of kernel functions";
  }
}

bool OpenCLModuleNode::IsProgramCreated(const std::string& func_name, int device_id) {
  auto size = programs_[func_name].size();
  if (size > 0 && programs_[func_name][device_id] != nullptr) return true;
  auto dev_size = GetGlobalWorkspace()->devices.size();
  TVM_FFI_ICHECK(device_id < static_cast<int>(dev_size))
      << "Device id " << device_id << " is bigger than number of available devices";
  // zero initialize cl_program pointers for each device kernel
  if (size == 0) programs_[func_name].resize(dev_size, nullptr);
  return false;
}

cl_kernel OpenCLModuleNode::InstallKernel(cl::OpenCLWorkspace* w, cl::OpenCLThreadEntry* t,
                                          const std::string& func_name, const KTRefEntry& e) {
  std::lock_guard<std::mutex> lock(build_lock_);
  int device_id = t->device.device_id;
  auto did = w->GetCLDeviceID(device_id);
  auto platform = w->device_info[did].platform_id;
  if (!IsProgramCreated(func_name, device_id)) {
    // create program
    if (fmt_ == "cl") {
      const char* s = parsed_kernels_[func_name].c_str();
      size_t len = parsed_kernels_[func_name].length();
      cl_int err;
      programs_[func_name][device_id] =
          clCreateProgramWithSource(w->contexts[platform], 1, &s, &len, &err);
      OPENCL_CHECK_ERROR(err);
    } else if (fmt_ == "xclbin" || fmt_ == "awsxclbin" || fmt_ == "aocx") {
      const unsigned char* s = reinterpret_cast<const unsigned char*>(code_.data());
      size_t len = code_.size();
      cl_int err;
      cl_device_id dev = w->devices[device_id];
      programs_[func_name][device_id] =
          clCreateProgramWithBinary(w->contexts[platform], 1, &dev, &len, &s, nullptr, &err);
      OPENCL_CHECK_ERROR(err);
    } else {
      TVM_FFI_THROW(InternalError) << "Unknown OpenCL format " << fmt_;
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
      TVM_FFI_THROW(InternalError) << "OpenCL build error for device=" << dev
                                   << "\nError: " << cl::CLGetErrorString(err) << "\n"
                                   << log;
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

void OpenCLModuleNode::SetPreCompiledPrograms(const std::string& bytes) {
  workspace_->Init();
  support::BytesInStream strm(bytes);
  uint64_t kernels_num;
  strm.Read(&kernels_num);
  cl::OpenCLThreadEntry* t = workspace_->GetThreadEntry();
  int device_id = t->device.device_id;
  for (size_t i = 0; i < kernels_num; ++i) {
    std::string name;
    std::vector<unsigned char> bin_vector;
    strm.Read(&name);
    strm.Read(&bin_vector);
    if (!IsProgramCreated(name, device_id)) {
      cl_int err = 0;
      cl_int binaryStatus;
      size_t binarySize = bin_vector.size();
      const unsigned char* programBinary = bin_vector.data();

      cl_device_id dev = workspace_->GetCLDeviceID(device_id);
      auto platform = workspace_->device_info[dev].platform_id;
      programs_[name][device_id] =
          clCreateProgramWithBinary(workspace_->contexts[platform], 1, &dev, &binarySize,
                                    &programBinary, &binaryStatus, &err);
      OPENCL_CHECK_ERROR(err);
      OPENCL_CHECK_ERROR(binaryStatus);

      err = clBuildProgram(programs_[name][device_id], 0, nullptr, nullptr, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        size_t len;
        std::string log;
        clGetProgramBuildInfo(programs_[name][device_id], dev, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                              &len);
        log.resize(len);
        clGetProgramBuildInfo(programs_[name][device_id], dev, CL_PROGRAM_BUILD_LOG, len, &log[0],
                              nullptr);
        TVM_FFI_THROW(InternalError) << "OpenCL build error for device=" << dev << "\n" << log;
      }
    }
  }
}

std::string OpenCLModuleNode::GetPreCompiledPrograms() {
  workspace_->Init();
  std::string result;
  support::BytesOutStream strm(&result);
  strm.Write(static_cast<uint64_t>(parsed_kernels_.size()));
  for (auto& it : parsed_kernels_) {
    std::string name = it.first;
    cl::OpenCLThreadEntry* t = workspace_->GetThreadEntry();
    int device_id = t->device.device_id;
    t->kernel_table.resize(workspace_->num_registered_kernels);
    if (!IsProgramCreated(name, device_id)) {
      InstallKernel(workspace_, t, name, kid_map_[name]);
    }
    size_t size;
    clGetProgramInfo(programs_[name][device_id], CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size,
                     nullptr);
    TVM_FFI_ICHECK(size > 0) << "Size of binary is 0";
    std::vector<unsigned char> bin_vector(size);
    unsigned char* binary = bin_vector.data();
    clGetProgramInfo(programs_[name][device_id], CL_PROGRAM_BINARIES, sizeof(unsigned char*),
                     &binary, nullptr);

    strm.Write(name);
    strm.Write(bin_vector);
  }
  return result;
}

ffi::Optional<ffi::Function> OpenCLModuleNode::GetFunction(const ffi::String& name) {
  ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);
  TVM_FFI_ICHECK_EQ(sptr_to_self.get(), this);
  if (name == "opencl.GetPreCompiledPrograms") {
    return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
      *rv = this->GetPreCompiledPrograms();
    });
  } else if (name == "opencl.SetPreCompiledPrograms") {
    return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
      this->SetPreCompiledPrograms(args[0].cast<std::string>());
    });
  }
  return OpenCLModuleNodeBase::GetFunction(name);
}

static ffi::Module OpenCLModuleCreateImpl(ffi::Bytes code, ffi::String fmt,
                                          ffi::Map<ffi::String, FunctionInfo> fmap,
                                          ffi::Map<ffi::String, ffi::String> source) {
  auto n = ffi::make_object<OpenCLModuleNode>(std::move(code), std::move(fmt), std::move(fmap),
                                              std::move(source));
  n->Init();
  return ffi::Module(n);
}

ffi::Module OpenCLModuleLoadFromBytes(const ffi::Bytes& bytes) {
  support::BytesInStream stream(bytes);
  ffi::String fmt;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Bytes code;
  stream.Read(&fmt);
  TVM_FFI_ICHECK(stream.Read(&fmap));
  stream.Read(&code);
  // Source map is not serialized — reconstructed empty on load.
  return OpenCLModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                                ffi::Map<ffi::String, ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Registry: "ffi.Module.create.opencl" — codegen-time OpenCL module factory.
  // Used by src/target/opencl/opencl_fallback_module.h:OpenCLModuleCreateWithFallback.
  // Registry: "ffi.Module.load_from_bytes.opencl" — disk loader.  Only this
  // (real) module registers a loader; the fallback is codegen-only.
  refl::GlobalDef()
      .def("ffi.Module.load_from_bytes.opencl", OpenCLModuleLoadFromBytes)
      .def("ffi.Module.create.opencl",
           [](ffi::Bytes code, ffi::String fmt, ffi::Map<ffi::String, FunctionInfo> fmap,
              ffi::Map<ffi::String, ffi::String> source) {
             return OpenCLModuleCreateImpl(std::move(code), std::move(fmt), std::move(fmap),
                                           std::move(source));
           });
}
}  // namespace runtime
}  // namespace tvm
