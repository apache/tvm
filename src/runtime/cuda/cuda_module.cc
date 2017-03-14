/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_module.cc
 */
#include "./cuda_module.h"

#if TVM_CUDA_RUNTIME

#include <tvm/runtime/registry.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include "./cuda_common.h"
#include "../void_addr_args.h"
#include "../thread_storage_scope.h"
#include "../meta_data.h"
#include "../file_util.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class CUDAModuleNode : public runtime::ModuleNode {
 public:
  explicit CUDAModuleNode(std::string data,
                          std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // destructor
  ~CUDAModuleNode() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        CUDA_CALL(cudaSetDevice(static_cast<int>(i)));
        CUDA_DRIVER_CALL(cuModuleUnload(module_[i]));
      }
    }
  }

  const char* type_key() const final {
    return "cuda";
  }

  void PreCompile(const std::string& name, TVMContext ctx) final {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaFree(nullptr);
    this->GetFunc(ctx.device_id, name);
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    CHECK_EQ(fmt, fmt_)
        << "Can only save to format=" << fmt_;
    std::string meta_file = GetMetaFilePath(file_name);
    SaveMetaDataToFile(meta_file, fmap_);
    SaveBinaryToFile(file_name, data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    } else {
      if (fmt_ == "ptx") return data_;
      return "";
    }
  }

  // get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL)
          << "CUDAError: cuModuleGetFunction " << func_name
          << " failed with error: " << msg;
    }
    return func;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  std::string cuda_source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed fucn.
class CUDAWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(CUDAModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            const std::string& func_name,
            size_t num_void_args,
            const std::vector<std::string>& thread_axis_tags) {
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    thread_axis_cfg_.Init(num_void_args, thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    CUDA_DRIVER_CALL(cuLaunchKernel(
        fcache_[device_id],
        wl.grid_dim(0),
        wl.grid_dim(1),
        wl.grid_dim(2),
        wl.block_dim(0),
        wl.block_dim(1),
        wl.block_dim(2),
        0, nullptr, void_args, 0));
  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  std::shared_ptr<ModuleNode> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUfunction, kMaxNumGPUs> fcache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

void AutoSetCUDADevice(const TVMArgs& args, TVMRetValue* rv) {
  CHECK_EQ(args.size(), 3);
  TVMValue* values = static_cast<TVMValue*>(args[0].operator void*());
  int* type_codes = static_cast<int*>(args[1].operator void*());
  int num_args = args[2].operator int();

  int device_id = -1;
  for (int i = 0; i < num_args; ++i) {
    if (type_codes[i] == kArrayHandle) {
      TVMContext ctx = static_cast<TVMArray*>(values[i].v_handle)->ctx;
      CHECK_EQ(ctx.device_type, kGPU)
          << "All operands need to be GPU";
      if (device_id == -1) {
        device_id = ctx.device_id;
      } else {
        CHECK_EQ(device_id, ctx.device_id)
            << "Operands comes from different devices ";
      }
    }
  }
  CHECK_NE(device_id, -1)
      << "Cannot detect device id from list";
  CUDA_CALL(cudaSetDevice(device_id));
}

PackedFunc CUDAModuleNode::GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
      << "Device function do not have main";
  if (name == symbol::tvm_entry_setdevice) {
    return PackedFunc(AutoSetCUDADevice);
  }
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  CUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.thread_axis_tags);
  return PackFromVoidAddrArgs(f, info.arg_types);
}

Module CUDAModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string cuda_source) {
  std::shared_ptr<CUDAModuleNode> n =
      std::make_shared<CUDAModuleNode>(data, fmt, fmap, cuda_source);
  return Module(n);
}

// Load module from module.
Module CUDAModuleLoad(const std::string& file_name,
                      const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL(_module_loadfile_cubin)
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = CUDAModuleLoad(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL(_module_loadfile_ptx)
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = CUDAModuleLoad(args[0], args[1]);
  });
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_CUDA_RUNTIME
