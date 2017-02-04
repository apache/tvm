/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_module.cc
 */
#include "./cuda_module.h"

#if TVM_CUDA_RUNTIME
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include "./cuda_common.h"
#include "../void_addr_args.h"
#include "../thread_storage_scope.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Internal data structure to support multi-gpu execution.
 *  Try to use CUDA runtime's primary context.
 */
class CUDAModule::Internal {
 public:
  explicit Internal(std::string data)
      : data_(data) {
    std::fill(module_.begin(), module_.end(), nullptr);
  }
  // get a CUfunction from primary context in dev_id
  CUfunction GetFunc(int dev_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[dev_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[dev_id]), data_.c_str()));
    }
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_[dev_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char *msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL)
          << "CUDAError: cuModuleGetFunction " << func_name
          << " failed with error: " << msg;
    }
    return func;
  }
  // destructor
  ~Internal() {
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        CUDA_CALL(cudaSetDevice(i));
        CUDA_DRIVER_CALL(cuModuleUnload(module_[i]));
      }
    }
  }

 private:
  // the binary data
  std::string data_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<CUmodule, CUDAModule::kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

// a wrapped function class to get packed fucn.
class CUDAWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(std::shared_ptr<CUDAModule::Internal> m,
            const std::string& func_name,
            size_t num_void_args,
            const std::vector<std::string>& thread_axis_tags)  {
    m_ = m;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    thread_axis_cfg_.Init(num_void_args, thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    int dev_id;
    CUDA_CALL(cudaGetDevice(&dev_id));
    if (fcache_[dev_id] == nullptr) {
      fcache_[dev_id] = m_->GetFunc(dev_id, func_name_);
    }
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    CUDA_DRIVER_CALL(cuLaunchKernel(
        fcache_[dev_id],
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
  std::shared_ptr<CUDAModule::Internal> m_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUfunction, CUDAModule::kMaxNumGPUs> fcache_;
  // thread axis configuration
  ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc CUDAModule::GetPackedFunc(
    const std::string& func_name,
    const std::vector<TVMType> arg_types,
    const std::vector<std::string> thread_axis_tags) const {
  CUDAWrappedFunc f;
  f.Init(ptr_, func_name, arg_types.size(), thread_axis_tags);
  return PackFromVoidAddrArgs(f, arg_types);
}

CUDAModule CUDAModule::Create(std::string ptx) {
  // call a runtime API to make sure the context is created.
  CUDAModule m;
  m.ptr_ = std::make_shared<Internal>(ptx);
  return m;
}
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_CUDA_RUNTIME
