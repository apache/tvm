/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_module.cc
 */
#include "./opencl_common.h"
#include "./opencl_module.h"

#if TVM_OPENCL_RUNTIME

#include <vector>
#include <string>
#include <unordered_map>
#include "../void_addr_args.h"
#include "../thread_axis_args.h"

namespace tvm {
namespace runtime {

using namespace detail;

/*!
 * \brief Internal data structure to support multi-gpu execution.
 *  Try to use OpenCL runtime's primary context.
 */
class OpenCLModule::Internal {
 public:
  // the binary data
  cl_program program;
  // kernel id cache
  std::unordered_map<std::string, size_t> kid_map;

  explicit Internal(cl_program program)
      : program(program) {
  }
  // destructor
  ~Internal() {
    OPENCL_CALL(clReleaseProgram(program));
  }
  // get kernel id given key(function name.
  size_t GetKernelID(const std::string& key) {
    cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
    std::lock_guard<std::mutex> lock(w->mu);
    if (kid_map.count(key)) return kid_map.at(key);
    size_t kid = w->num_registered_kernels++;
    kid_map[key] = kid;
    return kid;
  }
};

class OpenCLWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(std::shared_ptr<OpenCLModule::Internal> m,
            size_t kernel_id,
            std::string func_name,
            std::vector<size_t> arg_size,
            const std::vector<std::string>& thread_axis_tags)  {
    m_ = m;
    kernel_id_ = kernel_id;
    func_name_ = func_name;
    arg_size_ = arg_size;
    thread_axis_cfg_.Init(arg_size.size(), thread_axis_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args,
                  TVMRetValue* rv,
                  void** void_args) const {
    cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
    cl::OpenCLThreadEntry* t = cl::OpenCLThreadEntry::ThreadLocal();
    CHECK(w->initialized());
    // get the kernel from thread local kernel table.
    if (kernel_id_ >= t->kernel_table.size()) {
      t->kernel_table.resize(kernel_id_ + 1, nullptr);
    }
    cl_kernel kernel = t->kernel_table[kernel_id_];
    if (kernel == nullptr) {
      cl_int err;
      kernel = clCreateKernel(m_->program, func_name_.c_str(), &err);
      OPENCL_CHECK_ERROR(err);
      t->kernel_table[kernel_id_] = kernel;
    }
    // setup arguments.
    for (cl_uint i = 0; i < arg_size_.size(); ++i) {
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], void_args[i]));
    }
    cl_command_queue queue = w->GetQueue(t->context);
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    cl_uint work_dim = static_cast<cl_uint>(thread_axis_cfg_.work_dim());
    for (cl_uint i = 0; i < work_dim; ++i) {
      wl.work_size[i + 3] *= wl.work_size[i];
    }
    // launch kernel
    OPENCL_CALL(clEnqueueNDRangeKernel(
        queue, kernel, work_dim, nullptr,
        wl.work_size + 3,
        wl.work_size,
        0, nullptr, nullptr));
  }

 private:
  // modulex
  std::shared_ptr<OpenCLModule::Internal> m_;
  // global kernel id in the kernel table.
  size_t kernel_id_;
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // thread axis config
  ThreadAxisConfig thread_axis_cfg_;
};

PackedFunc OpenCLModule::GetPackedFunc(
    const std::string& func_name,
    const std::vector<TVMType> arg_types,
    const std::vector<std::string> thread_axis_tags) const {
  OpenCLWrappedFunc f;
  // get the kernel id.
  size_t kid = ptr_->GetKernelID(func_name);
  std::vector<size_t> arg_size(arg_types.size());
  for (size_t i = 0; i < arg_types.size(); ++i) {
    TVMType t = arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    uint32_t bits = t.bits;
    CHECK_EQ(bits % 8, 0U);
    arg_size[i] = bits / 8;
  }
  // initialize the wrapped func.
  f.Init(ptr_, kid, func_name, arg_size, thread_axis_tags);
  return PackFromVoidAddrArgs(f, arg_types);
}

OpenCLModule OpenCLModule::CreateWithSource(std::string source) {
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  CHECK(w->initialized());
  const char* s = source.c_str();
  size_t len = source.length();
  cl_int err;
  cl_program prog = clCreateProgramWithSource(
      w->context, 1, &s, &len, &err);
  OPENCL_CHECK_ERROR(err);

  for (cl_device_id dev_id : w->devices) {
    err = clBuildProgram(prog, 1, &dev_id, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t len;
      std::string log;
      clGetProgramBuildInfo(
          prog, dev_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
      log.resize(len);
      clGetProgramBuildInfo(
          prog, dev_id, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
      LOG(FATAL) << "OpenCL build error for device=" << dev_id << log;
    }
  }
  OpenCLModule m;
  m.ptr_ = std::make_shared<Internal>(prog);
  return m;
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENCL_RUNTIME
