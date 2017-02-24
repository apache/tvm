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
#include "../thread_storage_scope.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-device execution.
// OpenCL runtime is a bit tricky because clSetKernelArg is not thread-safe
// To make the call thread-safe, we create a thread-local kernel table
// and lazily install new kernels into the kernel table when the kernel is called.
// The kernels are recycled when the module get destructed.
class OpenCLModuleNode : public ModuleNode {
 public:
  // Kernel table reference entry.
  struct KTRefEntry {
    size_t kernel_id;
    size_t version;
  };
  explicit OpenCLModuleNode(std::string data,
                            std::string fmt,
                            std::unordered_map<std::string, FunctionInfo> fmap)
      : data_(data), fmt_(fmt), fmap_(fmap) {}
  // destructor
  ~OpenCLModuleNode() {
    {
      // free the kernel ids in global table.
      cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
      std::lock_guard<std::mutex> lock(w->mu);
      for (auto& kv : kid_map_) {
        w->free_kernel_ids.push_back(kv.second.kernel_id);
      }
    }
    // free the kernels
    for (cl_kernel k : kernels_) {
      OPENCL_CALL(clReleaseKernel(k));
    }
    if (program_) {
      OPENCL_CALL(clReleaseProgram(program_));
    }
  }

  const char* type_key() const final {
    return "opencl";
  }

  void PreCompile(const std::string& name, TVMContext ctx) final {
    InstallKernel(cl::OpenCLWorkspace::Global(),
                  cl::OpenCLThreadEntry::ThreadLocal(),
                  name, kid_map_.at(name));
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    LOG(FATAL) << "Not implemented";
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (fmt_ == "cl") {
      return data_;
    } else {
      return "";
    }
  }

  // Initialize the programs
  void InitProgram() {
    cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
    CHECK(w->initialized());
    if (fmt_ == "cl") {
      const char* s = data_.c_str();
      size_t len = data_.length();
      cl_int err;
      program_ = clCreateProgramWithSource(
          w->context, 1, &s, &len, &err);
      OPENCL_CHECK_ERROR(err);
    } else {
      LOG(FATAL) << "Unknown OpenCL format " << fmt_;
    }
    device_built_flag_.resize(w->devices.size(), false);
    // initialize the kernel id, need to lock global table.
    std::lock_guard<std::mutex> lock(w->mu);
    for (const auto& kv : fmap_) {
      const std::string& key = kv.first;
      KTRefEntry e;
      if (w->free_kernel_ids.size() != 0) {
        e.kernel_id = w->free_kernel_ids.back();
        w->free_kernel_ids.pop_back();
      } else {
        e.kernel_id = w->num_registered_kernels++;
      }
      e.version = w->timestamp++;
      kid_map_[key] = e;
    }
  }
  // install a new kernel to thread local entry
  cl_kernel InstallKernel(cl::OpenCLWorkspace* w,
                          cl::OpenCLThreadEntry* t,
                          const std::string& func_name,
                          const KTRefEntry& e) {
    std::lock_guard<std::mutex> lock(build_lock_);
    int dev_id = t->context.dev_id;
    if (!device_built_flag_[dev_id]) {
      // build program
      cl_int err;
      cl_device_id dev = w->devices[dev_id];
      err = clBuildProgram(program_, 1, &dev, nullptr, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        size_t len;
        std::string log;
        clGetProgramBuildInfo(
            program_, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        log.resize(len);
        clGetProgramBuildInfo(
            program_, dev, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        LOG(FATAL) << "OpenCL build error for device=" << dev << log;
      }
      device_built_flag_[dev_id] = true;
    }
    // build kernel
    cl_int err;
    cl_kernel kernel = clCreateKernel(program_, func_name.c_str(), &err);
    OPENCL_CHECK_ERROR(err);
    t->kernel_table[e.kernel_id].kernel = kernel;
    t->kernel_table[e.kernel_id].version = e.version;
    kernels_.push_back(kernel);
    return kernel;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // Module local mutex
  std::mutex build_lock_;
  // the binary data
  cl_program program_{nullptr};
  // build info
  std::vector<bool> device_built_flag_;
  // kernel id cache
  std::unordered_map<std::string, KTRefEntry> kid_map_;
  // kernels build so far.
  std::vector<cl_kernel> kernels_;
};

class OpenCLWrappedFunc {
 public:
  // initialize the CUDA function.
  void Init(OpenCLModuleNode* m,
            std::shared_ptr<ModuleNode> sptr,
            OpenCLModuleNode::KTRefEntry entry,
            std::string func_name,
            std::vector<size_t> arg_size,
            const std::vector<std::string>& thread_axis_tags)  {
    m_ = m;
    sptr_ = sptr;
    entry_ = entry;
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
    if (entry_.kernel_id >= t->kernel_table.size()) {
      t->kernel_table.resize(entry_.kernel_id + 1);
    }
    const auto& e = t->kernel_table[entry_.kernel_id];
    cl_kernel kernel = e.kernel;
    if (kernel == nullptr || e.version != entry_.version) {
      kernel = m_->InstallKernel(w, t, func_name_, entry_);
    }
    // setup arguments.
    for (cl_uint i = 0; i < arg_size_.size(); ++i) {
      OPENCL_CALL(clSetKernelArg(kernel, i, arg_size_[i], void_args[i]));
    }
    cl_command_queue queue = w->GetQueue(t->context);
    ThreadWorkLoad wl = thread_axis_cfg_.Extract(args);
    cl_uint work_dim = static_cast<cl_uint>(thread_axis_cfg_.work_dim());
    for (cl_uint i = 0; i < work_dim; ++i) {
      wl.work_size[i] *= wl.work_size[i + 3];
    }
    // launch kernel
    OPENCL_CALL(clEnqueueNDRangeKernel(
        queue, kernel, work_dim, nullptr,
        wl.work_size,
        wl.work_size + 3,
        0, nullptr, nullptr));
  }

 private:
  // The module
  OpenCLModuleNode* m_;
  // resource handle
  std::shared_ptr<ModuleNode> sptr_;
  // global kernel id in the kernel table.
  OpenCLModuleNode::KTRefEntry entry_;
  // The name of the function.
  std::string func_name_;
  // convert code for void argument
  std::vector<size_t> arg_size_;
  // thread axis config
  ThreadAxisConfig thread_axis_cfg_;
};

/*!
 * \brief Automatically detect and set cuda device.
 * \param args The arguments.
 */
void AutoSetOpenCLDevice(const TVMArgs& args, TVMRetValue* rv) {
  CHECK_EQ(args.size(), 3);
  TVMValue* values = static_cast<TVMValue*>(args[0].operator void*());
  int* type_codes = static_cast<int*>(args[1].operator void*());
  int num_args = args[2].operator int();

  // TODO(tqchen): merge this with CUDA logic.
  int dev_id = -1;
  for (int i = 0; i < num_args; ++i) {
    if (type_codes[i] == kArrayHandle) {
      TVMContext ctx = static_cast<TVMArray*>(values[i].v_handle)->ctx;
      CHECK_EQ(ctx.dev_mask, kOpenCL)
          << "All operands need to be OpenCL";
      if (dev_id == -1) {
        dev_id = ctx.dev_id;
      } else {
        CHECK_EQ(dev_id, ctx.dev_id)
            << "Operands comes from different devices ";
      }
    }
  }
  CHECK_NE(dev_id, -1)
      << "Cannot detect device id from list";
  cl::OpenCLThreadEntry::ThreadLocal()->context.dev_id = dev_id;
}

PackedFunc OpenCLModuleNode::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  CHECK_EQ(sptr_to_self.get(), this);
  CHECK_NE(name, symbol::tvm_module_main)
      << "Device function do not have main";
  if (name == symbol::tvm_entry_setdevice) {
    return PackedFunc(AutoSetOpenCLDevice);
  }
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  OpenCLWrappedFunc f;
  std::vector<size_t> arg_size(info.arg_types.size());
  for (size_t i = 0; i < info.arg_types.size(); ++i) {
    TVMType t = info.arg_types[i];
    CHECK_EQ(t.lanes, 1U);
    uint32_t bits = t.bits;
    CHECK_EQ(bits % 8, 0U);
    arg_size[i] = bits / 8;
  }
  // initialize the wrapped func.
  f.Init(this, sptr_to_self, kid_map_.at(name),
         name, arg_size, info.thread_axis_tags);
  return PackFromVoidAddrArgs(f, info.arg_types);
}

Module OpenCLModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap) {
  std::shared_ptr<OpenCLModuleNode> n =
      std::make_shared<OpenCLModuleNode>(data, fmt, fmap);
  n->InitProgram();
  return Module(n);
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENCL_RUNTIME
