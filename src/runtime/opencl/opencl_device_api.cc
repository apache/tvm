/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_device_api.cc
 */
#include "./opencl_common.h"

#if TVM_OPENCL_RUNTIME

#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>

namespace tvm {
namespace runtime {
namespace cl {

OpenCLWorkspace* OpenCLWorkspace::Global() {
  static OpenCLWorkspace inst;
  return &inst;
}

void* OpenCLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment) {
  cl_int err_code;
  cl_mem mptr = clCreateBuffer(
      this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

void OpenCLWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  cl_mem mptr = static_cast<cl_mem>(ptr);
  OPENCL_CALL(clReleaseMemObject(mptr));
}

void OpenCLWorkspace::CopyDataFromTo(const void* from,
                                     void* to,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  if (ctx_from.device_type == kOpenCL && ctx_to.device_type == kOpenCL) {
    OPENCL_CALL(clEnqueueCopyBuffer(
        this->GetQueue(ctx_to),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        static_cast<cl_mem>(to),
        0, 0, size, 0, nullptr, nullptr));
  } else if (ctx_from.device_type == kOpenCL && ctx_to.device_type == kCPU) {
    OPENCL_CALL(clEnqueueReadBuffer(
        this->GetQueue(ctx_from),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        CL_FALSE, 0, size, to,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_from)));
  } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kOpenCL) {
    OPENCL_CALL(clEnqueueWriteBuffer(
        this->GetQueue(ctx_to),
        static_cast<cl_mem>(to),
        CL_FALSE, 0, size, from,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
  }
}

void OpenCLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  OPENCL_CALL(clFinish(this->GetQueue(ctx)));
}

typedef dmlc::ThreadLocalStore<OpenCLThreadEntry> OpenCLThreadStore;

OpenCLThreadEntry* OpenCLThreadEntry::ThreadLocal() {
  return OpenCLThreadStore::Get();
}

std::string GetPlatformInfo(
    cl_platform_id pid, cl_platform_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::string GetDeviceInfo(
    cl_device_id pid, cl_device_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_platform_id> GetPlatformIDs() {
  cl_uint ret_size;
  OPENCL_CALL(clGetPlatformIDs(0, nullptr, &ret_size));
  std::vector<cl_platform_id> ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_device_id> GetDeviceIDs(
    cl_platform_id pid, std::string device_type) {
  cl_device_type dtype = CL_DEVICE_TYPE_ALL;
  if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
  if (device_type == "gpu") dtype = CL_DEVICE_TYPE_CPU;
  if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
  cl_uint ret_size;
  OPENCL_CALL(clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size));
  std::vector<cl_device_id> ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetDeviceIDs(pid, dtype, ret_size, &ret[0], nullptr));
  return ret;
}

bool MatchPlatformInfo(
    cl_platform_id pid,
    cl_platform_info param_name,
    std::string value) {
  if (value.length() == 0) return true;
  std::string param_value = GetPlatformInfo(pid, param_name);
  return param_value.find(value) != std::string::npos;
}

bool InitOpenCL(TVMArgs args, TVMRetValue* rv) {
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  std::lock_guard<std::mutex>(w->mu);
  if (w->initialized()) return false;
  // matching conditions
  std::string platform_name, device_type;

  for (size_t i = 0; i < args.num_args; ++i) {
    std::string arg = args[i];
    size_t pos = arg.find_first_of('=');
    CHECK_EQ(pos, std::string::npos)
        << "Argumentes need to be key=value";
    std::string key = arg.substr(0, pos);
    std::string val = arg.substr(pos + 1, arg.length() - pos - 1);
    if (key == "platform_name") {
      platform_name = val;
    } else if (key == "device_type") {
      device_type = val;
    } else {
      LOG(FATAL) << "unknown DeviceInit option " << key;
    }
  }
  // matched platforms
  std::vector<cl_platform_id> platform_matched;
  for (cl_platform_id pid : cl::GetPlatformIDs()) {
    bool matched = true;
    if (!cl::MatchPlatformInfo(pid, CL_PLATFORM_NAME, platform_name)) matched = false;
    if (matched) platform_matched.push_back(pid);
  }
  if (platform_matched.size() == 0) {
    LOG(FATAL) << "No OpenCL platform matched given existing options ...";
  }
  if (platform_matched.size() > 1) {
    LOG(WARNING) << "Multiple OpenCL platforms matched, use the first one ... ";
  }
  w->platform_id = platform_matched[0];

  LOG(INFO) << "Initialize OpenCL platform \'"
            << cl::GetPlatformInfo(w->platform_id, CL_PLATFORM_NAME) << '\'';
  std::vector<cl_device_id> devices_matched =
      cl::GetDeviceIDs(w->platform_id, device_type);
  CHECK_GT(devices_matched.size(), 0U)
      << "No OpenCL device any device matched given the options";
  w->devices = devices_matched;
  cl_int err_code;
  w->context = clCreateContext(
      nullptr, w->devices.size(), &(w->devices[0]),
      nullptr, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  CHECK_EQ(w->queues.size(), 0U);
  for (size_t i = 0; i < w->devices.size(); ++i) {
    cl_device_id did = w->devices[i];
    w->queues.push_back(
        clCreateCommandQueue(w->context, did, 0, &err_code));
    OPENCL_CHECK_ERROR(err_code);
    LOG(INFO) << "opencl(" << i
              << ")=\'" << cl::GetDeviceInfo(did, CL_DEVICE_NAME)
              << "\' cl_device_id=" << did;
  }
  return true;
}

TVM_REGISTER_GLOBAL(_module_init_opencl)
.set_body(InitOpenCL);

TVM_REGISTER_GLOBAL(_device_api_opencl)
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = OpenCLWorkspace::Global();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace cl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENCL_RUNTIME
