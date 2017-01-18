/*!
 *  Copyright (c) 2016 by Contributors
 * \file device_api_opencl.h
 * \brief OpenCL specific API
 */
#ifndef TVM_RUNTIME_DEVICE_API_OPENCL_H_
#define TVM_RUNTIME_DEVICE_API_OPENCL_H_

#if TVM_OPENCL_RUNTIME

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <mutex>
#include <string>
#include <vector>


namespace tvm {
namespace runtime {
namespace cl {

static_assert(sizeof(cl_mem) ==sizeof(void*),
              "Required to store cl_mem inside void*");

inline const char* CLGetErrorString(cl_int error) {
  switch (error) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    default: return "Unknown OpenCL error code";
  }
}

/*!
 * \brief Protected OpenCL call
 * \param func Expression to call.
 */
#define OPENCL_CHECK_ERROR(e)                                           \
  {                                                                     \
    CHECK(e == CL_SUCCESS)                                              \
        << "OpenCL Error, code=" << e << ": " << cl::CLGetErrorString(e); \
  }

#define OPENCL_CALL(func)                                             \
  {                                                                   \
    cl_int e = (func);                                                \
    OPENCL_CHECK_ERROR(e);                                            \
  }

// Process local opencl workspace
class OpenCLWorkspace {
 public:
  // global platform id
  cl_platform_id platform_id;
  // global context of this process
  cl_context context{nullptr};
  // the devices
  std::vector<cl_device_id> devices;
  // the queues
  std::vector<cl_command_queue> queues;
  // the mutex for initialization
  std::mutex mu;
  // destructor
  ~OpenCLWorkspace() {
    if (context != nullptr) {
      OPENCL_CALL(clReleaseContext(context));
    }
  }
  // whether the workspace is initialized.
  inline bool initialized() const {
    return context != nullptr;
  }
  // get the queue of the context
  cl_command_queue GetQueue(TVMContext ctx) const {
    CHECK_EQ(ctx.dev_mask, kOpenCL);
    CHECK(initialized())
        << "The OpenCL is not initialized";
    CHECK(ctx.dev_id >= 0  && static_cast<size_t>(ctx.dev_id) < queues.size())
        << "Invalid OpenCL dev_id=" << ctx.dev_id;
    return queues[ctx.dev_id];
  }
  // get the global workspace
  static OpenCLWorkspace* Global() {
    static OpenCLWorkspace inst;
    return &inst;
  }
};

inline std::string GetPlatformInfo(
    cl_platform_id pid, cl_platform_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

inline std::string GetDeviceInfo(
    cl_device_id pid, cl_device_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

inline std::vector<cl_platform_id> GetPlatformIDs() {
  cl_uint ret_size;
  OPENCL_CALL(clGetPlatformIDs(0, nullptr, &ret_size));
  std::vector<cl_platform_id> ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
  return ret;
}

inline std::vector<cl_device_id> GetDeviceIDs(
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

inline bool MatchPlatformInfo(
    cl_platform_id pid,
    cl_platform_info param_name,
    std::string value) {
  if (value.length() == 0) return true;
  std::string param_value = GetPlatformInfo(pid, param_name);
  return param_value.find(value) != std::string::npos;
}

}  // namespace cl

template<>
inline bool DeviceInit<kOpenCL>(const char** option_keys,
                                const char** option_vals,
                                int num_options) {
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  std::lock_guard<std::mutex>(w->mu);
  if (w->initialized()) return false;
  // matching conditions
  std::string platform_name, device_type;
  for (int i = 0; i < num_options; ++i) {
    std::string key = option_keys[i];
    std::string val = option_vals[i];
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

template<>
inline void* AllocDataSpace<kOpenCL>(TVMContext ctx, size_t size, size_t alignment) {
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  cl_int err_code;
  cl_mem mptr = clCreateBuffer(
      w->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

template<>
inline void FreeDataSpace<kOpenCL>(TVMContext ctx, void* ptr) {
  cl_mem mptr = static_cast<cl_mem>(ptr);
  OPENCL_CALL(clReleaseMemObject(mptr));
}

template<>
inline void CopyDataFromTo<kOpenCL>(const void* from,
                                    void* to,
                                    size_t size,
                                    TVMContext ctx_from,
                                    TVMContext ctx_to,
                                    TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  if (ctx_from.dev_mask == kOpenCL && ctx_to.dev_mask == kOpenCL) {
    OPENCL_CALL(clEnqueueCopyBuffer(
        w->GetQueue(ctx_to),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        static_cast<cl_mem>(to),
        0, 0, size, 0, nullptr, nullptr));
  } else if (ctx_from.dev_mask == kOpenCL && ctx_to.dev_mask == kCPU) {
    OPENCL_CALL(clEnqueueReadBuffer(
        w->GetQueue(ctx_from),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        CL_FALSE, 0, size, to,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(w->GetQueue(ctx_from)));
  } else if (ctx_from.dev_mask == kCPU && ctx_to.dev_mask == kOpenCL) {
    OPENCL_CALL(clEnqueueWriteBuffer(
        w->GetQueue(ctx_to),
        static_cast<cl_mem>(to),
        CL_FALSE, 0, size, from,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(w->GetQueue(ctx_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to GPU or between GPU";
  }
}

template<>
inline void StreamSync<kOpenCL>(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  cl::OpenCLWorkspace* w = cl::OpenCLWorkspace::Global();
  OPENCL_CALL(clFinish(w->GetQueue(ctx)));
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_OPENCL_RUNTIME
#endif  // TVM_RUNTIME_DEVICE_API_OPENCL_H_
