/*!
 *  Copyright (c) 2017 by Contributors
 * \file opencl_device_api.cc
 */
#include <tvm/runtime/registry.h>
#include <dmlc/thread_local.h>
#include <tvm/container.h>
#include <tvm/ir.h>
#include <tvm/packed_func_ext.h>
#include "./opencl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

template <OpenCLPlatform T>
const std::shared_ptr<OpenCLWorkspace<T>>& OpenCLWorkspace<T>::Global() {
  static std::shared_ptr<OpenCLWorkspace<T>> inst = std::make_shared<OpenCLWorkspace<T>>();
  return inst;
}

template <OpenCLPlatform T>
void OpenCLWorkspace<T>::SetDevice(TVMContext ctx) {
  OpenCLThreadEntry<T>::ThreadLocal()->context.device_id = ctx.device_id;
}

template <OpenCLPlatform T>
void OpenCLWorkspace<T>::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index< devices.size());
    return;
  }
  CHECK_LT(index, devices.size())
      << "Invalid device id " << index;
  switch (kind) {
    case kExist: break;
    case kMaxThreadsPerBlock: {
      size_t value;
      OPENCL_CALL(clGetDeviceInfo(
          devices[index],  CL_DEVICE_MAX_WORK_GROUP_SIZE,
          sizeof(size_t), &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kWarpSize: {
      /* TODO: the warp size of OpenCL device is not always 1
               e.g. Intel Graphics has a sub group concept which contains 8 - 32 work items,
               corresponding to the number of SIMD entries the heardware configures.
               We need to figure out a way to query this information from the hardware.
      */
      *rv = 1;
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      cl_ulong value;
      OPENCL_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_LOCAL_MEM_SIZE,
          sizeof(cl_ulong), &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kComputeVersion: return;
    case kDeviceName: {
      char value[128] = {0};
      OPENCL_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_NAME,
          sizeof(value) - 1, value, nullptr));
      *rv = std::string(value);
      break;
    }
    case kMaxClockRate: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_MAX_CLOCK_FREQUENCY,
          sizeof(cl_uint), &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMultiProcessorCount: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_MAX_COMPUTE_UNITS,
          sizeof(cl_uint), &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxThreadDimensions: {
      size_t dims[3];
      OPENCL_CALL(clGetDeviceInfo(
          devices[index], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, nullptr));

      std::stringstream ss;  // use json string to return multiple int values;
      ss << "[" << dims[0] <<", " << dims[1] << ", " << dims[2] << "]";
      *rv = ss.str();
      break;
    }
  }
}

template <OpenCLPlatform T>
void* OpenCLWorkspace<T>::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment, TVMType type_hint) {
  this->Init();
  CHECK(context != nullptr) << "No OpenCL device";
  cl_int err_code;
  cl_mem mptr = clCreateBuffer(
      this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

template <OpenCLPlatform T>
void OpenCLWorkspace<T>::FreeDataSpace(TVMContext ctx, void* ptr) {
  cl_mem mptr = static_cast<cl_mem>(ptr);
  OPENCL_CALL(clReleaseMemObject(mptr));
}

template <OpenCLPlatform T>
void OpenCLWorkspace<T>::CopyDataFromTo(const void* from,
                                        size_t from_offset,
                                        void* to,
                                        size_t to_offset,
                                        size_t size,
                                        TVMContext ctx_from,
                                        TVMContext ctx_to,
                                        TVMType type_hint,
                                        TVMStreamHandle stream) {
  this->Init();
  CHECK(stream == nullptr);
  if (IsOpenCLDevice(ctx_from) && IsOpenCLDevice(ctx_to)) {
    OPENCL_CALL(clEnqueueCopyBuffer(
        this->GetQueue(ctx_to),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        static_cast<cl_mem>(to),
        from_offset, to_offset, size, 0, nullptr, nullptr));
  } else if (IsOpenCLDevice(ctx_from) && ctx_to.device_type == kDLCPU) {
    OPENCL_CALL(clEnqueueReadBuffer(
        this->GetQueue(ctx_from),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        CL_FALSE, from_offset, size,
        static_cast<char*>(to) + to_offset,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_from)));
  } else if (ctx_from.device_type == kDLCPU && IsOpenCLDevice(ctx_to)) {
    OPENCL_CALL(clEnqueueWriteBuffer(
        this->GetQueue(ctx_to),
        static_cast<cl_mem>(to),
        CL_FALSE, to_offset, size,
        static_cast<const char*>(from) + from_offset,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
  }
}

template <OpenCLPlatform T>
void OpenCLWorkspace<T>::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  OPENCL_CALL(clFinish(this->GetQueue(ctx)));
}

template <OpenCLPlatform T>
void* OpenCLWorkspace<T>::AllocWorkspace(TVMContext ctx,
                                         size_t size,
                                         TVMType type_hint) {
  return OpenCLThreadEntry<T>::ThreadLocal()->pool.AllocWorkspace(ctx, size);
}

template <OpenCLPlatform T>
void OpenCLWorkspace<T>::FreeWorkspace(TVMContext ctx, void* data) {
  OpenCLThreadEntry<T>::ThreadLocal()->pool.FreeWorkspace(ctx, data);
}

template <OpenCLPlatform T>
using OpenCLThreadStore = dmlc::ThreadLocalStore<OpenCLThreadEntry<T>>;

template <OpenCLPlatform T>
OpenCLThreadEntry<T>* OpenCLThreadEntry<T>::ThreadLocal() {
  return OpenCLThreadStore<T>::Get();
}

template <>
OpenCLThreadEntry<OpenCLPlatform::kSDAccel>::OpenCLThreadEntry()
    : pool(static_cast<DLDeviceType>(kDLSDAccel),
           OpenCLWorkspace<OpenCLPlatform::kSDAccel>::Global()) {
  context.device_id = 0;
  context.device_type = static_cast<DLDeviceType>(kDLSDAccel);
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
  cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
  std::vector<cl_platform_id> ret;
  if (code != CL_SUCCESS) return ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_device_id> GetDeviceIDs(
    cl_platform_id pid, std::string device_type) {
  cl_device_type dtype = CL_DEVICE_TYPE_ALL;
  if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
  if (device_type == "gpu") dtype = CL_DEVICE_TYPE_GPU;
  if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
  cl_uint ret_size;
  cl_int code = clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size);
  std::vector<cl_device_id> ret;
  if (code != CL_SUCCESS) return ret;
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

template <OpenCLPlatform T>
void OpenCLWorkspace<T>::Init(const std::vector<std::string>& device_types,
                              const std::string& platform_name) {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  if (initialized_) return;
  initialized_ = true;
  if (context != nullptr) return;
  // matched platforms
  std::vector<cl_platform_id> platform_ids = cl::GetPlatformIDs();
  if (platform_ids.size() == 0) {
    LOG(WARNING) << "No OpenCL platform matched given existing options ...";
    return;
  }
  this->platform_id = nullptr;
  for (auto platform_id : platform_ids) {
    if (!MatchPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name)) {
      continue;
    }
    for (auto device_type : device_types) {
      std::vector<cl_device_id> devices_matched = cl::GetDeviceIDs(platform_id, device_type);
      if (devices_matched.size() > 0) {
        this->platform_id = platform_id;
        this->platform_name = cl::GetPlatformInfo(platform_id, CL_PLATFORM_NAME);
        this->device_type = device_type;
        this->devices = devices_matched;
        LOG(INFO) << "Initialize OpenCL platform \'" << this->platform_name << '\'';
        break;
      }
      LOG(INFO) << "\'" << cl::GetPlatformInfo(platform_id, CL_PLATFORM_NAME)
                << "\' platform has no OpenCL device: " << device_type << " mode";
    }
  }
  if (this->platform_id == nullptr) {
    LOG(WARNING) << "No OpenCL device";
    return;
  }
  cl_int err_code;
  this->context = clCreateContext(
      nullptr, this->devices.size(), &(this->devices[0]),
      nullptr, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  CHECK_EQ(this->queues.size(), 0U);
  for (size_t i = 0; i < this->devices.size(); ++i) {
    cl_device_id did = this->devices[i];
    this->queues.push_back(
        clCreateCommandQueue(this->context, did, 0, &err_code));
    OPENCL_CHECK_ERROR(err_code);
    LOG(INFO) << "opencl(" << i
              << ")=\'" << cl::GetDeviceInfo(did, CL_DEVICE_NAME)
              << "\' cl_device_id=" << did;
  }
}

template <>
void OpenCLWorkspace<OpenCLPlatform::kGPU>::Init() {
  Init({"gpu", "cpu"});
}

template <>
void OpenCLWorkspace<OpenCLPlatform::kSDAccel>::Init() {
  Init({"accelerator"}, "Xilinx");
}

template <>
bool OpenCLWorkspace<OpenCLPlatform::kSDAccel>::IsOpenCLDevice(TVMContext ctx) {
  return ctx.device_type == static_cast<DLDeviceType>(kDLSDAccel);
}

TVM_REGISTER_GLOBAL("device_api.opencl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = OpenCLWorkspace<OpenCLPlatform::kGPU>::Global().get();
    *rv = static_cast<void*>(ptr);
  });

TVM_REGISTER_GLOBAL("device_api.sdaccel")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = OpenCLWorkspace<OpenCLPlatform::kSDAccel>::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
