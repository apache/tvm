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

const std::shared_ptr<OpenCLWorkspace>& OpenCLWorkspace::Global() {
  static std::shared_ptr<OpenCLWorkspace> inst = std::make_shared<OpenCLWorkspace>();
  return inst;
}

void OpenCLWorkspace::SetDevice(TVMContext ctx) {
  OpenCLThreadEntry::ThreadLocal()->context.device_id = ctx.device_id;
}

void OpenCLWorkspace::GetAttr(
    TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index< devices.size());
    return;
  }
  CHECK_LT(index, devices.size())
      << "Invalid device id " << index;
  size_t value;
  switch (kind) {
    case kMaxThreadsPerBlock: {
      OPENCL_CALL(clGetDeviceInfo(
          devices[index],  CL_DEVICE_MAX_WORK_GROUP_SIZE,
          sizeof(size_t), &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kWarpSize: {
      *rv = 1;
      break;
    }
  case kComputeVersion: return;
  case kExist: break;
  }
}

void* OpenCLWorkspace::AllocDataSpace(
    TVMContext ctx, size_t size, size_t alignment, TVMType type_hint) {
  this->Init();
  CHECK(context != nullptr) << "No OpenCL device";
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
                                     size_t from_offset,
                                     void* to,
                                     size_t to_offset,
                                     size_t size,
                                     TVMContext ctx_from,
                                     TVMContext ctx_to,
                                     TVMStreamHandle stream) {
  this->Init();
  CHECK(stream == nullptr);
  if (ctx_from.device_type == kDLOpenCL && ctx_to.device_type == kDLOpenCL) {
    OPENCL_CALL(clEnqueueCopyBuffer(
        this->GetQueue(ctx_to),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        static_cast<cl_mem>(to),
        from_offset, to_offset, size, 0, nullptr, nullptr));
  } else if (ctx_from.device_type == kDLOpenCL && ctx_to.device_type == kDLCPU) {
    OPENCL_CALL(clEnqueueReadBuffer(
        this->GetQueue(ctx_from),
        static_cast<cl_mem>((void*)from),  // NOLINT(*)
        CL_FALSE, from_offset, size,
        static_cast<char*>(to) + to_offset,
        0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_from)));
  } else if (ctx_from.device_type == kDLCPU && ctx_to.device_type == kDLOpenCL) {
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

void OpenCLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  CHECK(stream == nullptr);
  OPENCL_CALL(clFinish(this->GetQueue(ctx)));
}

void* OpenCLWorkspace::AllocWorkspace(TVMContext ctx,
                                      size_t size,
                                      TVMType type_hint) {
  return OpenCLThreadEntry::ThreadLocal()->pool.AllocWorkspace(ctx, size);
}

void OpenCLWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  OpenCLThreadEntry::ThreadLocal()->pool.FreeWorkspace(ctx, data);
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

void OpenCLWorkspace::Init() {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  if (initialized_) return;
  initialized_ = true;
  if (context != nullptr) return;
  // matched platforms
  std::vector<cl_platform_id> platform_matched = cl::GetPlatformIDs();
  if (platform_matched.size() == 0) {
    LOG(WARNING) << "No OpenCL platform matched given existing options ...";
    return;
  }
  if (platform_matched.size() > 1) {
    LOG(WARNING) << "Multiple OpenCL platforms matched, use the first one ... ";
  }
  this->platform_id = platform_matched[0];
  LOG(INFO) << "Initialize OpenCL platform \'"
            << cl::GetPlatformInfo(this->platform_id, CL_PLATFORM_NAME) << '\'';
  std::vector<cl_device_id> devices_matched =
      cl::GetDeviceIDs(this->platform_id, "gpu");
  if (devices_matched.size() == 0) {
    LOG(WARNING) << "No OpenCL device any device matched given the options: gpu mode";
    LOG(WARNING) << "Now try OpenCL cpu mode";
    devices_matched = cl::GetDeviceIDs(this->platform_id, "cpu");
    if (devices_matched.size() == 0) {
      LOG(WARNING) << "No OpenCL device any device matched given the options: cpu mode";
      return;
    }
  }
  this->devices = devices_matched;
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

bool InitOpenCL(TVMArgs args, TVMRetValue* rv) {
  cl::OpenCLWorkspace::Global()->Init();
  return true;
}

TVM_REGISTER_GLOBAL("device_api.opencl")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = OpenCLWorkspace::Global().get();
    *rv = static_cast<void*>(ptr);
  });

}  // namespace cl
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_OPENCL_RUNTIME
