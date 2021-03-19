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
 * \file opencl_device_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "opencl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

std::string GetPlatformInfo(cl_platform_id pid, cl_platform_info param_name);
std::string GetDeviceInfo(cl_device_id pid, cl_device_info param_name);

OpenCLThreadEntry* OpenCLWorkspace::GetThreadEntry() { return OpenCLThreadEntry::ThreadLocal(); }

OpenCLWorkspace* OpenCLWorkspace::Global() {
  static OpenCLWorkspace* inst = new OpenCLWorkspace();
  return inst;
}

void OpenCLWorkspace::SetDevice(Device dev) { GetThreadEntry()->device.device_id = dev.device_id; }

void OpenCLWorkspace::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(dev.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < devices.size());
    return;
  }
  ICHECK_LT(index, devices.size()) << "Invalid device id " << index;
  switch (kind) {
    case kExist:
      break;
    case kMaxThreadsPerBlock: {
      size_t value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                                  &value, nullptr));
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
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
                                  &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kComputeVersion: {
      // String returned is "OpenCL $MAJOR.$MINOR $VENDOR_INFO".  To
      // match other implementations, we want to return "$MAJOR.$MINOR"
      std::string ret = GetDeviceInfo(devices[index], CL_DEVICE_VERSION);

      const size_t version_start = 7;  // Length of initial "OpenCL " prefix to skip
      const size_t version_end = ret.find(' ', version_start);
      *rv = ret.substr(version_start, version_end - version_start);
      break;
    }
      return;
    case kDeviceName:
      *rv = GetDeviceInfo(devices[index], CL_DEVICE_NAME);
      break;
    case kMaxClockRate: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint),
                                  &value, nullptr));
      // OpenCL returns the clock rate in MHz, while CUDA/ROCm return the
      // clock rate in kHz.  Converting to the same units for each.
      *rv = static_cast<int32_t>(value * 1000);
      break;
    }
    case kMultiProcessorCount: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                                  &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxThreadDimensions: {
      size_t dims[3];
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims,
                                  nullptr));

      std::stringstream ss;  // use json string to return multiple int values;
      ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
      *rv = ss.str();
      break;
    }
    case kMaxRegistersPerBlock:
      return;
    case kGcnArch:
      return;
    case kApiVersion: {
      *rv = CL_TARGET_OPENCL_VERSION;
      break;
    }
    case kDriverVersion: {
      char value[128] = {0};
      OPENCL_CALL(
          clGetDeviceInfo(devices[index], CL_DRIVER_VERSION, sizeof(value) - 1, value, nullptr));
      *rv = std::string(value);
      break;
    }
  }
}

void* OpenCLWorkspace::AllocDataSpace(Device dev, size_t size, size_t alignment,
                                      DLDataType type_hint) {
  this->Init();
  ICHECK(context != nullptr) << "No OpenCL device";
  cl_int err_code;
  cl_mem mptr = clCreateBuffer(this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

void OpenCLWorkspace::FreeDataSpace(Device dev, void* ptr) {
  // We have to make sure that the memory object is not in the command queue
  // for some OpenCL platforms.
  OPENCL_CALL(clFinish(this->GetQueue(dev)));

  cl_mem mptr = static_cast<cl_mem>(ptr);
  OPENCL_CALL(clReleaseMemObject(mptr));
}

void* OpenCLWorkspace::AllocTexture(TVMContext ctx, size_t width, size_t height, DLDataType type_hint) {
  this->Init();
  ICHECK(context != nullptr) << "No OpenCL device";
  cl_int err_code;
  ICHECK(type_hint.code == DLDataTypeCode::kDLFloat) << "Only float and half image2d allocations are supported";
  ICHECK(type_hint.bits == 16 || type_hint.bits == 32) << "Only CL_FLOAT and CL_HALF_FLOAT image2d allocations are supported";
  cl_channel_type cl_type = CL_FLOAT;
  if (type_hint.bits == 16) {
    cl_type = CL_HALF_FLOAT;
  }

  cl_image_format format = { CL_RGBA, cl_type };
  cl_image_desc descriptor = { CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0 };
  cl_mem mptr = clCreateImage(
    this->context,
    CL_MEM_READ_WRITE,
    &format,
    &descriptor,
    nullptr,
    &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

void OpenCLWorkspace::FreeTexture(TVMContext ctx, void* ptr) {
  OPENCL_CALL(clFinish(this->GetQueue(ctx)));
  cl_mem mptr = static_cast<cl_mem>(ptr);
  OPENCL_CALL(clReleaseMemObject(mptr));
}

void* OpenCLWorkspace::AllocTextureWorkspace(TVMContext ctx, size_t width, size_t height, DLDataType type_hint) {
  return GetThreadEntry()->texture_pool.AllocTexture(ctx, width, height, type_hint);
}

void OpenCLWorkspace::FreeTextureWorkspace(TVMContext ctx, void* ptr) {
  GetThreadEntry()->texture_pool.FreeTexture(ctx, ptr);
}

void OpenCLWorkspace::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                     size_t to_offset, size_t size, Device dev_from, Device dev_to,
                                     DLDataType type_hint, TVMStreamHandle stream) {
  this->Init();
  ICHECK(stream == nullptr);
  if (IsOpenCLDevice(dev_from) && IsOpenCLDevice(dev_to)) {
    OPENCL_CALL(clEnqueueCopyBuffer(this->GetQueue(dev_to),
                                    static_cast<cl_mem>((void*)from),  // NOLINT(*)
                                    static_cast<cl_mem>(to), from_offset, to_offset, size, 0,
                                    nullptr, nullptr));
  } else if (IsOpenCLDevice(dev_from) && dev_to.device_type == kDLCPU) {
    OPENCL_CALL(clEnqueueReadBuffer(this->GetQueue(dev_from),
                                    static_cast<cl_mem>((void*)from),  // NOLINT(*)
                                    CL_FALSE, from_offset, size, static_cast<char*>(to) + to_offset,
                                    0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(dev_from)));
  } else if (dev_from.device_type == kDLCPU && IsOpenCLDevice(dev_to)) {
    OPENCL_CALL(clEnqueueWriteBuffer(this->GetQueue(dev_to), static_cast<cl_mem>(to), CL_FALSE,
                                     to_offset, size, static_cast<const char*>(from) + from_offset,
                                     0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(dev_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
  }
}

void OpenCLWorkspace::StreamSync(Device dev, TVMStreamHandle stream) {
  ICHECK(stream == nullptr);
  OPENCL_CALL(clFinish(this->GetQueue(dev)));
}

void* OpenCLWorkspace::AllocWorkspace(Device dev, size_t size, DLDataType type_hint) {
  return GetThreadEntry()->pool.AllocWorkspace(dev, size);
}

void OpenCLWorkspace::FreeWorkspace(Device dev, void* data) {
  GetThreadEntry()->pool.FreeWorkspace(dev, data);
}

typedef dmlc::ThreadLocalStore<OpenCLThreadEntry> OpenCLThreadStore;

OpenCLThreadEntry* OpenCLThreadEntry::ThreadLocal() { return OpenCLThreadStore::Get(); }

std::string GetPlatformInfo(cl_platform_id pid, cl_platform_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::string GetDeviceInfo(cl_device_id pid, cl_device_info param_name) {
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

std::vector<cl_device_id> GetDeviceIDs(cl_platform_id pid, std::string device_type) {
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

bool MatchPlatformInfo(cl_platform_id pid, cl_platform_info param_name, std::string value) {
  if (value.length() == 0) return true;
  std::string param_value = GetPlatformInfo(pid, param_name);
  return param_value.find(value) != std::string::npos;
}

void OpenCLWorkspace::Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name) {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  if (initialized_) return;
  if (context != nullptr) return;
  this->type_key = type_key;
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
    std::vector<cl_device_id> devices_matched = cl::GetDeviceIDs(platform_id, device_type);
    if ((devices_matched.size() == 0) && (device_type == "gpu")) {
      LOG(WARNING) << "Using CPU OpenCL device";
      devices_matched = cl::GetDeviceIDs(platform_id, "cpu");
    }
    if (devices_matched.size() > 0) {
      this->platform_id = platform_id;
      this->platform_name = cl::GetPlatformInfo(platform_id, CL_PLATFORM_NAME);
      this->device_type = device_type;
      this->devices = devices_matched;
      break;
    }
  }
  if (this->platform_id == nullptr) {
    LOG(WARNING) << "No OpenCL device";
    return;
  }
  cl_int err_code;
  this->context = clCreateContext(nullptr, this->devices.size(), &(this->devices[0]), nullptr,
                                  nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  ICHECK_EQ(this->queues.size(), 0U);
  for (size_t i = 0; i < this->devices.size(); ++i) {
    cl_device_id did = this->devices[i];
    this->queues.push_back(clCreateCommandQueue(this->context, did, 0, &err_code));
    OPENCL_CHECK_ERROR(err_code);
  }
  initialized_ = true;
}

TVM_REGISTER_GLOBAL("device_api.opencl").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = OpenCLWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
