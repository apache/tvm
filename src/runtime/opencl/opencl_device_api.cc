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
#include <dmlc/parameter.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <sstream>

#include "opencl_common.h"

#ifdef OPENCL_ENABLE_HOST_PTR
#define CL_MEM_CREATE_FLAGS CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR
#else
#define CL_MEM_CREATE_FLAGS CL_MEM_READ_WRITE
#endif

namespace tvm {
namespace runtime {
namespace cl {

std::string GetPlatformInfo(cl_platform_id pid, cl_platform_info param_name);
std::string GetDeviceInfo(cl_device_id pid, cl_device_info param_name);
std::string GetOpenCLVersion(cl_device_id pid);

struct ImageInfo {
  size_t origin[3] = {};
  size_t region[3] = {};
  size_t row_pitch = 0;
  size_t slice_pitch = 0;
};

/*!
 * \brief Utility to apply a memory layout specific lowering convention
 * to infer the physical shape from the provided DLTensor's logical shape.
 * \param desc Descriptor which contains the buffer and layout tag.
 * \param The DLTensor used to infer the tensors physical shape.
 */
ImageInfo GetImageInfo(const cl::BufferDescriptor* desc, const DLTensor* tensor) {
  ImageInfo info{};
  ICHECK(tensor->dtype.lanes == 1) << "Image dtype has lanes: " << tensor->dtype.lanes;

  info.origin[0] = info.origin[1] = info.origin[2] = 0;
  info.row_pitch = 0;
  info.slice_pitch = 0;

  size_t axis = DefaultTextureLayoutSeparator(
      tensor->ndim, cl::BufferDescriptor::ScopeFromMemoryLayout(desc->layout));
  auto texture_shape = ApplyTexture2DFlattening<int64_t>(tensor->shape, tensor->ndim, axis);
  info.region[0] = texture_shape.width;
  info.region[1] = texture_shape.height;
  info.region[2] = 1;
  return info;
}

cl::BufferDescriptor::MemoryLayout cl::BufferDescriptor::MemoryLayoutFromScope(
    Optional<String> mem_scope) {
  if (!mem_scope.defined()) {
    return cl::BufferDescriptor::MemoryLayout::kBuffer1D;
  } else if (mem_scope.value() == "global.texture") {
    return cl::BufferDescriptor::MemoryLayout::kImage2DActivation;
  } else if (mem_scope.value() == "global.texture-weight") {
    return cl::BufferDescriptor::MemoryLayout::kImage2DWeight;
  } else if (mem_scope.value() == "global.texture-nhwc") {
    return cl::BufferDescriptor::MemoryLayout::kImage2DNHWC;
  }
  LOG(FATAL) << "No memory layout defined for memory of scope: " << mem_scope.value();
}

String cl::BufferDescriptor::ScopeFromMemoryLayout(cl::BufferDescriptor::MemoryLayout layout) {
  switch (layout) {
    case cl::BufferDescriptor::MemoryLayout::kBuffer1D:
      return "global";
    case cl::BufferDescriptor::MemoryLayout::kImage2DActivation:
      return "global.texture";
    case cl::BufferDescriptor::MemoryLayout::kImage2DWeight:
      return "global.texture-weight";
    case cl::BufferDescriptor::MemoryLayout::kImage2DNHWC:
      return "global.texture-nhwc";
  }
  LOG(FATAL) << "No scope corresponding to the provided memory layout: "
             << static_cast<int>(layout);
  return "";
}

OpenCLThreadEntry* OpenCLWorkspace::GetThreadEntry() { return OpenCLThreadEntry::ThreadLocal(); }

OpenCLWorkspace* OpenCLWorkspace::Global() {
  static OpenCLWorkspace* inst = new OpenCLWorkspace();
  return inst;
}

cl_device_id OpenCLWorkspace::GetCLDeviceID(int device_id) {
  this->Init();
  ICHECK_LT(device_id, devices.size()) << "Invalid device id " << device_id << ". " << GetError();
  return devices[device_id];
}

void OpenCLWorkspace::SetDevice(Device dev) { GetThreadEntry()->device.device_id = dev.device_id; }

void OpenCLWorkspace::GetAttr(Device dev, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(dev.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < devices.size());
    return;
  }
  cl_device_id device_id = GetCLDeviceID(index);
  switch (kind) {
    case kExist:
      break;
    case kMaxThreadsPerBlock: {
      size_t value;
      OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &value,
                                  nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kWarpSize: {
      /* TODO: the warp size of OpenCL device is not always 1
               e.g. Intel Graphics has a sub group concept which contains 8 - 32 work items,
               corresponding to the number of SIMD entries the heardware configures.
               We need to figure out a way to query this information from the hardware.
      */
      const int warp_size = dmlc::GetEnv("TVM_OPENCL_WARP_SIZE", 1);
      *rv = warp_size;
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      cl_ulong value;
      OPENCL_CALL(
          clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kComputeVersion:
      *rv = GetOpenCLVersion(device_id);
      break;
    case kDeviceName:
      *rv = GetDeviceInfo(device_id, CL_DEVICE_NAME);
      break;
    case kMaxClockRate: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &value,
                                  nullptr));
      // OpenCL returns the clock rate in MHz, while CUDA/ROCm return the
      // clock rate in kHz.  Converting to the same units for each.
      *rv = static_cast<int32_t>(value * 1000);
      break;
    }
    case kMultiProcessorCount: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &value,
                                  nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxThreadDimensions: {
      size_t dims[3];
      OPENCL_CALL(
          clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, nullptr));

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
      OPENCL_CALL(clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(value) - 1, value, nullptr));
      *rv = std::string(value);
      break;
    }
    case kL2CacheSizeBytes: {
      // NOTE(Zihao): this API cannot reflect the real L2 cache size in both CUDA/AMD GPUs.
      cl_ulong value;
      OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(value), &value,
                                  nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kTotalGlobalMemory: {
      cl_ulong total_global_memory;
      OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(total_global_memory),
                                  &total_global_memory, nullptr));
      *rv = static_cast<int64_t>(total_global_memory);
      return;
    }
  }
}

void* OpenCLWorkspace::CreateHostPtrIfEnabled(cl::BufferDescriptor* desc, Device dev, size_t size) {
#if defined(OPENCL_ENABLE_HOST_PTR)
  this->Init();
  cl_int err_code;
  desc->host_ptr = reinterpret_cast<cl_uchar*>(
      clEnqueueMapBuffer(this->GetQueue(dev), desc->buffer, CL_TRUE, CL_MAP_WRITE, 0,
                         sizeof(cl_uchar) * size, 0, nullptr, nullptr, &err_code));
  OPENCL_CHECK_ERROR(err_code);
#endif  // OPENCL_ENABLE_HOST_PTR
  return desc;
}

void* OpenCLWorkspace::AllocDataSpace(Device dev, size_t size, size_t alignment,
                                      DLDataType type_hint) {
  this->Init();
  cl_device_id device_id = GetCLDeviceID(dev.device_id);
  auto platform = device_to_platform[device_id];
  cl_int err_code;
  cl::BufferDescriptor* desc = new cl::BufferDescriptor;
  // CL_INVALID_BUFFER_SIZE if size is 0.
  if (size == 0) {
    size = 1;
  }
  desc->buffer =
      clCreateBuffer(this->contexts[platform], CL_MEM_CREATE_FLAGS, size, nullptr, &err_code);
  desc->layout = cl::BufferDescriptor::MemoryLayout::kBuffer1D;
  OPENCL_CHECK_ERROR(err_code);
  return CreateHostPtrIfEnabled(desc, dev, size);
}

void* OpenCLWorkspace::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                      Optional<String> mem_scope) {
  if (!mem_scope.defined() || mem_scope.value().empty() || mem_scope.value() == "global") {
    return DeviceAPI::AllocDataSpace(dev, ndim, shape, dtype, mem_scope);
  }
  ICHECK(IsTextureStorage(std::string(mem_scope.value())))
      << "Device does not support allocate data space with "
      << "specified memory scope: " << mem_scope.value();

  ICHECK(ndim > 2) << "Shape for texture allocation must be at least rank 3; "
                   << "provided shape is rank " << ndim;

  cl::BufferDescriptor* desc = new cl::BufferDescriptor(mem_scope);
  size_t axis = DefaultTextureLayoutSeparator(ndim, mem_scope.value());
  auto texture = ApplyTexture2DFlattening<int64_t>(shape, ndim, axis);
  desc->buffer = AllocTexture(dev, texture.width, texture.height, dtype);
  return desc;
}

void* OpenCLWorkspace::GetNativePtr(const tvm::runtime::NDArray& narr) {
  cl::BufferDescriptor* desc = static_cast<cl::BufferDescriptor*>(narr.operator->()->data);
  return desc->host_ptr;
}

void OpenCLWorkspace::FreeDataSpace(Device dev, void* ptr) {
  // We have to make sure that the memory object is not in the command queue
  // for some OpenCL platforms.
  OPENCL_CALL(clFinish(this->GetQueue(dev)));

  cl::BufferDescriptor* desc = static_cast<cl::BufferDescriptor*>(ptr);
  if (desc->host_ptr) {
    clEnqueueUnmapMemObject(this->GetQueue(dev), desc->buffer,
                            reinterpret_cast<void*>(desc->host_ptr), 0, nullptr, nullptr);
  }
  OPENCL_CALL(clReleaseMemObject(desc->buffer));
  delete desc;
}

cl_mem OpenCLWorkspace::AllocTexture(Device dev, size_t width, size_t height,
                                     DLDataType type_hint) {
  this->Init();
  cl_device_id device_id = GetCLDeviceID(dev.device_id);
  auto platform = device_to_platform[device_id];
  cl_int err_code;
  cl_channel_type cl_type = DTypeToOpenCLChannelType(type_hint);
  cl_image_format format = {CL_RGBA, cl_type};
  cl_image_desc descriptor = {CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0};
  cl_mem mptr = clCreateImage(this->contexts[platform], CL_MEM_CREATE_FLAGS, &format, &descriptor,
                              nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

void* OpenCLWorkspace::AllocTextureWorkspace(Device dev, size_t width, size_t height,
                                             DLDataType type_hint) {
  return GetThreadEntry()->texture_pool.AllocTexture(dev, width, height, type_hint);
}

void OpenCLWorkspace::FreeTextureWorkspace(Device dev, void* ptr) {
  GetThreadEntry()->texture_pool.FreeTexture(dev, ptr);
}

void OpenCLWorkspace::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  this->Init();
  size_t nbytes = GetDataSize(*from);
  ICHECK_EQ(nbytes, GetDataSize(*to));
  ICHECK(IsContiguous(*from) && IsContiguous(*to))
      << "CopyDataFromTo only support contiguous array for now";

  if (IsOpenCLDevice(from->device) && IsOpenCLDevice(to->device)) {
    const auto* from_desc = static_cast<const cl::BufferDescriptor*>(from->data);
    auto* to_desc = static_cast<cl::BufferDescriptor*>(to->data);
    if (to_desc->layout == cl::BufferDescriptor::MemoryLayout::kBuffer1D &&
        from_desc->layout == cl::BufferDescriptor::MemoryLayout::kBuffer1D) {
      OPENCL_CALL(clEnqueueCopyBuffer(this->GetQueue(to->device), from_desc->buffer,
                                      to_desc->buffer, from->byte_offset, to->byte_offset, nbytes,
                                      0, nullptr, nullptr));
    } else if (to_desc->layout != cl::BufferDescriptor::MemoryLayout::kBuffer1D &&
               from_desc->layout == cl::BufferDescriptor::MemoryLayout::kBuffer1D) {
      auto image_info = GetImageInfo(to_desc, to);
      OPENCL_CALL(clEnqueueCopyBufferToImage(this->GetQueue(to->device), from_desc->buffer,
                                             to_desc->buffer, from->byte_offset, image_info.origin,
                                             image_info.region, 0, nullptr, nullptr));
    } else if (to_desc->layout == cl::BufferDescriptor::MemoryLayout::kBuffer1D &&
               from_desc->layout != cl::BufferDescriptor::MemoryLayout::kBuffer1D) {
      auto image_info = GetImageInfo(from_desc, from);
      OPENCL_CALL(clEnqueueCopyImageToBuffer(this->GetQueue(to->device), from_desc->buffer,
                                             to_desc->buffer, image_info.origin, image_info.region,
                                             to->byte_offset, 0, nullptr, nullptr));
    } else {
      auto to_image_info = GetImageInfo(to_desc, to);
      auto from_image_info = GetImageInfo(from_desc, from);
      OPENCL_CALL(clEnqueueCopyImage(this->GetQueue(to->device), from_desc->buffer, to_desc->buffer,
                                     from_image_info.origin, to_image_info.origin,
                                     to_image_info.region, 0, nullptr, nullptr));
    }
  } else if (IsOpenCLDevice(from->device) && to->device.device_type == kDLCPU) {
    const auto* from_desc = static_cast<const cl::BufferDescriptor*>(from->data);
    switch (from_desc->layout) {
      case cl::BufferDescriptor::MemoryLayout::kBuffer1D:
        OPENCL_CALL(clEnqueueReadBuffer(
            this->GetQueue(from->device), from_desc->buffer, CL_FALSE, from->byte_offset, nbytes,
            static_cast<char*>(to->data) + to->byte_offset, 0, nullptr, nullptr));
        break;
      case cl::BufferDescriptor::MemoryLayout::kImage2DActivation:
      case cl::BufferDescriptor::MemoryLayout::kImage2DWeight:
      case cl::BufferDescriptor::MemoryLayout::kImage2DNHWC:
        auto image_info = GetImageInfo(from_desc, from);
        // TODO(csullivan): Support calculating row_pitch correctly in the case of reuse.
        // Note that when utilizing texture pools for memory reuse, the allocated image
        // size can be larger than the size to be read.
        OPENCL_CALL(clEnqueueReadImage(
            this->GetQueue(from->device), from_desc->buffer, CL_FALSE, image_info.origin,
            image_info.region, image_info.row_pitch, image_info.slice_pitch,
            static_cast<char*>(to->data) + to->byte_offset, 0, nullptr, nullptr));
        break;
    }
    OPENCL_CALL(clFinish(this->GetQueue(from->device)));
  } else if (from->device.device_type == kDLCPU && IsOpenCLDevice(to->device)) {
    auto* to_desc = static_cast<cl::BufferDescriptor*>(to->data);
    switch (to_desc->layout) {
      case cl::BufferDescriptor::MemoryLayout::kBuffer1D:
        OPENCL_CALL(clEnqueueWriteBuffer(
            this->GetQueue(to->device), to_desc->buffer, CL_FALSE, to->byte_offset, nbytes,
            static_cast<const char*>(from->data) + from->byte_offset, 0, nullptr, nullptr));
        break;
      case cl::BufferDescriptor::MemoryLayout::kImage2DActivation:
      case cl::BufferDescriptor::MemoryLayout::kImage2DWeight:
      case cl::BufferDescriptor::MemoryLayout::kImage2DNHWC:
        auto image_info = GetImageInfo(to_desc, to);
        OPENCL_CALL(clEnqueueWriteImage(
            this->GetQueue(to->device), to_desc->buffer, CL_FALSE, image_info.origin,
            image_info.region, image_info.row_pitch, image_info.slice_pitch,
            static_cast<const char*>(from->data) + from->byte_offset, 0, nullptr, nullptr));
        break;
    }
    OPENCL_CALL(clFinish(this->GetQueue(to->device)));
  } else {
    LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
  }
}

void OpenCLWorkspace::StreamSync(Device dev, TVMStreamHandle stream) {
  this->Init();
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
  char* info = new char[ret_size];
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, ret_size, info, nullptr));
  std::string ret = info;
  delete[] info;
  return ret;
}

std::string GetOpenCLVersion(cl_device_id pid) {
  // String returned is "OpenCL $MAJOR.$MINOR $VENDOR_INFO".  To
  // match other implementations, we want to return "$MAJOR.$MINOR"
  std::string ret = GetDeviceInfo(pid, CL_DEVICE_VERSION);

  const size_t version_start = 7;  // Length of initial "OpenCL " prefix to skip
  const size_t version_end = ret.find(' ', version_start);
  return ret.substr(version_start, version_end - version_start);
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
  this->type_key = type_key;
  // matched platforms
  std::vector<cl_platform_id> platform_ids = cl::GetPlatformIDs();
  if (platform_ids.size() == 0) {
    LOG(WARNING) << "No OpenCL platform matched given existing options ...";
    return;
  }
  auto find_opencl_device = [&](const std::string& device_type, const std::string& platform_name) {
    std::unordered_map<cl_platform_id, std::vector<cl_device_id>> device_map;
    for (auto platform_id : platform_ids) {
      if (!MatchPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name)) {
        continue;
      }
      std::vector<cl_device_id> devices_matched = cl::GetDeviceIDs(platform_id, device_type);
      std::vector<cl_device_id> supported_devices = {};
      auto get_version_str = [](int version) {
        std::ostringstream out;
        out.precision(1);
        out << std::fixed << version / 100.f;
        return out.str();
      };
      for (auto& device : devices_matched) {
        std::string ver = GetOpenCLVersion(device);
        int opencl_version = std::stod(ver) * 100;
        if (opencl_version >= CL_TARGET_OPENCL_VERSION) {
          supported_devices.push_back(device);
        } else {
          std::string dev_msg = GetDeviceInfo(device, CL_DEVICE_NAME) +
                                " has OpenCL version == " + get_version_str(opencl_version);
          LOG(WARNING) << "TVM supports devices with OpenCL version >= "
                       << get_version_str(CL_TARGET_OPENCL_VERSION) << ", device " << dev_msg
                       << ". This device will be ignored.";

          if (noDevicesErrorMsg.empty()) {
            noDevicesErrorMsg =
                "Probably this error happen because TVM supports devices with OpenCL version >= " +
                get_version_str(CL_TARGET_OPENCL_VERSION) + ". We found the following devices:\n";
          }
          noDevicesErrorMsg += "\t" + dev_msg + "\n";
        }
      }
      if (supported_devices.size()) {
        device_map[platform_id] = supported_devices;
      }
    }
    return device_map;
  };
  auto device_map = find_opencl_device(device_type, platform_name);
  if ((device_map.size() == 0) && (device_type == "gpu")) {
    LOG(WARNING) << "Using CPU OpenCL device";
    device_map = find_opencl_device("cpu", "");
  }
  if (device_map.empty()) {
    LOG(WARNING) << "No OpenCL device";
    initialized_ = true;
    return;
  }
  ICHECK_EQ(this->queues.size(), 0U);
  cl_int err_code;
  for (auto& [platform, devices] : device_map) {
    this->platform_ids.push_back(platform);
    this->contexts[platform] =
        clCreateContext(nullptr, devices.size(), &(devices[0]), nullptr, nullptr, &err_code);
    this->devices.insert(this->devices.end(), devices.begin(), devices.end());
    for (size_t i = 0; i < devices.size(); ++i) {
      cl_device_id did = devices[i];
      device_to_platform[did] = platform;
      this->queues.push_back(clCreateCommandQueue(this->contexts[platform], did, 0, &err_code));
      OPENCL_CHECK_ERROR(err_code);
    }
    OPENCL_CHECK_ERROR(err_code);
  }
  this->events.resize(this->devices.size());
  initialized_ = true;
}

TVM_REGISTER_GLOBAL("device_api.opencl.alloc_nd").set_body([](TVMArgs args, TVMRetValue* rv) {
  int32_t device_type = args[0];
  int32_t device_id = args[1];
  int32_t dtype_code_hint = args[2];
  int32_t dtype_bits_hint = args[3];
  std::string scope = args[4];
  CHECK(scope.find("texture") != std::string::npos);
  int64_t ndim = args[5];
  CHECK_EQ(ndim, 2);
  int64_t* shape = static_cast<int64_t*>(static_cast<void*>(args[6]));
  int64_t width = shape[0];
  int64_t height = shape[1];

  Device dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;

  DLDataType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  OpenCLWorkspace* ptr = OpenCLWorkspace::Global();
  *rv = ptr->AllocTextureWorkspace(dev, static_cast<size_t>(width), static_cast<size_t>(height),
                                   type_hint);
});

TVM_REGISTER_GLOBAL("device_api.opencl.free_nd").set_body([](TVMArgs args, TVMRetValue* rv) {
  int32_t device_type = args[0];
  int32_t device_id = args[1];
  std::string scope = args[2];
  CHECK(scope.find("texture") != std::string::npos);
  void* data = args[3];
  OpenCLWorkspace* ptr = OpenCLWorkspace::Global();
  Device dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  ptr->FreeTextureWorkspace(dev, data);
  *rv = static_cast<int32_t>(0);
});

TVM_REGISTER_GLOBAL("device_api.opencl").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = OpenCLWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

TVM_REGISTER_OBJECT_TYPE(OpenCLTimerNode);

TVM_REGISTER_GLOBAL("profiling.timer.opencl").set_body_typed([](Device dev) {
  return Timer(make_object<OpenCLTimerNode>(dev));
});

}  // namespace cl
size_t OpenCLTimerNode::count_timer_execs = 0;
std::vector<size_t> OpenCLTimerNode::event_start_idxs;
}  // namespace runtime
}  // namespace tvm
