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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/timer.h>

#include <limits>
#include <sstream>

#include "../../../support/env.h"
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
  TVM_FFI_ICHECK(tensor->dtype.lanes == 1) << "Image dtype has lanes: " << tensor->dtype.lanes;

  info.origin[0] = info.origin[1] = info.origin[2] = 0;
  info.row_pitch = 0;
  info.slice_pitch = 0;

  size_t axis = DefaultTextureLayoutSeparator(
      tensor->ndim, cl::BufferDescriptor::ScopeFromMemoryLayout(desc->layout));
  auto texture_shape = ApplyTexture2DFlattening<int64_t>(tensor->shape, tensor->ndim, axis);
  info.region[0] = texture_shape.width;
  info.region[1] = texture_shape.height;
  info.region[2] = texture_shape.depth;
  return info;
}

cl::BufferDescriptor::MemoryLayout cl::BufferDescriptor::MemoryLayoutFromScope(
    ffi::Optional<ffi::String> mem_scope) {
  if (!mem_scope.has_value()) {
    return cl::BufferDescriptor::MemoryLayout::kBuffer1D;
  } else if (mem_scope.value() == "global.texture") {
    return cl::BufferDescriptor::MemoryLayout::kImage2DActivation;
  } else if (mem_scope.value() == "global.texture-weight") {
    return cl::BufferDescriptor::MemoryLayout::kImage2DWeight;
  } else if (mem_scope.value() == "global.texture-nhwc") {
    return cl::BufferDescriptor::MemoryLayout::kImage2DNHWC;
  }
  TVM_FFI_THROW(InternalError) << "No memory layout defined for memory of scope: "
                               << mem_scope.value();
}

ffi::String cl::BufferDescriptor::ScopeFromMemoryLayout(cl::BufferDescriptor::MemoryLayout layout) {
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
  TVM_FFI_THROW(InternalError) << "No scope corresponding to the provided memory layout: "
                               << static_cast<int>(layout);
  return "";
}

static size_t CheckedMultiply(size_t lhs, size_t rhs) {
  TVM_FFI_ICHECK(rhs == 0 || lhs <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) / rhs)
      << "OpenCL allocation size overflow";
  return lhs * rhs;
}

static size_t GetMemObjectSize(Device dev, int ndim, const int64_t* shape, DLDataType dtype) {
  TVM_FFI_ICHECK_GE(ndim, 0);
  TVM_FFI_ICHECK_GT(dtype.bits, 0);
  TVM_FFI_ICHECK_GT(dtype.lanes, 0);
  size_t count = 1;
  for (int i = ndim; i-- > 0;) {
    TVM_FFI_ICHECK_GE(shape[i], 0);
    count = CheckedMultiply(count, shape[i]);
  }
  uint64_t element_bits = uint64_t(dtype.bits) * dtype.lanes;
  TVM_FFI_ICHECK_LE(count, (std::numeric_limits<uint64_t>::max() - 7) / element_bits)
      << "OpenCL allocation size overflow";
  uint64_t nbytes = (uint64_t(count) * element_bits + 7) / 8;
  TVM_FFI_ICHECK_LE(nbytes, std::numeric_limits<size_t>::max());
  return static_cast<size_t>(nbytes);
}

static void CheckBufferCapacity(const BufferDescriptor* desc, size_t nbytes) {
  TVM_FFI_ICHECK(desc != nullptr);
  TVM_FFI_ICHECK(desc->layout == BufferDescriptor::MemoryLayout::kBuffer1D);
  size_t capacity;
  OPENCL_CALL(clGetMemObjectInfo(desc->buffer, CL_MEM_SIZE, sizeof(capacity), &capacity, nullptr));
  TVM_FFI_ICHECK_LE(nbytes, capacity) << "OpenCL view exceeds its backing buffer";
}

static size_t ImageRowPitch(size_t width, DLDataType dtype, size_t alignment) {
  size_t row_size = CheckedMultiply(width, CheckedMultiply((dtype.bits * dtype.lanes + 7) / 8, 4));
  alignment = std::max<size_t>(alignment, 1);
  TVM_FFI_ICHECK_LE(row_size,
                    static_cast<size_t>(std::numeric_limits<int64_t>::max()) - alignment + 1);
  return ((row_size + alignment - 1) / alignment) * alignment;
}

static Texture2DShape<int64_t> TextureShape(ffi::ShapeView shape,
                                            ffi::Optional<ffi::String> scope) {
  TVM_FFI_ICHECK_GE(shape.size(), 2);
  size_t count = 1;
  for (auto dim : shape) {
    TVM_FFI_ICHECK_GT(dim, 0) << "OpenCL textures require positive dimensions";
    count = CheckedMultiply(count, dim);
  }
  size_t axis = DefaultTextureLayoutSeparator(shape.size(), scope.value());
  return ApplyTexture2DFlattening<int64_t>(shape.data(), shape.size(), axis);
}

OpenCLThreadEntry* OpenCLWorkspace::GetThreadEntry() { return OpenCLThreadEntry::ThreadLocal(); }

OpenCLWorkspace* OpenCLWorkspace::Global() {
  static OpenCLWorkspace* inst = new OpenCLWorkspace();
  return inst;
}

cl_device_id OpenCLWorkspace::GetCLDeviceID(int device_id) {
  this->Init();
  TVM_FFI_ICHECK_LT(device_id, devices.size())
      << "Invalid device id " << device_id << ". " << GetError();
  return devices[device_id];
}

void OpenCLWorkspace::SetDevice(Device dev) { GetThreadEntry()->device.device_id = dev.device_id; }

ffi::Any OpenCLWorkspace::GetAttr(Device dev, const ffi::String& name) {
  this->Init();
  size_t index = static_cast<size_t>(dev.device_id);
  if (name == "exist") {
    return dev.device_type == kDLOpenCL && index < devices.size();
  }
  cl_device_id device_id = GetCLDeviceID(index);
  if (name == "max_threads_per_block") {
    size_t value;
    OPENCL_CALL(
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &value, nullptr));
    return static_cast<int64_t>(value);
  }
  if (name == "warp_size") {
    /* TODO: the warp size of OpenCL device is not always 1
             e.g. Intel Graphics has a sub group concept which contains 8 - 32 work items,
             corresponding to the number of SIMD entries the heardware configures.
             We need to figure out a way to query this information from the hardware.
    */
    const int warp_size = support::GetEnv("TVM_OPENCL_WARP_SIZE", 1);
    return warp_size;
  }
  if (name == "max_shared_memory_per_block") {
    cl_ulong value;
    OPENCL_CALL(
        clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &value, nullptr));
    return static_cast<int64_t>(value);
  }
  if (name == "compute_version") {
    return GetOpenCLVersion(device_id);
  }
  if (name == "device_name") {
    return GetDeviceInfo(device_id, CL_DEVICE_NAME);
  }
  if (name == "max_clock_rate") {
    cl_uint value;
    OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &value,
                                nullptr));
    // OpenCL returns the clock rate in MHz, while CUDA/ROCm return the
    // clock rate in kHz.  Converting to the same units for each.
    return static_cast<int32_t>(value * 1000);
  }
  if (name == "multi_processor_count") {
    cl_uint value;
    OPENCL_CALL(
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &value, nullptr));
    return static_cast<int32_t>(value);
  }
  if (name == "max_thread_dimensions") {
    size_t dims[3];
    OPENCL_CALL(
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims, nullptr));

    std::stringstream ss;  // use json string to return multiple int values;
    ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
    return ss.str();
  }
  if (name == "max_registers_per_block") {
    return {};
  }
  if (name == "gcn_arch") {
    return {};
  }
  if (name == "api_version") {
    return CL_TARGET_OPENCL_VERSION;
  }
  if (name == "driver_version") {
    char value[128] = {0};
    OPENCL_CALL(clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(value) - 1, value, nullptr));
    return std::string(value);
  }
  if (name == "l2_cache_size_bytes") {
    // NOTE(Zihao): this API cannot reflect the real L2 cache size in both CUDA/AMD GPUs.
    cl_ulong value;
    OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(value), &value,
                                nullptr));
    return static_cast<int64_t>(value);
  }
  if (name == "total_global_memory") {
    cl_ulong total_global_memory;
    OPENCL_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(total_global_memory),
                                &total_global_memory, nullptr));
    return static_cast<int64_t>(total_global_memory);
  }

  if (name == "available_global_memory") {
    // Not currently implemented.  Based on
    // https://stackoverflow.com/a/3568223, may not be implementable
    // at all through OpenCL API.
    return {};
  }
  if (name == "image_pitch_alignment") {
    return static_cast<int64_t>(device_info[device_id].image_row_align);
  }
  return {};
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

// Each workspace has one queue per device. Recycled buffers stay on that queue,
// whose in-order execution supplies the ordering required for asynchronous reuse.
class OpenCLTensorMemoryPool final : public TensorMemoryPool {
 public:
  explicit OpenCLTensorMemoryPool(OpenCLWorkspace* workspace) : workspace_(workspace) {}

 protected:
  ffi::Tensor AllocBacking(size_t nbytes, size_t alignment, DLDevice device) final {
    struct NDAlloc {
      OpenCLWorkspace* workspace;
      size_t alignment;
      void AllocData(DLTensor* tensor) {
        tensor->data = workspace->AllocCLBuffer(tensor->device, ffi::GetDataSize(*tensor),
                                                alignment, tensor->dtype);
      }
      void FreeData(DLTensor* tensor) { workspace->FreeDataSpace(tensor->device, tensor->data); }
    };
    auto allocate = [&]() {
      return ffi::Tensor::FromNDAlloc(NDAlloc{workspace_, alignment},
                                      {static_cast<int64_t>(nbytes)}, DLDataType{kDLUInt, 8, 1},
                                      device);
    };
    try {
      return allocate();
    } catch (const ffi::Error&) {
      // Free unused cached buffers before retrying a failed device allocation.
      Clear();
      return allocate();
    }
  }

 private:
  OpenCLWorkspace* workspace_;
};

ffi::ObjectPtr<TensorMemoryPool> OpenCLWorkspace::GetMemoryPool() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  if (!memory_pool_) {
    memory_pool_ = ffi::make_object<OpenCLTensorMemoryPool>(this);
  }
  return memory_pool_;
}

void OpenCLWorkspace::ClearMemoryPool() { GetMemoryPool()->Clear(); }

ffi::Tensor OpenCLWorkspace::Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                                   ffi::Optional<ffi::String> mem_scope) {
  TVM_FFI_ICHECK(IsOpenCLDevice(device));
  if (!mem_scope.has_value() || (*mem_scope).empty() || *mem_scope == "global") {
    return GetMemoryPool()->Empty(shape, dtype, device);
  }
  TextureShape(shape, mem_scope);
  GetMemObjectSize(device, shape.size(), shape.data(), dtype);
  struct NDAlloc {
    OpenCLWorkspace* workspace;
    ffi::Optional<ffi::String> scope;
    void AllocData(DLTensor* tensor) {
      tensor->data = workspace->AllocDataSpace(tensor->device, tensor->ndim, tensor->shape,
                                               tensor->dtype, scope);
    }
    void FreeData(DLTensor* tensor) { workspace->FreeDataSpace(tensor->device, tensor->data); }
  };
  return ffi::Tensor::FromNDAlloc(NDAlloc{this, mem_scope}, shape, dtype, device);
}

void* OpenCLWorkspace::AllocDataSpace(Device dev, size_t width, size_t height, size_t depth,
                                      DLDataType type_hint, ffi::Optional<ffi::String> mem_scope) {
  // Texture allocation given width and height.
  TVM_FFI_ICHECK_GT(width, 0);
  TVM_FFI_ICHECK_GT(height, 0);
  TVM_FFI_ICHECK_GT(depth, 0);
  size_t row_pitch = ImageRowPitch(width, type_hint, GetImageAlignment(dev.device_id));
  size_t mem_size = CheckedMultiply(CheckedMultiply(row_pitch, height), depth);

  // The image owns the pooled byte tensor, never a VM allocation record.
  ffi::Tensor backing;
  cl::BufferDescriptor* back_buffer = nullptr;
  if (IsBufferToImageSupported(dev.device_id)) {
    TVM_FFI_ICHECK_LE(mem_size, static_cast<size_t>(std::numeric_limits<int64_t>::max()));
    backing =
        GetMemoryPool()->Empty({static_cast<int64_t>(mem_size)}, DLDataType{kDLUInt, 8, 1}, dev);
    back_buffer = static_cast<cl::BufferDescriptor*>(backing.GetDLTensorPtr()->data);
  }
  if (!mem_scope.has_value()) {
    mem_scope = ffi::String("global.texture");
  }
  auto* image = static_cast<cl::BufferDescriptor*>(
      AllocCLImage(dev, back_buffer, width, height, depth, row_pitch, type_hint, mem_scope));
  image->backing = std::move(backing);
  return image;
}

void* OpenCLWorkspace::AllocDataSpace(Device dev, int ndim, const int64_t* shape, DLDataType dtype,
                                      ffi::Optional<ffi::String> mem_scope) {
  this->Init();
  if (!mem_scope.has_value() || (*mem_scope).empty() || (*mem_scope) == "global") {
    size_t size = GetMemObjectSize(dev, ndim, shape, dtype);
    return AllocCLBuffer(dev, size, kTempAllocaAlignment, dtype);
  }
  auto texture = TextureShape(ffi::ShapeView(shape, ndim), mem_scope);

  return AllocDataSpace(dev, texture.width, texture.height, texture.depth, dtype, mem_scope);
}

void* OpenCLWorkspace::AllocCLBuffer(Device dev, size_t size, size_t alignment,
                                     DLDataType type_hint) {
  this->Init();
  cl_device_id device_id = GetCLDeviceID(dev.device_id);
  auto platform = device_info[device_id].platform_id;
  cl_int err_code;
  auto desc = std::make_unique<cl::BufferDescriptor>();
  // CL_INVALID_BUFFER_SIZE if size is 0.
  if (size == 0) {
    size = 1;
  }
  desc->buffer =
      clCreateBuffer(this->contexts[platform], CL_MEM_CREATE_FLAGS, size, nullptr, &err_code);
  desc->layout = cl::BufferDescriptor::MemoryLayout::kBuffer1D;
  OPENCL_CHECK_ERROR(err_code);
  try {
    CreateHostPtrIfEnabled(desc.get(), dev, size);
  } catch (...) {
    clReleaseMemObject(desc->buffer);
    throw;
  }
  return desc.release();
}

void* OpenCLWorkspace::AllocCLImage(Device dev, void* back_buffer, size_t width, size_t height,
                                    size_t depth, size_t row_pitch, DLDataType type_hint,
                                    ffi::Optional<ffi::String> mem_scope) {
  this->Init();
  TVM_FFI_ICHECK(std::string(mem_scope.value()).find("texture") != std::string::npos)
      << "Expect texture scope while creating an Image object";
  cl::BufferDescriptor* back_desc = static_cast<cl::BufferDescriptor*>(back_buffer);
  cl_device_id device_id = GetCLDeviceID(dev.device_id);
  auto platform = device_info[device_id].platform_id;
  cl_int err_code;
  cl_channel_type cl_type = DTypeToOpenCLChannelType(type_hint);
  cl_image_format format = {CL_RGBA, cl_type};
  cl_image_desc descriptor = {CL_MEM_OBJECT_IMAGE2D_ARRAY, width, height, 0, depth, 0, 0, 0, 0};

  if (IsBufferToImageSupported(dev.device_id)) {
    CheckBufferCapacity(back_desc, CheckedMultiply(CheckedMultiply(row_pitch, height), depth));
    descriptor.image_row_pitch = row_pitch;
    descriptor.buffer = back_desc->buffer;
  }
  auto desc = std::make_unique<cl::BufferDescriptor>(mem_scope);
  cl_mem mptr = clCreateImage(this->contexts[platform], CL_MEM_CREATE_FLAGS, &format, &descriptor,
                              nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);

  desc->buffer = mptr;
  desc->back_buffer = back_desc;
  return desc.release();
}

void* OpenCLWorkspace::AllocDataSpaceView(Device dev, void* data, ffi::Shape shape,
                                          DLDataType dtype, ffi::Optional<ffi::String> mem_scope) {
  cl::BufferDescriptor* desc = static_cast<cl::BufferDescriptor*>(data);
  TVM_FFI_ICHECK(desc != nullptr);
  if (!mem_scope.has_value() || (*mem_scope).empty() || *mem_scope == "global") {
    size_t nbytes = GetMemObjectSize(dev, shape.size(), shape.data(), dtype);
    if (desc->layout == BufferDescriptor::MemoryLayout::kBuffer1D) {
      CheckBufferCapacity(desc, nbytes);
    } else if (IsBufferToImageSupported(dev.device_id)) {
      CheckBufferCapacity(desc->back_buffer, nbytes);
    }
  }

  // Fall back for devices w/o "cl_khr_image2d_from_buffer"
  if (!IsBufferToImageSupported(dev.device_id)) {
    cl::BufferDescriptor* ret_desc = desc;  // buffer -> buffer
    if (!mem_scope.has_value() || (*mem_scope).empty() || (*mem_scope) == "global") {
      if (desc->layout != cl::BufferDescriptor::MemoryLayout::kBuffer1D) {
        // image -> buffer
        size_t nbytes = GetMemObjectSize(dev, shape.size(), shape.data(), dtype);
        ret_desc = static_cast<cl::BufferDescriptor*>(
            OpenCLWorkspace::AllocCLBuffer(dev, nbytes, kTempAllocaAlignment, dtype));
        ret_desc->is_compat_view = true;
      }
    } else {
      // Any -> Image
      auto texture = TextureShape(shape, mem_scope);
      size_t row_pitch = ImageRowPitch(texture.width, dtype, GetImageAlignment(dev.device_id));

      ret_desc = static_cast<cl::BufferDescriptor*>(OpenCLWorkspace::Global()->AllocCLImage(
          dev, nullptr, texture.width, texture.height, texture.depth, row_pitch, dtype, mem_scope));
      ret_desc->is_compat_view = true;
    }
    return ret_desc;
  }

  if (!mem_scope.has_value() || (*mem_scope).empty() || (*mem_scope) == "global") {
    if (desc->layout == cl::BufferDescriptor::MemoryLayout::kBuffer1D) {
      //  buffer -> buffer
      return desc;
    } else {
      // image -> buffer
      return desc->back_buffer;
    }
  }
  auto texture = TextureShape(shape, mem_scope);
  size_t row_pitch = ImageRowPitch(texture.width, dtype, GetImageAlignment(dev.device_id));

  cl::BufferDescriptor* back_buffer;
  if (desc->back_buffer) {
    // image -> image
    back_buffer = desc->back_buffer;
  } else {
    // buffer -> image
    back_buffer = desc;
  }

  return (cl::BufferDescriptor*)AllocCLImage(dev, back_buffer, texture.width, texture.height,
                                             texture.depth, row_pitch, dtype, mem_scope);
}

void OpenCLWorkspace::FreeDataSpaceView(Device dev, void* ptr) {
  if (!ptr) return;
  auto* desc = static_cast<const cl::BufferDescriptor*>(ptr);
  // Handle the fall back
  if (!IsBufferToImageSupported(dev.device_id)) {
    if (desc->is_compat_view) {
      FreeDataSpace(dev, ptr);
    }
    return;
  }

  if (desc->layout != cl::BufferDescriptor::MemoryLayout::kBuffer1D) {
    OPENCL_CALL(clReleaseMemObject(desc->buffer));
    delete desc;
  }
}

void* OpenCLWorkspace::GetNativePtr(const tvm::ffi::Tensor& narr) {
  TVM_FFI_ICHECK(narr.defined());
  cl::BufferDescriptor* desc = static_cast<cl::BufferDescriptor*>(narr.GetDLTensorPtr()->data);
  return desc->host_ptr;
}

void OpenCLWorkspace::SetNativePtr(const tvm::ffi::Tensor& narr, void* host_ptr, size_t buf_size) {
  TVM_FFI_ICHECK(narr.defined());
  cl::BufferDescriptor* desc = static_cast<cl::BufferDescriptor*>(narr.GetDLTensorPtr()->data);

  this->Init();
  if (desc->layout == cl::BufferDescriptor::MemoryLayout::kBuffer1D) {
#ifdef USE_OPENCL_EXTN_QCOM
    Device dev = narr.GetDLTensorPtr()->device;
    cl_device_id device_id = GetCLDeviceID(dev.device_id);
    auto platform = device_info[device_id].platform_id;

    cl_int err_code;
    cl_mem replacement =
        clCreateBuffer(this->contexts[platform],
                       CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, buf_size,
                       host_ptr, &err_code);
    OPENCL_CHECK_ERROR(err_code);
    if (desc->host_ptr) {
      cl_int err =
          clEnqueueUnmapMemObject(this->GetQueue(dev), desc->buffer,
                                  reinterpret_cast<void*>(desc->host_ptr), 0, nullptr, nullptr);
      if (err != CL_SUCCESS) {
        clReleaseMemObject(replacement);
        OPENCL_CHECK_ERROR(err);
      }
      desc->host_ptr = nullptr;
    }
    clReleaseMemObject(desc->buffer);
    desc->buffer = replacement;
    // External host storage belongs to the caller. Do not cache this allocation
    // after its tensor dies, even if its original pooled capacity was larger.
    ClearMemoryPool();
#endif
  } else {
    TVM_FFI_THROW(InternalError) << "Native Ptr not enabled over image objects";
  }
}

void OpenCLWorkspace::SetPerfHint(Device dev, cl_uint perf_hint) {
#if defined(USE_OPENCL_EXTN_QCOM) && defined(CL_CONTEXT_PERF_HINT_QCOM)
  cl_device_id device_id = GetCLDeviceID(dev.device_id);
  auto platform = device_info[device_id].platform_id;
  OPENCL_CALL(clSetPerfHintQCOM(this->contexts[platform], perf_hint));
#endif
}

void OpenCLWorkspace::FreeDataSpace(Device dev, void* ptr) {
  auto* desc = static_cast<cl::BufferDescriptor*>(ptr);
  if (!desc) return;
  if (desc->host_ptr) {
    OPENCL_CALL(clEnqueueUnmapMemObject(this->GetQueue(dev), desc->buffer,
                                        reinterpret_cast<void*>(desc->host_ptr), 0, nullptr,
                                        nullptr));
  }
  OPENCL_CALL(clReleaseMemObject(desc->buffer));
  // Destroy the image first, then return its backing tensor to the pool.
  delete desc;
}

void OpenCLWorkspace::CopyDataFromTo(DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  this->Init();
  size_t nbytes = ffi::GetDataSize(*from);
  TVM_FFI_ICHECK_EQ(nbytes, ffi::GetDataSize(*to));
  TVM_FFI_ICHECK(ffi::IsContiguous(*from) && ffi::IsContiguous(*to))
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
    TVM_FFI_THROW(InternalError) << "Expect copy from/to OpenCL or between OpenCL";
  }
}

void OpenCLWorkspace::StreamSync(Device dev, TVMStreamHandle stream) {
  this->Init();
  TVM_FFI_ICHECK(stream == nullptr);
  OPENCL_CALL(clFinish(this->GetQueue(dev)));
}

OpenCLThreadEntry* OpenCLThreadEntry::ThreadLocal() {
  static thread_local OpenCLThreadEntry inst;
  return &inst;
}

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
  // ffi::String returned is "OpenCL $MAJOR.$MINOR $VENDOR_INFO".  To
  // match other implementations, we want to return "$MAJOR.$MINOR"
  std::string ret = GetDeviceInfo(pid, CL_DEVICE_VERSION);

  const size_t version_start = 7;  // Length of initial "OpenCL " prefix to skip
  const size_t version_end = ret.find(' ', version_start);
  return ret.substr(version_start, version_end - version_start);
}

std::vector<cl_platform_id> GetPlatformIDs() {
  cl_uint ret_size = 0;
  cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
  std::vector<cl_platform_id> ret;
  if (code == CL_PLATFORM_NOT_FOUND_KHR) return ret;
  OPENCL_CHECK_ERROR(code);
  if (ret_size == 0) return ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_device_id> GetDeviceIDs(cl_platform_id pid, std::string device_type) {
  cl_device_type dtype = CL_DEVICE_TYPE_ALL;
  if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
  if (device_type == "gpu") dtype = CL_DEVICE_TYPE_GPU;
  if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
  cl_uint ret_size = 0;
  cl_int code = clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size);
  std::vector<cl_device_id> ret;
  if (code == CL_DEVICE_NOT_FOUND) return ret;
  OPENCL_CHECK_ERROR(code);
  if (ret_size == 0) return ret;
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
                           const std::string& platform_name, cl_context_properties ctx_props[]) {
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
  TVM_FFI_ICHECK_EQ(this->queues.size(), 0U);
  cl_int err_code;
  for (auto& [platform, devices] : device_map) {
    this->platform_ids.push_back(platform);
    this->contexts[platform] =
        clCreateContext(ctx_props, devices.size(), &(devices[0]), nullptr, nullptr, &err_code);
    this->devices.insert(this->devices.end(), devices.begin(), devices.end());
    for (size_t i = 0; i < devices.size(); ++i) {
      cl_device_id did = devices[i];
      CLDeviceInfo dev_info;
      dev_info.platform_id = platform;
      this->queues.push_back(clCreateCommandQueue(this->contexts[platform], did, 0, &err_code));
      OPENCL_CHECK_ERROR(err_code);
      cl_uint row_pitch;
      OPENCL_CALL(clGetDeviceInfo(did, CL_DEVICE_IMAGE_PITCH_ALIGNMENT_KHR, sizeof(row_pitch),
                                  &row_pitch, nullptr));
      if (0 == row_pitch) {
        row_pitch = kAllocAlignment;  // Fallback
      }
      dev_info.image_row_align = row_pitch;
      dev_info.image_from_buffer_support =
          IsOpenCLExtensionSupported(did, "cl_khr_image2d_from_buffer");
      device_info.insert({did, dev_info});
    }
    OPENCL_CHECK_ERROR(err_code);
  }
  this->events.resize(this->devices.size());
  initialized_ = true;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("device_api.opencl.alloc_nd",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    int32_t device_type = args[0].cast<int32_t>();
                    int32_t device_id = args[1].cast<int32_t>();
                    auto scope = args[4].cast<std::string>();
                    TVM_FFI_ICHECK(scope.find("texture") != std::string::npos);
                    int64_t ndim = args[5].cast<int64_t>();
                    TVM_FFI_ICHECK_EQ(ndim, 3);
                    int64_t* shape = static_cast<int64_t*>(args[6].cast<void*>());
                    int64_t width = shape[0];
                    int64_t height = shape[1];
                    int64_t depth = shape[2];
                    int64_t channel_size = args[7].cast<int64_t>();
                    DLDataType channel_type = GetChannelType(channel_size);
                    Device dev;
                    dev.device_type = static_cast<DLDeviceType>(device_type);
                    dev.device_id = device_id;
                    DLDataType type_hint;
                    type_hint = channel_type;

                    *rv = OpenCLWorkspace::Global()->AllocDataSpace(
                        dev, static_cast<size_t>(width), static_cast<size_t>(height),
                        static_cast<size_t>(depth), type_hint, ffi::String("global.texture"));
                  })
      .def_packed("device_api.opencl.free_nd",
                  [](ffi::PackedArgs args, ffi::Any* rv) {
                    int32_t device_type = args[0].cast<int32_t>();
                    int32_t device_id = args[1].cast<int32_t>();
                    auto scope = args[2].cast<std::string>();
                    TVM_FFI_ICHECK(scope.find("texture") != std::string::npos);
                    void* data = args[3].cast<void*>();
                    OpenCLWorkspace* ptr = OpenCLWorkspace::Global();
                    Device dev;
                    dev.device_type = static_cast<DLDeviceType>(device_type);
                    dev.device_id = device_id;
                    ptr->FreeDataSpace(dev, data);
                    *rv = static_cast<int32_t>(0);
                  })
      .def_packed("device_api.opencl", [](ffi::PackedArgs args, ffi::Any* rv) {
        DeviceAPI* ptr = OpenCLWorkspace::Global();
        *rv = static_cast<void*>(ptr);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("runtime.timer.opencl", [](Device dev) {
    return Timer(ffi::UnsafeInit{}, ffi::make_object<OpenCLTimerObj>(dev));
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.opencl.clear_pool", []() { OpenCLWorkspace::Global()->ClearMemoryPool(); })
      .def("runtime.opencl.alloc_tensor",
           [](ffi::Shape shape, DLDataType dtype, DLDevice device,
              ffi::Optional<ffi::String> scope) {
             return OpenCLWorkspace::Global()->Empty(shape, dtype, device, scope);
           })
      .def("runtime.opencl.create_view", [](ffi::Tensor backing, ffi::Shape shape, DLDataType dtype,
                                            ffi::Optional<ffi::String> scope) {
        TVM_FFI_ICHECK(OpenCLWorkspace::Global()->IsOpenCLDevice(backing.device()));
        GetMemObjectSize(backing.device(), shape.size(), shape.data(), dtype);
        if (scope.has_value() && !(*scope).empty() && *scope != "global") {
          TextureShape(shape, scope);
        }
        TVM_FFI_ICHECK_EQ(backing.GetDLTensorPtr()->byte_offset, 0)
            << "OpenCL physical views require a zero byte offset";
        struct NDAlloc {
          ffi::Tensor backing;
          ffi::Optional<ffi::String> scope;
          void AllocData(DLTensor* tensor) {
            tensor->data = OpenCLWorkspace::Global()->AllocDataSpaceView(
                tensor->device, backing.GetDLTensorPtr()->data,
                ffi::Shape(tensor->shape, tensor->shape + tensor->ndim), tensor->dtype, scope);
          }
          void FreeData(DLTensor* tensor) {
            if (tensor->data != backing.GetDLTensorPtr()->data) {
              OpenCLWorkspace::Global()->FreeDataSpaceView(tensor->device, tensor->data);
            }
          }
        };
        auto device = backing.device();
        return ffi::Tensor::FromNDAlloc(NDAlloc{std::move(backing), scope}, shape, dtype, device);
      });
}

}  // namespace cl
size_t OpenCLTimerObj::count_timer_execs = 0;
std::vector<size_t> OpenCLTimerObj::event_start_idxs;
}  // namespace runtime
}  // namespace tvm
