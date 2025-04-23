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
 * \file ndarray.cc
 * \brief NDArray container infratructure.
 */
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include "runtime_base.h"
#include "tvm/runtime/data_type.h"

namespace tvm {
namespace runtime {

inline void VerifyDataType(DLDataType dtype) {
  ICHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    ICHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    // allow int1/uint4/int4
    else if (dtype.bits == 1 && dtype.code == kDLInt)
      return;
    else if (dtype.bits == 4 && dtype.code == kDLUInt)
      return;
    else if (dtype.bits == 4 && dtype.code == kDLInt)
      return;
    else if (dtype.bits == 4 && dtype.code == DataType::kFloat4_e2m1fn)
      return;
    else
      ICHECK_EQ(dtype.bits % 8, 0);
  }
  ICHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

void ArrayCopyFromBytes(DLTensor* handle, const void* data, size_t nbytes) {
  size_t arr_size = GetDataSize(*handle);
  ICHECK_EQ(arr_size, nbytes) << "ArrayCopyFromBytes: size mismatch";
  ICHECK(IsContiguous(*handle)) << "ArrayCopyFromBytes only support contiguous array for now";

  DLTensor from;
  from.data = const_cast<void*>(data);
  from.device = Device{kDLCPU, 0};
  from.ndim = handle->ndim;
  from.dtype = handle->dtype;
  from.shape = handle->shape;
  from.strides = nullptr;
  from.byte_offset = 0;
  DeviceAPI::Get(handle->device)->CopyDataFromTo(&from, handle, nullptr);
  // Synchronize in case data become unavailable later.
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, nullptr);
}

void ArrayCopyToBytes(const DLTensor* handle, void* data, size_t nbytes) {
  size_t arr_size = GetDataSize(*handle);
  ICHECK_EQ(arr_size, nbytes) << "ArrayCopyToBytes: size mismatch";
  ICHECK(IsContiguous(*handle)) << "ArrayCopyToBytes only support contiguous array for now";

  DLTensor to;
  to.data = const_cast<void*>(data);
  to.device = Device{kDLCPU, 0};
  to.ndim = handle->ndim;
  to.dtype = handle->dtype;
  to.shape = handle->shape;
  to.strides = nullptr;
  to.byte_offset = 0;

  DeviceAPI::Get(handle->device)->CopyDataFromTo(const_cast<DLTensor*>(handle), &to, nullptr);
  // Synchronize in case data become unavailable later.
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, nullptr);
}

NDArray NDArray::Empty(ShapeTuple shape, DLDataType dtype, Device dev, Optional<String> mem_scope) {
  struct DeviceAPIAlloc {
    void AllocData(DLTensor* tensor, ffi::Optional<ffi::String> mem_scope) {
      tensor->data = DeviceAPI::Get(tensor->device)
                         ->AllocDataSpace(tensor->device, tensor->ndim, tensor->shape,
                                          tensor->dtype, mem_scope);
    }
    void FreeData(DLTensor* tensor) {
      DeviceAPI::Get(tensor->device)->FreeDataSpace(tensor->device, tensor->data);
    }
  };
  return ffi::NDArray::FromNDAlloc(DeviceAPIAlloc(), shape, dtype, dev, mem_scope);
}

struct NDArray::Internal {
  // Implementation of API function
  static DLTensor* MoveToFFIHandle(NDArray arr) {
    DLTensor* handle = NDArray::FFIGetHandle(arr);
    // move and discard as handle is already obtained in FFIGetHandle
    ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(arr));
    return handle;
  }
  static void FFIDecRef(TVMArrayHandle tensor) { NDArray::FFIDecRef(tensor); }
};

NDArray NDArray::CreateView(ShapeTuple shape, DLDataType dtype,
                            uint64_t relative_byte_offset) const {
  ICHECK(data_ != nullptr);

  const DLTensor& orig = *get_mutable();
  CHECK(IsContiguous()) << [&orig]() {
    std::stringstream ss;
    ss << "Can only create view for compact tensor, but found strides ";

    ss << "[";
    for (int i = 0; i < orig.ndim; i++) {
      if (i) ss << ", ";
      ss << orig.strides[i];
    }
    ss << "]";

    ss << ", for shape ";
    ss << "[";
    for (int i = 0; i < orig.ndim; i++) {
      if (i) ss << ", ";
      ss << orig.shape[i];
    }
    ss << "]";
    return ss.str();
  }();
  const auto& curr_dl_tensor = *get_mutable();
  size_t curr_size = GetDataSize(curr_dl_tensor);
  size_t view_size = ffi::GetPackedDataSize(shape.Product(), dtype);
  CHECK_LE(relative_byte_offset + view_size, curr_size)
      << "ValueError: "
      << "View with shape " << shape << " and datatype " << dtype << " would have a size of "
      << view_size << " bytes.  "
      << "This would occupy bytes " << relative_byte_offset << " <= i_byte < "
      << (relative_byte_offset + view_size) << " within the backing array.  "
      << "However, the NDArray being viewed only contains " << curr_size << " bytes (shape = "
      << ShapeTuple(curr_dl_tensor.shape, curr_dl_tensor.shape + curr_dl_tensor.ndim)
      << ", dtype= " << curr_dl_tensor.dtype << ").";

  // helper allocator class that retains ref count of original NDArray
  class ViewBasedAlloc {
   public:
    ViewBasedAlloc(NDArray source) : source_(source) {}
    void AllocData(DLTensor* tensor, int64_t byte_offset) {
      tensor->data = source_.get_mutable()->data;
      tensor->byte_offset = byte_offset;
    }

    void FreeData(DLTensor* tensor) {}

   private:
    NDArray source_;
  };

  NDArray ret = NDArray::FromNDAlloc(ViewBasedAlloc(NDArray(*this)), shape, dtype, (*this)->device,
                                     curr_dl_tensor.byte_offset + relative_byte_offset);
  return ret;
}

void NDArray::CopyToBytes(void* data, size_t nbytes) const {
  ICHECK(data != nullptr);
  ICHECK(data_ != nullptr);
  ArrayCopyToBytes(get_mutable(), data, nbytes);
}

void NDArray::CopyFromBytes(const void* data, size_t nbytes) {
  ICHECK(data != nullptr);
  ICHECK(data_ != nullptr);
  ArrayCopyFromBytes(get_mutable(), data, nbytes);
}

NDArray NDArray::CopyTo(const Device& dev, Optional<String> mem_scope) const {
  ICHECK(data_ != nullptr);
  const DLTensor* dptr = operator->();
  NDArray ret =
      Empty(ShapeTuple(dptr->shape, dptr->shape + dptr->ndim), dptr->dtype, dev, mem_scope);
  this->CopyTo(ret);
  Device copy_gpu_dev = dptr->device.device_type != kDLCPU ? dptr->device : dev;
  DeviceAPI::Get(copy_gpu_dev)->StreamSync(copy_gpu_dev, nullptr);
  return ret;
}

void NDArray::CopyFromTo(const DLTensor* from, DLTensor* to, TVMStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  ICHECK_EQ(from_size, to_size) << "TVMArrayCopyFromTo: The size in bytes must exactly match.";

  ICHECK(from->device.device_type == to->device.device_type || from->device.device_type == kDLCPU ||
         to->device.device_type == kDLCPU || from->device.device_type == kDLCUDAHost ||
         to->device.device_type == kDLCUDAHost || from->device.device_type == kDLROCMHost ||
         to->device.device_type == kDLROCMHost)
      << "Can not copy across different device types directly. From device type: "
      << from->device.device_type << " to device type: " << to->device.device_type;

  // Use the device that is *not* a cpu device to get the correct device
  // api manager.
  Device dev = from->device.device_type != kDLCPU ? from->device : to->device;

  DeviceAPI::Get(dev)->CopyDataFromTo(const_cast<DLTensor*>(from), to, stream);
}

}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

int TVMArrayGetTypeIndex(TVMArrayHandle handle, unsigned* out_tindex) {
  API_BEGIN();
  *out_tindex =
      tvm::ffi::details::ObjectUnsafe::GetHeader(TVMArrayHandleToObjectHandle(handle))->type_index;
  API_END();
}

int TVMArrayAlloc(const tvm_index_t* shape, int ndim, int dtype_code, int dtype_bits,
                  int dtype_lanes, int device_type, int device_id, TVMArrayHandle* out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  tvm::Device dev;
  dev.device_type = static_cast<DLDeviceType>(device_type);
  dev.device_id = device_id;
  auto ndarray = NDArray::Empty(ShapeTuple(shape, shape + ndim), dtype, dev);

  *out = NDArray::Internal::MoveToFFIHandle(ndarray);
  API_END();
}

TVM_REGISTER_GLOBAL("runtime.TVMArrayAllocWithScope").set_body_typed(NDArray::Empty);

TVM_REGISTER_GLOBAL("runtime.TVMArrayCreateView").set_body_method(&NDArray::CreateView);

int TVMArrayFree(TVMArrayHandle handle) {
  API_BEGIN();
  NDArray::Internal::FFIDecRef(handle);
  API_END();
}

int TVMArrayCopyFromTo(TVMArrayHandle from, TVMArrayHandle to, TVMStreamHandle stream) {
  API_BEGIN();
  NDArray::CopyFromTo(from, to, stream);
  API_END();
}

int TVMArrayFromDLPack(DLManagedTensor* from, TVMArrayHandle* out) {
  API_BEGIN();
  *out = NDArray::Internal::MoveToFFIHandle(NDArray::FromDLPack(from));
  API_END();
}

int TVMArrayToDLPack(TVMArrayHandle from, DLManagedTensor** out) {
  API_BEGIN();
  *out = static_cast<tvm::ffi::NDArrayObj*>(TVMArrayHandleToObjectHandle(from))->ToDLPack();
  API_END();
}

int TVMArrayCopyFromBytes(TVMArrayHandle handle, void* data, size_t nbytes) {
  API_BEGIN();
  ArrayCopyFromBytes(handle, data, nbytes);
  API_END();
}

int TVMArrayCopyToBytes(TVMArrayHandle handle, void* data, size_t nbytes) {
  API_BEGIN();
  ArrayCopyToBytes(handle, data, nbytes);
  API_END();
}
