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

extern "C" {
// C-mangled dlpack deleter.
static void TVMNDArrayDLPackDeleter(DLManagedTensor* tensor);
// helper function to get NDArray's type index, only used by ctypes.
TVM_DLL int TVMArrayGetTypeIndex(TVMArrayHandle handle, unsigned* out_tindex);
}

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

struct NDArray::Internal {
  // Default deleter for the container
  static void DefaultDeleter(Object* ptr_obj) {
    auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
    if (ptr->manager_ctx != nullptr) {
      static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
    } else if (ptr->dl_tensor.data != nullptr) {
      tvm::runtime::DeviceAPI::Get(ptr->dl_tensor.device)
          ->FreeDataSpace(ptr->dl_tensor.device, ptr->dl_tensor.data);
    }
    delete ptr;
  }
  // Deleter for NDArray converted from DLPack
  // This is used from data which is passed from external DLPack(DLManagedTensor)
  // that are not allocated inside of TVM.
  // This enables us to create NDArray from memory allocated by other
  // frameworks that are DLPack compatible
  static void DLPackDeleter(Object* ptr_obj) {
    auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete ptr;
  }
  // Deleter for NDArray based on external DLTensor
  // The memory is allocated from outside and it is assumed that
  // responsibility for its freeing is also outside
  static void SelfDeleter(Object* ptr_obj) {
    auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
    delete ptr;
  }
  // Local create function which allocates tensor metadata
  // but does not allocate space for the data.
  static NDArray Create(ShapeTuple shape, DLDataType dtype, Device dev) {
    VerifyDataType(dtype);

    // critical zone: construct header
    NDArray::Container* data = new NDArray::Container();
    data->SetDeleter(DefaultDeleter);

    // RAII now in effect
    NDArray ret(GetObjectPtr<Object>(data));
    // setup shape
    data->shape_ = std::move(shape);
    data->dl_tensor.shape = const_cast<ShapeTuple::index_type*>(data->shape_.data());
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup dtype
    data->dl_tensor.dtype = dtype;
    // setup device
    data->dl_tensor.device = dev;
    return ret;
  }
  // Implementation of API function
  static DLTensor* MoveToFFIHandle(NDArray arr) {
    DLTensor* handle = NDArray::FFIGetHandle(arr);
    ObjectRef::FFIClearAfterMove(&arr);
    return handle;
  }
  static void FFIDecRef(TVMArrayHandle tensor) { NDArray::FFIDecRef(tensor); }
  // Container to DLManagedTensor
  static DLManagedTensor* ToDLPack(TVMArrayHandle handle) {
    auto* from =
        static_cast<NDArray::Container*>(reinterpret_cast<NDArray::ContainerBase*>(handle));
    return ToDLPack(from);
  }

  static DLManagedTensor* ToDLPack(NDArray::Container* from) {
    ICHECK(from != nullptr);
    DLManagedTensor* ret = new DLManagedTensor();
    ret->dl_tensor = from->dl_tensor;
    ret->manager_ctx = from;
    from->IncRef();
    ret->deleter = TVMNDArrayDLPackDeleter;
    return ret;
  }
  // Delete dlpack object.
  static void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
    static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
    delete tensor;
  }
};

NDArray NDArray::CreateView(ShapeTuple shape, DLDataType dtype, uint64_t relative_byte_offset) {
  ICHECK(data_ != nullptr);

  const DLTensor& orig = get_mutable()->dl_tensor;
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

  const auto& curr_dl_tensor = get_mutable()->dl_tensor;

  NDArray ret = Internal::Create(shape, dtype, curr_dl_tensor.device);

  size_t curr_size = GetDataSize(this->get_mutable()->dl_tensor);
  size_t view_size = GetDataSize(ret.get_mutable()->dl_tensor);
  CHECK_LE(relative_byte_offset + view_size, curr_size)
      << "ValueError: "
      << "View with shape " << shape << " and datatype " << dtype << " would have a size of "
      << view_size << " bytes.  "
      << "This would occupy bytes " << relative_byte_offset << " <= i_byte < "
      << (relative_byte_offset + view_size) << " within the backing array.  "
      << "However, the NDArray being viewed only contains " << curr_size << " bytes (shape = "
      << ShapeTuple(curr_dl_tensor.shape, curr_dl_tensor.shape + curr_dl_tensor.ndim)
      << ", dtype= " << curr_dl_tensor.dtype << ").";

  // increase ref count
  get_mutable()->IncRef();
  ret.get_mutable()->manager_ctx = get_mutable();
  ret.get_mutable()->dl_tensor.data = get_mutable()->dl_tensor.data;
  ret.get_mutable()->dl_tensor.byte_offset =
      get_mutable()->dl_tensor.byte_offset + relative_byte_offset;
  return ret;
}

DLManagedTensor* NDArray::ToDLPack() const { return Internal::ToDLPack(get_mutable()); }

NDArray NDArray::Empty(ShapeTuple shape, DLDataType dtype, Device dev, Optional<String> mem_scope) {
  NDArray ret = Internal::Create(shape, dtype, dev);
  ret.get_mutable()->dl_tensor.data =
      DeviceAPI::Get(ret->device)
          ->AllocDataSpace(ret->device, shape.size(), shape.data(), ret->dtype, mem_scope);
  return ret;
}

NDArray NDArray::FromExternalDLTensor(const DLTensor& dl_tensor) {
  ICHECK(::tvm::runtime::IsContiguous(dl_tensor)) << "External DLTensor must be contiguous.";
  ICHECK(IsAligned(dl_tensor)) << "Data in DLTensor is not aligned as required by NDArray";
  NDArray::Container* data = new NDArray::Container();

  data->SetDeleter(Internal::SelfDeleter);
  data->dl_tensor = dl_tensor;
  std::vector<ShapeTuple::index_type> shape;
  shape.resize(data->dl_tensor.ndim);
  shape.assign(data->dl_tensor.shape, data->dl_tensor.shape + data->dl_tensor.ndim);
  data->shape_ = ShapeTuple(shape);
  data->dl_tensor.shape = const_cast<ShapeTuple::index_type*>(data->shape_.data());

  return NDArray(GetObjectPtr<Object>(data));
}

NDArray NDArray::NewFromDLTensor(DLTensor* tensor, const Device& dev) {
  ICHECK(::tvm::runtime::IsContiguous(*tensor))
      << "DLTensor is not contiguous. Copying from non-contiguous data is currently not supported";
  std::vector<int64_t> shape;
  for (int64_t i = 0; i < tensor->ndim; i++) {
    shape.push_back(tensor->shape[i]);
  }
  NDArray ary = NDArray::Empty(shape, tensor->dtype, dev);
  ary.CopyFrom(tensor);
  return ary;
}

NDArray NDArray::FromDLPack(DLManagedTensor* tensor) {
  NDArray::Container* data = new NDArray::Container();
  // construct header
  data->SetDeleter(Internal::DLPackDeleter);
  // fill up content.
  data->manager_ctx = tensor;
  ICHECK(::tvm::runtime::IsContiguous(tensor->dl_tensor)) << "DLManagedTensor must be contiguous.";
  ICHECK(IsAligned(tensor->dl_tensor))
      << "Data in DLManagedTensor is not aligned as required by NDArray";
  data->dl_tensor = tensor->dl_tensor;
  // update shape_
  std::vector<ShapeTuple::index_type> shape;
  shape.resize(data->dl_tensor.ndim);
  shape.assign(data->dl_tensor.shape, data->dl_tensor.shape + data->dl_tensor.ndim);
  data->shape_ = ShapeTuple(shape);
  data->dl_tensor.shape = const_cast<ShapeTuple::index_type*>(data->shape_.data());
  return NDArray(GetObjectPtr<Object>(data));
}

void NDArray::CopyToBytes(void* data, size_t nbytes) const {
  ICHECK(data != nullptr);
  ICHECK(data_ != nullptr);
  ArrayCopyToBytes(&get_mutable()->dl_tensor, data, nbytes);
}

void NDArray::CopyFromBytes(const void* data, size_t nbytes) {
  ICHECK(data != nullptr);
  ICHECK(data_ != nullptr);
  ArrayCopyFromBytes(&get_mutable()->dl_tensor, data, nbytes);
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

ShapeTuple NDArray::Shape() const {
  return static_cast<const NDArray::Container*>(data_.get())->shape_;
}

runtime::DataType NDArray::DataType() const {
  return runtime::DataType(get_mutable()->dl_tensor.dtype);
}

bool NDArray::AbilityOfZeroCopyForDLTensor(DLTensor* tensor, const Device& dev) {
  bool device_check = (dev.device_type == tensor->device.device_type);
  bool device_id_check = (dev.device_id == tensor->device.device_id);
  bool alignment_check = IsAligned(*tensor);
  return device_check && device_id_check && alignment_check;
}

bool NDArray::IsAligned(const DLTensor& tensor) {
  return (reinterpret_cast<size_t>(static_cast<char*>(tensor.data) + tensor.byte_offset) %
              tvm::runtime::kAllocAlignment ==
          0);
}

TVM_REGISTER_OBJECT_TYPE(NDArray::Container);

}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

void TVMNDArrayDLPackDeleter(DLManagedTensor* tensor) {
  NDArray::Internal::NDArrayDLPackDeleter(tensor);
}

int TVMArrayGetTypeIndex(TVMArrayHandle handle, unsigned* out_tindex) {
  API_BEGIN();
  *out_tindex = TVMArrayHandleToObjectHandle(handle)->type_index();
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
  *out = NDArray::Internal::ToDLPack(from);
  API_END();
}

void TVMDLManagedTensorCallDeleter(DLManagedTensor* dltensor) { (*(dltensor->deleter))(dltensor); }

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
