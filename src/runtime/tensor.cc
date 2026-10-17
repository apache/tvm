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
 * \file tensor.cc
 * \brief Tensor container infrastructure.
 */
#include <tvm/ffi/endian.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/base64.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/tensor.h>
#include <tvm/support/bytes_io.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "tvm/ffi/dtype.h"

namespace tvm {
namespace runtime {

void CopyToBytes(ffi::TensorView source, void* data, size_t nbytes, TVMStreamHandle stream) {
  const DLTensor* handle = ffi::AnyView(source).cast<DLTensor*>();
  TVM_FFI_ICHECK(data != nullptr || nbytes == 0);
  size_t arr_size = ffi::GetDataSize(*handle);
  TVM_FFI_ICHECK_EQ(arr_size, nbytes) << "CopyToBytes: size mismatch";
  TVM_FFI_ICHECK(ffi::IsContiguous(*handle)) << "CopyToBytes only supports contiguous tensors";

  DLTensor to;
  to.data = const_cast<void*>(data);
  to.device = Device{kDLCPU, 0};
  to.ndim = handle->ndim;
  to.dtype = handle->dtype;
  to.shape = handle->shape;
  to.strides = nullptr;
  to.byte_offset = 0;

  DeviceAPI::Get(handle->device)->CopyDataFromTo(const_cast<DLTensor*>(handle), &to, stream);
  // Synchronize in case data become unavailable later.
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, stream);
}

void CopyFromBytes(ffi::TensorView destination, const void* data, size_t nbytes,
                   TVMStreamHandle stream) {
  const DLTensor* handle = ffi::AnyView(destination).cast<DLTensor*>();
  TVM_FFI_ICHECK(data != nullptr || nbytes == 0);
  size_t arr_size = ffi::GetDataSize(*handle);
  TVM_FFI_ICHECK_EQ(arr_size, nbytes) << "CopyFromBytes: size mismatch";
  TVM_FFI_ICHECK(ffi::IsContiguous(*handle)) << "CopyFromBytes only supports contiguous tensors";

  DLTensor from;
  from.data = const_cast<void*>(data);
  from.device = Device{kDLCPU, 0};
  from.ndim = handle->ndim;
  from.dtype = handle->dtype;
  from.shape = handle->shape;
  from.strides = nullptr;
  from.byte_offset = 0;

  DeviceAPI::Get(handle->device)->CopyDataFromTo(&from, const_cast<DLTensor*>(handle), stream);
  // Synchronize in case data become unavailable later.
  DeviceAPI::Get(handle->device)->StreamSync(handle->device, stream);
}

ffi::Tensor Empty(ffi::Shape shape, DLDataType dtype, Device dev,
                  ffi::Optional<MemoryScope> mem_scope) {
  return DeviceAPI::Get(dev)->Empty(shape, dtype, dev, mem_scope);
}

ffi::ObjectPtr<TensorMemoryPool> CreateTensorMemoryPool(DLDevice device) {
  auto factory = ffi::Function::GetGlobal(std::string("runtime.create_tensor_memory_pool.") +
                                          DLDeviceType2Str(device.device_type));
  if (factory.has_value()) {
    auto pool = (*factory)(device).cast<ffi::ObjectPtr<TensorMemoryPool>>();
    TVM_FFI_CHECK(pool != nullptr, ValueError) << "Tensor memory pool factory returned null";
    return pool;
  }
  return ffi::make_object<TensorMemoryPool>();
}

ffi::Tensor CreateView(ffi::Tensor source, ffi::Shape shape, DLDataType dtype,
                       uint64_t relative_byte_offset) {
  TVM_FFI_ICHECK(source.defined());

  const DLTensor& orig = *source.GetDLTensorPtr();
  TVM_FFI_ICHECK(source.IsContiguous()) << [&orig]() {
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
  const auto& curr_dl_tensor = *source.GetDLTensorPtr();
  size_t curr_size = ffi::GetDataSize(curr_dl_tensor);
  size_t view_size = ffi::GetDataSize(shape.Product(), dtype);
  TVM_FFI_CHECK(relative_byte_offset <= curr_size && view_size <= curr_size - relative_byte_offset,
                ValueError)
      << "View with shape " << shape << " and datatype " << dtype << " would have a size of "
      << view_size << " bytes.  "
      << "This would occupy bytes " << relative_byte_offset << " <= i_byte < "
      << (relative_byte_offset + view_size) << " within the backing array.  "
      << "However, the Tensor being viewed only contains " << curr_size << " bytes (shape = "
      << ffi::Shape(curr_dl_tensor.shape, curr_dl_tensor.shape + curr_dl_tensor.ndim)
      << ", dtype= " << curr_dl_tensor.dtype << ").";

  TVM_FFI_CHECK_LE(relative_byte_offset,
                   std::numeric_limits<uint64_t>::max() - curr_dl_tensor.byte_offset, ValueError)
      << "View byte offset overflows the tensor offset range.";

  // helper allocator class that retains ref count of original Tensor
  class ViewBasedAlloc {
   public:
    explicit ViewBasedAlloc(ffi::Tensor source) : source_(source) {}
    void AllocData(DLTensor* tensor, int64_t byte_offset) {
      tensor->data = source_.GetDLTensorPtr()->data;
      tensor->byte_offset = byte_offset;
    }

    void FreeData(DLTensor* tensor) {}

   private:
    ffi::Tensor source_;
  };

  ffi::Tensor ret = ffi::Tensor::FromNDAlloc(ViewBasedAlloc(source), shape, dtype, source.device(),
                                             curr_dl_tensor.byte_offset + relative_byte_offset);
  return ret;
}

ffi::Tensor CopyTo(ffi::TensorView source, Device dev, ffi::Optional<MemoryScope> mem_scope) {
  ffi::Tensor ret = Empty(ffi::Shape(source.shape()), source.dtype(), dev, mem_scope);
  CopyFromTo(source, ret);
  Device copy_gpu_dev = source.device().device_type != kDLCPU ? source.device() : dev;
  DeviceAPI::Get(copy_gpu_dev)->StreamSync(copy_gpu_dev, nullptr);
  return ret;
}

void CopyFromTo(ffi::TensorView source, ffi::TensorView destination, TVMStreamHandle stream) {
  const DLTensor* from = ffi::AnyView(source).cast<DLTensor*>();
  const DLTensor* to = ffi::AnyView(destination).cast<DLTensor*>();
  size_t from_size = ffi::GetDataSize(*from);
  size_t to_size = ffi::GetDataSize(*to);
  TVM_FFI_ICHECK_EQ(from_size, to_size) << "CopyFromTo: The size in bytes must exactly match.";

  TVM_FFI_ICHECK(from->device.device_type == to->device.device_type ||
                 from->device.device_type == kDLCPU || to->device.device_type == kDLCPU ||
                 from->device.device_type == kDLCUDAHost || to->device.device_type == kDLCUDAHost ||
                 from->device.device_type == kDLROCMHost || to->device.device_type == kDLROCMHost)
      << "Can not copy across different device types directly. From device type: "
      << from->device.device_type << " to device type: " << to->device.device_type;

  // Use the device that is *not* a cpu device to get the correct device
  // api manager.
  Device dev = from->device.device_type != kDLCPU ? from->device : to->device;

  DeviceAPI::Get(dev)->CopyDataFromTo(const_cast<DLTensor*>(from), const_cast<DLTensor*>(to),
                                      stream);
}

namespace {

/*! \brief Swap the byte order of every scalar lane stored in a packed buffer. */
void SwapTensorBytes(void* data, DLDataType dtype, size_t nbytes) {
  // Sub-byte and single-byte lanes have no byte order.
  if (dtype.bits <= 8) return;
  size_t lane_bytes = dtype.bits / 8;
  ffi::ByteSwap(data, lane_bytes, nbytes / lane_bytes);
}

}  // namespace

void SaveTensor(ffi::TensorView source, support::Stream* strm) {
  const DLTensor* tensor = ffi::AnyView(source).cast<DLTensor*>();
  uint64_t header = kTVMTensorMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  // Serialized tensors are always recorded as CPU tensors, so they can be loaded on
  // machines without the source device. Callers copy the result to the target device.
  Device cpu_dev{kDLCPU, 0};
  strm->Write(cpu_dev);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  strm->WriteArray(tensor->shape, tensor->ndim);
  size_t data_size = ffi::GetDataSize(*tensor);
  int64_t data_byte_size = static_cast<int64_t>(data_size);
  strm->Write(data_byte_size);

  if (TVM_FFI_IO_NO_ENDIAN_SWAP && tensor->device.device_type == kDLCPU &&
      ffi::IsContiguous(*tensor) && tensor->byte_offset == 0) {
    strm->Write(tensor->data, data_size);
  } else {
    std::vector<uint8_t> bytes(data_size);
    CopyToBytes(source, bytes.data(), data_size);
    if (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
      SwapTensorBytes(bytes.data(), tensor->dtype, data_size);
    }
    strm->Write(bytes.data(), data_size);
  }
}

ffi::Tensor LoadTensor(support::Stream* strm) {
  uint64_t header, reserved;
  TVM_FFI_CHECK(strm->Read(&header), ValueError) << "Invalid DLTensor file format";
  TVM_FFI_CHECK(strm->Read(&reserved), ValueError) << "Invalid DLTensor file format";
  TVM_FFI_CHECK(header == kTVMTensorMagic, ValueError) << "Invalid DLTensor file format";
  Device dev;
  int ndim;
  DLDataType dtype;
  TVM_FFI_CHECK(strm->Read(&dev), ValueError) << "Invalid DLTensor file format";
  TVM_FFI_CHECK(strm->Read(&ndim), ValueError) << "Invalid DLTensor file format";
  TVM_FFI_CHECK(strm->Read(&dtype), ValueError) << "Invalid DLTensor file format";
  TVM_FFI_CHECK_EQ(dev.device_type, kDLCPU, ValueError)
      << "Invalid DLTensor device: can only save as CPU tensor";
  TVM_FFI_CHECK_GE(ndim, 0, ValueError) << "Invalid DLTensor file format";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    TVM_FFI_CHECK(strm->ReadArray(shape.data(), ndim), ValueError)
        << "Invalid DLTensor file format";
  }
  ffi::Tensor ret = Empty(ffi::Shape(shape), dtype, dev);
  size_t data_size = ffi::GetDataSize(*ret.GetDLTensorPtr());
  int64_t data_byte_size;
  TVM_FFI_CHECK(strm->Read(&data_byte_size), ValueError) << "Invalid DLTensor file format";
  TVM_FFI_CHECK_EQ(data_byte_size, static_cast<int64_t>(data_size), ValueError)
      << "Invalid DLTensor file format";
  TVM_FFI_CHECK_EQ(strm->Read(ret.GetDLTensorPtr()->data, data_size), data_size, ValueError)
      << "Invalid DLTensor file format: truncated data";
  if (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
    SwapTensorBytes(ret.GetDLTensorPtr()->data, dtype, data_size);
  }
  return ret;
}

}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::TypeAttrDef<tvm::ffi::TensorObj>()
      .def(tvm::ffi::type_attr::kDataToJson.name,
           [](const tvm::ffi::TensorObj* node) {
             std::string result;
             tvm::support::BytesOutStream mstrm(&result);
             tvm::runtime::SaveTensor(node, &mstrm);
             return tvm::ffi::Base64Encode(tvm::ffi::Bytes(result));
           })
      .def(tvm::ffi::type_attr::kDataFromJson.name, [](const std::string& blob) {
        tvm::ffi::Bytes bytes = tvm::ffi::Base64Decode(blob);
        tvm::support::BytesInStream mstrm(bytes.data(), bytes.size());
        return tvm::runtime::LoadTensor(&mstrm);
      });
  refl::GlobalDef()
      .def("runtime.TensorAllocWithScope", Empty)
      .def("runtime.TensorCreateView", CreateView)
      .def("runtime.TensorCopyFromBytes",
           [](DLTensor* arr, void* data, size_t nbytes) { CopyFromBytes(arr, data, nbytes); })
      .def("runtime.TensorCopyToBytes",
           [](DLTensor* arr, void* data, size_t nbytes) { CopyToBytes(arr, data, nbytes); })
      .def("runtime.TensorCopyFromTo", [](DLTensor* from, DLTensor* to) { CopyFromTo(from, to); });
}
