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
 * \file tvm/runtime/tensor.h
 * \brief Device-independent tensor allocation, copies, views and serialization.
 */
#ifndef TVM_RUNTIME_TENSOR_H_
#define TVM_RUNTIME_TENSOR_H_

#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <atomic>
#include <functional>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Allocate an uninitialized tensor through the runtime device allocator.
 * \param shape The tensor shape.
 * \param dtype The tensor data type.
 * \param dev The allocation device.
 * \param mem_scope The optional device memory scope.
 */
TVM_RUNTIME_DLL ffi::Tensor Empty(ffi::Shape shape, DLDataType dtype, Device dev,
                                  ffi::Optional<ffi::String> mem_scope = std::nullopt);

/*!
 * \brief Copy between existing tensors with exactly matching byte sizes.
 * \note The copy may be asynchronous. The caller owns stream synchronization.
 */
TVM_RUNTIME_DLL void CopyFromTo(ffi::TensorView source, ffi::TensorView destination,
                                TVMStreamHandle stream = nullptr);

/*!
 * \brief Allocate a fresh tensor on dev, copy source, and synchronize the copy.
 * \note A fresh allocation is made even when source already resides on dev.
 */
TVM_RUNTIME_DLL ffi::Tensor CopyTo(ffi::TensorView source, Device dev,
                                   ffi::Optional<ffi::String> mem_scope = std::nullopt);

/*!
 * \brief Create a compact view retaining ownership of source.
 * \param source The owning source tensor, which must be compact.
 * \param shape The view shape.
 * \param dtype The view data type.
 * \param relative_byte_offset Byte offset relative to the source's byte offset.
 * \note The view must fit within the source's storage extent.
 */
TVM_RUNTIME_DLL ffi::Tensor CreateView(ffi::Tensor source, ffi::Shape shape, DLDataType dtype,
                                       uint64_t relative_byte_offset = 0);

/*!
 * \brief Copy exactly nbytes from a byte buffer into a contiguous tensor.
 * \note Synchronizes stream before returning.
 */
TVM_RUNTIME_DLL void CopyFromBytes(ffi::TensorView destination, const void* source_bytes,
                                   size_t nbytes, TVMStreamHandle stream = nullptr);

/*!
 * \brief Copy a contiguous tensor into a byte buffer of exactly nbytes.
 * \note Synchronizes stream before returning.
 */
TVM_RUNTIME_DLL void CopyToBytes(ffi::TensorView source, void* destination_bytes, size_t nbytes,
                                 TVMStreamHandle stream = nullptr);

/*! \brief Magic number for Tensor file */
constexpr uint64_t kTVMTensorMagic = 0xDD5E40F096B4A13F;

/*! \brief Save a tensor in the portable CPU tensor format. */
inline void SaveTensor(ffi::TensorView source, support::Stream* strm) {
  const DLTensor* tensor = ffi::AnyView(source).cast<DLTensor*>();
  uint64_t header = kTVMTensorMagic, reserved = 0;
  strm->Write(header);
  strm->Write(reserved);
  // Always save data as CPU context
  //
  // Parameters that get serialized should be in CPU by default.
  // So even the array's context is GPU, it will be stored as CPU array.
  // This is used to prevent case when another user loads the parameters
  // back on machine that do not have GPU or related context.
  //
  // We can always do array.CopyTo(target_dev) to get a corresponding
  // array in the target context.
  Device cpu_dev;
  cpu_dev.device_type = kDLCPU;
  cpu_dev.device_id = 0;
  strm->Write(cpu_dev);
  strm->Write(tensor->ndim);
  strm->Write(tensor->dtype);
  int ndim = tensor->ndim;
  strm->WriteArray(tensor->shape, ndim);
  int type_bytes = (tensor->dtype.bits + 7) / 8;
  int64_t num_elems = 1;
  for (int i = 0; i < ndim; ++i) {
    num_elems *= tensor->shape[i];
  }
  int64_t data_byte_size = type_bytes * num_elems;
  strm->Write(data_byte_size);

  if (TVM_FFI_IO_NO_ENDIAN_SWAP && tensor->device.device_type == kDLCPU &&
      ffi::IsContiguous(*tensor) && tensor->byte_offset == 0) {
    // quick path
    strm->Write(tensor->data, data_byte_size);
  } else {
    std::vector<uint8_t> bytes(data_byte_size);
    CopyToBytes(source, bytes.data(), data_byte_size);
    if (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
      ffi::ByteSwap(bytes.data(), type_bytes, num_elems);
    }
    strm->Write(bytes.data(), data_byte_size);
  }
}

/*! \brief Load a tensor from the portable format into CPU storage. */
inline ffi::Tensor LoadTensor(support::Stream* strm) {
  uint64_t header, reserved;
  TVM_FFI_ICHECK(strm->Read(&header)) << "Invalid DLTensor file format";
  TVM_FFI_ICHECK(strm->Read(&reserved)) << "Invalid DLTensor file format";
  TVM_FFI_ICHECK(header == kTVMTensorMagic) << "Invalid DLTensor file format";
  Device dev;
  int ndim;
  DLDataType dtype;
  TVM_FFI_ICHECK(strm->Read(&dev)) << "Invalid DLTensor file format";
  TVM_FFI_ICHECK(strm->Read(&ndim)) << "Invalid DLTensor file format";
  TVM_FFI_ICHECK(strm->Read(&dtype)) << "Invalid DLTensor file format";
  TVM_FFI_ICHECK_EQ(dev.device_type, kDLCPU)
      << "Invalid DLTensor device: can only save as CPU tensor";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    TVM_FFI_ICHECK(strm->ReadArray(&shape[0], ndim)) << "Invalid DLTensor file format";
  }
  ffi::Tensor ret = Empty(ffi::Shape(shape), dtype, dev);
  int64_t num_elems = 1;
  int elem_bytes = (ret.GetDLTensorPtr()->dtype.bits + 7) / 8;
  for (int i = 0; i < ret.GetDLTensorPtr()->ndim; ++i) {
    num_elems *= ret.shape()[i];
  }
  int64_t data_byte_size;
  TVM_FFI_ICHECK(strm->Read(&data_byte_size)) << "Invalid DLTensor file format";
  TVM_FFI_ICHECK(data_byte_size == num_elems * elem_bytes) << "Invalid DLTensor file format";
  auto read_ret = strm->Read(ret.GetDLTensorPtr()->data, data_byte_size);
  // Only check non-empty data
  if (ndim > 0 && shape[0] != 0) {
    TVM_FFI_ICHECK(read_ret) << "Invalid DLTensor file format";
  }
  if (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
    ffi::ByteSwap(ret.GetDLTensorPtr()->data, elem_bytes, num_elems);
  }
  return ret;
}

/*!
 * \brief Get the preferred host device from the input device.
 * - For CUDA and ROCm, CUDAHost and ROCMHost will be returned for pinned memory,
 * since pinned memory reduces copy overhead.
 * - For other devices, CPU is returned as a fallback.
 */
inline Device GetPreferredHostDevice(Device device) {
  if (device.device_type == DLDeviceType::kDLCUDA) {
    return Device{DLDeviceType::kDLCUDAHost, 0};
  } else if (device.device_type == DLDeviceType::kDLROCM) {
    return Device{DLDeviceType::kDLROCMHost, 0};
  } else {
    // Fallback to CPU.
    return Device{DLDeviceType::kDLCPU, 0};
  }
}

}  // namespace runtime
}  // namespace tvm

namespace std {
template <>
struct hash<tvm::Device> {
  std::size_t operator()(const tvm::Device& dev) const {
    return ((dev.device_id << 8) | dev.device_type);
  }
};

template <>
struct equal_to<tvm::Device> {
  bool operator()(const tvm::Device& lhs, const tvm::Device& rhs) const {
    return (lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id);
  }
};
}  // namespace std

#endif  // TVM_RUNTIME_TENSOR_H_
