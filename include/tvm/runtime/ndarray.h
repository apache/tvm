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
 * \file tvm/runtime/ndarray.h
 * \brief A device-independent managed NDArray abstraction.
 */
#ifndef TVM_RUNTIME_NDARRAY_H_
#define TVM_RUNTIME_NDARRAY_H_

#include <tvm/ffi/container/ndarray.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/serializer.h>

#include <atomic>
#include <functional>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

using ffi::GetDataSize;
using ffi::IsAligned;
using ffi::IsContiguous;

/*!
 * \brief Managed NDArray.
 *  The array is backed by reference counted blocks.
 */
class NDArray : public tvm::ffi::NDArray {
 public:
  using Container = ffi::NDArrayObj;
  NDArray() = default;
  /*!
   * \brief constructor.
   * \param data ObjectPtr to the data container.
   */
  explicit NDArray(ObjectPtr<Object> data) : tvm::ffi::NDArray(data) {}
  NDArray(ffi::NDArray&& other) : tvm::ffi::NDArray(std::move(other)) {}  // NOLINT(*)
  NDArray(const ffi::NDArray& other) : tvm::ffi::NDArray(other) {}        // NOLINT(*)

  ffi::Shape Shape() const { return this->shape(); }
  runtime::DataType DataType() const { return runtime::DataType(this->dtype()); }

  // DLPack handling
  static NDArray FromDLPack(DLManagedTensor* tensor) {
    return tvm::ffi::NDArray::FromDLPack(tensor, kAllocAlignment, true);
  }

  static NDArray FromDLPackVersioned(DLManagedTensorVersioned* tensor) {
    return tvm::ffi::NDArray::FromDLPackVersioned(tensor, kAllocAlignment, true);
  }
  /*!
   * \brief Copy data content from another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU context.
   *       TVMSynchronize is necessary.
   */
  inline void CopyFrom(const DLTensor* other);
  inline void CopyFrom(const NDArray& other);
  /*!
   * \brief Copy data content from a byte buffer.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the buffer in bytes
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a TVMSynchronize.
   */
  TVM_DLL void CopyFromBytes(const void* data, size_t nbytes);
  /*!
   * \brief Copy data content into another array.
   * \param other The source array to be copied from.
   * \note The copy may happen asynchronously if it involves a GPU context.
   *       TVMSynchronize is necessary.
   */
  inline void CopyTo(DLTensor* other) const;
  inline void CopyTo(const NDArray& other) const;
  /*!
   * \brief Copy data content into another array.
   * \param data The source bytes to be copied from.
   * \param nbytes The size of the data buffer.
   *        Must be equal to the size of the NDArray.
   * \note The copy always triggers a TVMSynchronize.
   */
  TVM_DLL void CopyToBytes(void* data, size_t nbytes) const;
  /*!
   * \brief Copy the data to another device.
   * \param dev The target device.
   * \param mem_scope The memory scope of the target array.
   * \return The array under another device.
   * \note The copy always triggers a TVMSynchronize.
   */
  TVM_DLL NDArray CopyTo(const Device& dev, Optional<String> mem_scope = std::nullopt) const;
  /*!
   * \brief Load NDArray from stream
   * \param stream The input data stream
   * \return Whether load is successful
   */
  inline bool Load(dmlc::Stream* stream);
  /*!
   * \brief Save NDArray to stream
   * \param stream The output data stream
   */
  inline void Save(dmlc::Stream* stream) const;

  /*!
   * \brief Create a NDArray that shares the data memory with the current one.
   *
   * \param shape The shape of the new array.
   *
   * \param dtype The data type of the new array.
   *
   * \param relative_byte_offset The offset of the output NDArray,
   *     relative to the current byte offset.
   *
   *     By default, the offset of the view is the same as the offset
   *     of the current array.
   *
   * \note The new array must not allow access of addresses which
   *       would be out of bounds in the current array.  If the new
   *       array is larger than the current array, or if the
   *       `relative_byte_offset` would place the end of the new array
   *       outside the bounds of the current array, this function will
   *       raise an exception.
   */
  TVM_DLL NDArray CreateView(ffi::Shape shape, DLDataType dtype,
                             uint64_t relative_byte_offset = 0) const;
  /*!
   * \brief Create an empty NDArray.
   * \param shape The shape of the new array.
   * \param dtype The data type of the new array.
   * \param dev The device of the array.
   * \param mem_scope The memory scope of the array.
   * \return The created Array
   */
  TVM_DLL static NDArray Empty(ffi::Shape shape, DLDataType dtype, Device dev,
                               Optional<String> mem_scope = std::nullopt);
  /*!
   * \brief Function to copy data from one array to another.
   * \param from The source array.
   * \param to The target array.
   * \param stream The stream used in copy.
   */
  TVM_DLL static void CopyFromTo(const DLTensor* from, DLTensor* to,
                                 TVMStreamHandle stream = nullptr);

  /*!
   * \brief Function to copy data from one array to a byte buffer.
   * \param from The source array.
   * \param to The target byte buffer.
   * \param nbytes The size of the data buffer.
   * \param stream The stream used in copy.
   */
  TVM_DLL static void CopyToBytes(const DLTensor* from, void* to, size_t nbytes,
                                  TVMStreamHandle stream = nullptr);
};

/*!
 * \brief Save a DLTensor to stream
 * \param strm The output stream
 * \param tensor The tensor to be saved.
 */
inline bool SaveDLTensor(dmlc::Stream* strm, const DLTensor* tensor);

inline void NDArray::CopyFrom(const DLTensor* other) {
  ICHECK(data_ != nullptr);
  CopyFromTo(other, get_mutable());
}

inline void NDArray::CopyFrom(const NDArray& other) {
  ICHECK(data_ != nullptr);
  ICHECK(other.data_ != nullptr);
  CopyFromTo(other.get_mutable(), get_mutable());
}

inline void NDArray::CopyTo(DLTensor* other) const {
  ICHECK(data_ != nullptr);
  CopyFromTo(get_mutable(), other);
}

inline void NDArray::CopyTo(const NDArray& other) const {
  ICHECK(data_ != nullptr);
  ICHECK(other.data_ != nullptr);
  CopyFromTo(get_mutable(), other.get_mutable());
}

/*! \brief Magic number for NDArray file */
constexpr uint64_t kTVMNDArrayMagic = 0xDD5E40F096B4A13F;

inline bool SaveDLTensor(dmlc::Stream* strm, const DLTensor* tensor) {
  uint64_t header = kTVMNDArrayMagic, reserved = 0;
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

  if (DMLC_IO_NO_ENDIAN_SWAP && tensor->device.device_type == kDLCPU &&
      tensor->strides == nullptr && tensor->byte_offset == 0) {
    // quick path
    strm->Write(tensor->data, data_byte_size);
  } else {
    std::vector<uint8_t> bytes(data_byte_size);
    NDArray::CopyToBytes(const_cast<DLTensor*>(tensor), dmlc::BeginPtr(bytes), data_byte_size);
    if (!DMLC_IO_NO_ENDIAN_SWAP) {
      dmlc::ByteSwap(dmlc::BeginPtr(bytes), type_bytes, num_elems);
    }
    strm->Write(dmlc::BeginPtr(bytes), data_byte_size);
  }
  return true;
}

inline void NDArray::Save(dmlc::Stream* strm) const { SaveDLTensor(strm, operator->()); }

inline bool NDArray::Load(dmlc::Stream* strm) {
  uint64_t header, reserved;
  ICHECK(strm->Read(&header)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&reserved)) << "Invalid DLTensor file format";
  ICHECK(header == kTVMNDArrayMagic) << "Invalid DLTensor file format";
  Device dev;
  int ndim;
  DLDataType dtype;
  ICHECK(strm->Read(&dev)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&ndim)) << "Invalid DLTensor file format";
  ICHECK(strm->Read(&dtype)) << "Invalid DLTensor file format";
  ICHECK_EQ(dev.device_type, kDLCPU) << "Invalid DLTensor device: can only save as CPU tensor";
  std::vector<int64_t> shape(ndim);
  if (ndim != 0) {
    ICHECK(strm->ReadArray(&shape[0], ndim)) << "Invalid DLTensor file format";
  }
  NDArray ret = NDArray::Empty(ffi::Shape(shape), dtype, dev);
  int64_t num_elems = 1;
  int elem_bytes = (ret->dtype.bits + 7) / 8;
  for (int i = 0; i < ret->ndim; ++i) {
    num_elems *= ret->shape[i];
  }
  int64_t data_byte_size;
  ICHECK(strm->Read(&data_byte_size)) << "Invalid DLTensor file format";
  ICHECK(data_byte_size == num_elems * elem_bytes) << "Invalid DLTensor file format";
  auto read_ret = strm->Read(ret->data, data_byte_size);
  // Only check non-empty data
  if (ndim > 0 && shape[0] != 0) {
    ICHECK(read_ret) << "Invalid DLTensor file format";
  }
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(ret->data, elem_bytes, num_elems);
  }
  *this = ret;
  return true;
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

#endif  // TVM_RUNTIME_NDARRAY_H_
