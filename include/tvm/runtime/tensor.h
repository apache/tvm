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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/device_api.h>
#include <tvm/support/io.h>
#include <tvm/support/serializer.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief A cache of owning device tensors, independent of VM storage policy.
 *
 * Construct pools with ffi::make_object. Live tensors retain the pool directly;
 * only unpooled backing tensors enter its cache. A backend supplies independent
 * backing owners, never a deleter that calls or retains this pool.
 *
 * This class protects host ownership, not device execution. An adapter must keep
 * separate pools for incompatible contexts/streams, or establish device ordering
 * before reuse. Callers must finish using tensors before releasing their owners.
 */
class TensorMemoryPool : public ffi::Object {
 public:
  virtual ~TensorMemoryPool() { Clear(); }

  /*! \brief Return an owning compact tensor in ordinary global memory. */
  virtual ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                            ffi::Optional<ffi::String> mem_scope = std::nullopt) {
    TVM_FFI_CHECK(!mem_scope.has_value() || (*mem_scope).empty() || *mem_scope == "global",
                  ValueError)
        << "TensorMemoryPool only supports global memory";
    TVM_FFI_CHECK_GE(device.device_id, 0, ValueError) << "Invalid device index";
    size_t nbytes = DataSize(shape, dtype);
    constexpr size_t kPageSize = 4096;
    const uint64_t max_capacity =
        std::min<uint64_t>(std::numeric_limits<size_t>::max(), std::numeric_limits<int64_t>::max());
    TVM_FFI_CHECK_LE(std::max<size_t>(nbytes, 1), max_capacity - (kPageSize - 1), ValueError)
        << "Rounded tensor allocation size overflows";
    size_t capacity = (std::max<size_t>(nbytes, 1) + kPageSize - 1) / kPageSize * kPageSize;
    // Backends may encode vector alignment through dtype. Round up instead of
    // passing a non-power-of-two alignment to a raw allocator.
    size_t alignment = kAllocAlignment;
    size_t element_bytes = (static_cast<size_t>(dtype.bits) * dtype.lanes + 7) / 8;
    while (alignment < element_bytes) alignment *= 2;

    Lease lease;
    lease.pool = ffi::GetObjectPtr<TensorMemoryPool>(this);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      lease.generation = generation_;
      lease.entry.device = device;
      lease.entry.alignment = alignment;
      lease.entry.capacity = capacity;
      auto best = cache_.end();
      for (auto it = cache_.begin(); it != cache_.end(); ++it) {
        const Entry& e = it->second;
        if (e.device.device_type == device.device_type && e.device.device_id == device.device_id &&
            e.alignment >= alignment && e.capacity >= capacity &&
            (best == cache_.end() || e.capacity < best->second.capacity)) {
          best = it;
        }
      }
      if (best != cache_.end()) {
        lease.id = best->first;
        lease.entry = std::move(best->second);
        cache_.erase(best);
      } else {
        lease.id = next_id_++;
      }
      ++live_allocations_;
    }
    try {
      if (!lease.entry.backing.defined()) {
        lease.entry.backing = AllocBacking(capacity, alignment, device);
        TVM_FFI_CHECK(lease.entry.backing.defined(), ValueError)
            << "AllocBacking returned an undefined tensor";
        try {
          const DLTensor& t = *lease.entry.backing.GetDLTensorPtr();
          TVM_FFI_CHECK(t.device.device_type == device.device_type &&
                            t.device.device_id == device.device_id && t.ndim == 1 &&
                            t.dtype.code == kDLUInt && t.dtype.bits == 8 && t.dtype.lanes == 1 &&
                            t.shape[0] == static_cast<int64_t>(capacity) && t.byte_offset == 0 &&
                            ffi::IsContiguous(t) && t.data != nullptr,
                        ValueError)
              << "AllocBacking must return an owning uint8[capacity] tensor";
        } catch (...) {
          lease.entry.backing = ffi::Tensor();
          throw;
        }
      }
      return ffi::Tensor::FromNDAlloc(lease, shape, dtype, device);
    } catch (...) {
      Recycle(lease.id, lease.generation, std::move(lease.entry));
      throw;
    }
  }

  /*!
   * \brief Release cached allocations. Live tensors remain valid.
   * \note Live allocations from before Clear are released, not cached, on return.
   */
  void Clear() {
    std::map<uint64_t, Entry> old;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++generation_;
      old.swap(cache_);
    }
    // Explicit allocation order also keeps collective backend frees deterministic.
    // Do not rely on an STL container's unspecified element destruction order.
    while (!old.empty()) old.erase(old.begin());
  }

  static constexpr bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("runtime.TensorMemoryPool", TensorMemoryPool, ffi::Object);

 protected:
  TensorMemoryPool() = default;

  /*!
   * \brief Allocate fresh owning uint8[nbytes] backing at the requested alignment.
   * \note Backing may contain an opaque device handle. Its deleter must own its
   * resources independently of the pool and may not route back through a pooled
   * DeviceAPI::Empty implementation.
   */
  virtual ffi::Tensor AllocBacking(size_t nbytes, size_t alignment, DLDevice device) = 0;

  /*! \brief Whether a backend must defer resource-session teardown. */
  bool HasLiveAllocations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_allocations_ != 0;
  }

 private:
  struct Entry {
    DLDevice device{};
    size_t alignment{0};
    size_t capacity{0};
    ffi::Tensor backing;
  };
  struct Lease {
    ffi::ObjectPtr<TensorMemoryPool> pool;
    uint64_t id{0};
    uint64_t generation{0};
    Entry entry;
    void AllocData(DLTensor* tensor) { tensor->data = entry.backing.data_ptr(); }
    void FreeData(DLTensor*) noexcept { pool->Recycle(id, generation, std::move(entry)); }
  };

  void Recycle(uint64_t id, uint64_t generation, Entry entry) noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (generation == generation_ && entry.backing.defined()) {
        cache_.emplace(id, std::move(entry));
        --live_allocations_;
        return;
      }
    } catch (...) {
      // A failed cache insertion must not escape tensor destruction.
    }
    // Backend finalizers use live ownership to exclude concurrent teardown. Keep
    // this lease live until uncached resources have finished releasing, outside
    // the cache mutex.
    entry.backing = ffi::Tensor();
    std::lock_guard<std::mutex> lock(mutex_);
    --live_allocations_;
  }

  static size_t DataSize(ffi::ShapeView shape, DLDataType dtype) {
    TVM_FFI_CHECK_LE(shape.size(), static_cast<size_t>(std::numeric_limits<int>::max()), ValueError)
        << "Tensor rank is too large";
    TVM_FFI_CHECK(dtype.bits != 0 && dtype.lanes != 0, ValueError)
        << "Tensor dtype must have a fixed nonzero element size";
    uint64_t numel = 1;
    // FFI materializes compact int64 strides, including suffixes of empty shapes.
    for (size_t i = shape.size(); i != 0; --i) {
      int64_t dim = shape[i - 1];
      TVM_FFI_CHECK_GE(dim, 0, ValueError) << "Tensor dimensions must be nonnegative";
      TVM_FFI_CHECK(dim == 0 || numel <= uint64_t(std::numeric_limits<int64_t>::max()) / dim,
                    ValueError)
          << "Tensor shape or compact strides overflow";
      numel *= static_cast<uint64_t>(dim);
    }
    uint64_t element_bits = uint64_t(dtype.bits) * dtype.lanes;
    TVM_FFI_CHECK_LE(numel, (std::numeric_limits<uint64_t>::max() - 7) / element_bits, ValueError)
        << "Tensor byte size overflows";
    uint64_t nbytes = (numel * element_bits + 7) / 8;
    TVM_FFI_CHECK_LE(nbytes, std::numeric_limits<size_t>::max(), ValueError)
        << "Tensor byte size exceeds addressable memory";
    return static_cast<size_t>(nbytes);
  }

  mutable std::mutex mutex_;
  std::map<uint64_t, Entry> cache_;
  uint64_t next_id_{0};
  uint64_t generation_{0};
  size_t live_allocations_{0};
};

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
