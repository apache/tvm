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
 * \file tvm/runtime/memory/memory_manager.cc
 * \brief Allocate and manage memory for the runtime.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <memory>
#include <utility>

#include "naive_allocator.h"
#include "pooled_allocator.h"

namespace tvm {
namespace runtime {
namespace memory {

Storage::Storage(Buffer buffer, Allocator* allocator) {
  auto n = ffi::make_object<StorageObj>();
  n->buffer = std::move(buffer);
  n->allocator = allocator;
  data_ = std::move(n);
}

Storage::Storage(ffi::Tensor backing) {
  TVM_FFI_CHECK(backing.defined(), ValueError) << "VM storage requires a defined backing tensor";
  TVM_FFI_CHECK(ffi::IsContiguous(*backing.GetDLTensorPtr()) && backing.byte_offset() == 0, ValueError)
      << "VM storage requires compact backing with zero byte offset";
  auto n = ffi::make_object<StorageObj>();
  n->buffer.data = backing.data_ptr();
  n->buffer.size = ffi::GetDataSize(backing);
  n->buffer.device = backing.device();
  n->buffer.alloc_type = kPooled;
  n->backing = std::move(backing);
  data_ = std::move(n);
}

inline void VerifyDataType(DLDataType dtype) {
  TVM_FFI_ICHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    TVM_FFI_ICHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    TVM_FFI_ICHECK_EQ(dtype.bits % 8, 0);
  }
  TVM_FFI_ICHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLDataType& dtype) {
  size_t align = dtype.lanes * dtype.bits / 8;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

ffi::Tensor StorageObj::AllocTensorScoped(int64_t offset, ffi::Shape shape, DLDataType dtype,
                                          ffi::String scope) {
  if (scope == "global" || scope.empty()) {
    return AllocTensor(offset, shape, dtype);
  }
  VerifyDataType(dtype);
  TVM_FFI_CHECK(allocator != nullptr, ValueError)
      << "This tensor-backed VM storage does not support physical scoped views";

  struct StorageScopedAlloc {
   public:
    explicit StorageScopedAlloc(Storage storage) : storage_(storage) {}

    void AllocData(DLTensor* tensor, const ffi::Shape& shape, const ffi::String& scope,
                   int64_t byte_offset) {
      tensor->data = storage_->allocator->CreateView(storage_->buffer, shape, tensor->dtype, scope);
      tensor->byte_offset = byte_offset;
    }
    void FreeData(DLTensor* tensor) { storage_->allocator->FreeView(tensor->device, tensor->data); }

   private:
    Storage storage_;
  };

  size_t needed_size = ffi::GetDataSize(shape.Product(), dtype);
  TVM_FFI_ICHECK(offset + needed_size <= this->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << this->buffer.size << "bytes";

  return ffi::Tensor::FromNDAlloc(StorageScopedAlloc(ffi::GetRef<Storage>(this)), shape, dtype,
                                  this->buffer.device, shape, scope, offset);
}

ffi::Tensor StorageObj::AllocTensor(int64_t offset, ffi::Shape shape, DLDataType dtype) {
  VerifyDataType(dtype);

  size_t needed_size = ffi::GetDataSize(shape.Product(), dtype);
  TVM_FFI_ICHECK(offset + needed_size <= this->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << this->buffer.size << "bytes";
  class StorageAlloc {
   public:
    explicit StorageAlloc(Storage storage) : storage_(storage) {}

    void AllocData(DLTensor* tensor, int64_t offset) {
      if (storage_->buffer.device.device_type == kDLHexagon) {
        // For Hexagon, non-zero offset support simply requires adjusting the
        // beginning of data pointer
        auto offset_ptr = reinterpret_cast<uint8_t*>(storage_->buffer.data) + offset;
        tensor->data = reinterpret_cast<void*>(offset_ptr);
        tensor->byte_offset = 0;
      } else {
        tensor->data = storage_->buffer.data;
        tensor->byte_offset = offset;
      }
    }
    void FreeData(DLTensor* tensor) {}

   private:
    Storage storage_;
  };

  return ffi::Tensor::FromNDAlloc(StorageAlloc(ffi::GetRef<Storage>(this)), shape, dtype,
                                  this->buffer.device, offset);
}

MemoryManager* MemoryManager::Global() {
  // NOTE: explicitly use new to avoid exit-time destruction of global state
  // Global state will be recycled by OS as the process exits.
  static auto* inst = new MemoryManager();
  return inst;
}

// The VM owns its legacy Buffer/Storage adaptation. Backends expose only owning
// tensors and physical views, so ordinary backend allocations never depend on it.
class OpenCLVMAllocator final : public Allocator {
 public:
  explicit OpenCLVMAllocator(AllocatorType type) : Allocator(type) {}

  Buffer Alloc(Device dev, size_t nbytes, size_t alignment, DLDataType dtype) final {
    alignment = std::max<size_t>(alignment, kAllocAlignment);
    TVM_FFI_CHECK_LE(alignment, std::numeric_limits<uint16_t>::max(), ValueError)
        << "OpenCL allocation alignment cannot be represented";
    TVM_FFI_CHECK_LE(nbytes, static_cast<size_t>(std::numeric_limits<int64_t>::max()), ValueError)
        << "OpenCL allocation size overflows";
    int64_t size = static_cast<int64_t>(nbytes / alignment + (nbytes % alignment != 0));
    return Keep(
        dev, nbytes,
        Allocate({size}, DLDataType{kDLUInt, 8, static_cast<uint16_t>(alignment)}, dev, "global"));
  }

  Buffer Alloc(Device dev, ffi::Shape shape, DLDataType dtype, const std::string& scope) final {
    auto owner = Allocate(shape, dtype, dev, scope);
    size_t nbytes = ffi::GetDataSize(owner);
    return Keep(dev, nbytes, std::move(owner));
  }

  void Free(const Buffer& buffer) final {
    ffi::Tensor owner;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = owners_.find(buffer.data);
      TVM_FFI_ICHECK(it != owners_.end());
      owner = std::move(it->second);
      owners_.erase(it);
      used_memory_ -= buffer.size;
    }
  }

  void* CreateView(const Buffer& buffer, ffi::Shape shape, DLDataType dtype,
                   const std::string& scope) final {
    ffi::Tensor backing;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = owners_.find(buffer.data);
      TVM_FFI_ICHECK(it != owners_.end());
      backing = it->second;
    }
    auto f = ffi::Function::GetGlobalRequired("runtime.opencl.create_view");
    auto view = f(backing, shape, dtype, ffi::String(scope)).cast<ffi::Tensor>();
    void* data = view.data_ptr();
    std::lock_guard<std::mutex> lock(mutex_);
    views_.emplace(data, std::move(view));
    return data;
  }

  void FreeView(Device, void* data) final {
    ffi::Tensor owner;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = views_.find(data);
      TVM_FFI_ICHECK(it != views_.end());
      owner = std::move(it->second);
      views_.erase(it);
    }
  }

  void Clear() final { ffi::Function::GetGlobalRequired("runtime.opencl.clear_pool")(); }
  size_t UsedMemory() const final { return used_memory_.load(std::memory_order_relaxed); }

 private:
  static ffi::Tensor Allocate(ffi::Shape shape, DLDataType dtype, Device dev,
                              const std::string& scope) {
    return ffi::Function::GetGlobalRequired("runtime.opencl.alloc_tensor")(shape, dtype, dev,
                                                                           ffi::String(scope))
        .cast<ffi::Tensor>();
  }

  Buffer Keep(Device dev, size_t nbytes, ffi::Tensor owner) {
    Buffer result{owner.data_ptr(), nbytes, dev, type()};
    std::lock_guard<std::mutex> lock(mutex_);
    owners_.emplace(result.data, std::move(owner));
    used_memory_ += nbytes;
    return result;
  }

  std::atomic<size_t> used_memory_{0};
  std::mutex mutex_;
  std::unordered_multimap<void*, ffi::Tensor> owners_;
  std::unordered_multimap<void*, ffi::Tensor> views_;
};

Allocator* GetDeviceSpecificAllocator(Device dev, AllocatorType type) {
  if (dev.device_type == kDLOpenCL) return new OpenCLVMAllocator(type);
  switch (type) {
    case kNaive:
      return new NaiveAllocator();
    case kPooled:
      return new PooledAllocator();
    default:
      TVM_FFI_THROW(InternalError) << "Unknown allocator type: " << type;
  }
}

Allocator* MemoryManager::GetOrCreateAllocator(Device dev, AllocatorType type) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  if (m->allocators_.find(dev) == m->allocators_.end()) {
    m->allocators_.emplace(dev, std::unordered_map<AllocatorType, std::unique_ptr<Allocator>>());
  }
  if (m->allocators_.at(dev).find(type) == m->allocators_.at(dev).end()) {
    std::unique_ptr<Allocator> alloc;
    alloc.reset(GetDeviceSpecificAllocator(dev, type));
    auto ret = alloc.get();
    m->allocators_.at(dev).emplace(type, std::move(alloc));
    return ret;
  }
  auto alloc = m->allocators_.at(dev).at(type).get();

  return alloc;
}

Allocator* MemoryManager::GetAllocator(Device dev, AllocatorType type) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  auto it = m->allocators_.find(dev);
  if (it == m->allocators_.end()) {
    TVM_FFI_THROW(InternalError) << "Allocator for " << dev << " has not been created yet.";
  }
  if (it->second.find(type) == it->second.end()) {
    TVM_FFI_THROW(InternalError) << "Allocator for " << dev << " of type " << type
                                 << " has not been created yet.";
  }
  return it->second.at(type).get();
}

void MemoryManager::Clear() {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  for (const auto& [device, allocators] : m->allocators_) {
    for (const auto& [allocator_type, allocator] : allocators) {
      allocator->Clear();
    }
  }
}

ffi::Tensor Allocator::Empty(ffi::Shape shape, DLDataType dtype, DLDevice dev,
                             ffi::Optional<ffi::String> mem_scope) {
  VerifyDataType(dtype);

  class BufferAlloc {
   public:
    explicit BufferAlloc(Buffer buffer) : buffer_(buffer) {}

    void AllocData(DLTensor* tensor) { tensor->data = buffer_.data; }
    void FreeData(DLTensor* tensor) {
      MemoryManager::GetAllocator(buffer_.device, buffer_.alloc_type)->Free(buffer_);
    }

   private:
    Buffer buffer_;
  };

  size_t alignment = GetDataAlignment(dtype);
  size_t size = ffi::GetDataSize(shape.Product(), dtype);

  Buffer buffer;
  if (!mem_scope.has_value() || (*mem_scope).empty() || (*mem_scope) == "global") {
    buffer = this->Alloc(dev, size, alignment, dtype);
  } else {
    buffer = this->Alloc(dev, shape, dtype, *mem_scope);
  }
  return ffi::Tensor::FromNDAlloc(BufferAlloc(buffer), shape, dtype, dev);
}

bool Allocator::AllowMemoryScope(const std::string& mem_scope) const {
  return mem_scope.empty() || mem_scope == "global";
}

Buffer Allocator::Alloc(Device dev, ffi::Shape shape, DLDataType type_hint,
                        const std::string& mem_scope) {
  if (AllowMemoryScope(mem_scope)) {
    // by default, we can always redirect to the flat memory allocations
    size_t alignment = GetDataAlignment(type_hint);
    size_t size = ffi::GetDataSize(shape.Product(), type_hint);
    return Alloc(dev, size, alignment, type_hint);
  }
  TVM_FFI_THROW(InternalError) << "Allocator cannot allocate data space with "
                               << "specified memory scope: " << mem_scope;
  return {};
}

void Allocator::Clear() {
  // This function by default does nothing.
  // For naive allocator, no explicit manual clear is needed.
  // Pooled allocator will override this method.
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("vm.builtin.memory_manager.clear", MemoryManager::Clear);
}

}  // namespace memory
}  // namespace runtime
}  // namespace tvm
