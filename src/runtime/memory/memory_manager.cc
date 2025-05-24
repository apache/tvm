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
#include <tvm/ffi/function.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <memory>
#include <utility>

#include "naive_allocator.h"
#include "pooled_allocator.h"

namespace tvm {
namespace runtime {
namespace memory {

Storage::Storage(Buffer buffer, Allocator* allocator) {
  auto n = make_object<StorageObj>();
  n->buffer = std::move(buffer);
  n->allocator = allocator;
  data_ = std::move(n);
}

inline void VerifyDataType(DLDataType dtype) {
  ICHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    ICHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    ICHECK_EQ(dtype.bits % 8, 0);
  }
  ICHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLDataType& dtype) {
  size_t align = dtype.lanes * dtype.bits / 8;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

NDArray StorageObj::AllocNDArrayScoped(int64_t offset, ffi::Shape shape, DLDataType dtype,
                                       String scope) {
  if (scope == "global" || scope.empty()) {
    return AllocNDArray(offset, shape, dtype);
  }
  VerifyDataType(dtype);

  struct StorageScopedAlloc {
   public:
    explicit StorageScopedAlloc(Storage storage) : storage_(storage) {}

    void AllocData(DLTensor* tensor, const ffi::Shape& shape, const String& scope,
                   int64_t byte_offset) {
      tensor->data = storage_->allocator->CreateView(storage_->buffer, shape, tensor->dtype, scope);
      tensor->byte_offset = byte_offset;
    }
    void FreeData(DLTensor* tensor) { storage_->allocator->FreeView(tensor->device, tensor->data); }

   private:
    Storage storage_;
  };

  size_t needed_size = ffi::GetDataSize(shape.Product(), dtype);
  ICHECK(offset + needed_size <= this->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << this->buffer.size << "bytes";

  return NDArray::FromNDAlloc(StorageScopedAlloc(GetRef<Storage>(this)), shape, dtype,
                              this->buffer.device, shape, scope, offset);
}

NDArray StorageObj::AllocNDArray(int64_t offset, ffi::Shape shape, DLDataType dtype) {
  VerifyDataType(dtype);

  size_t needed_size = ffi::GetDataSize(shape.Product(), dtype);
  ICHECK(offset + needed_size <= this->buffer.size)
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

  return NDArray::FromNDAlloc(StorageAlloc(GetRef<Storage>(this)), shape, dtype,
                              this->buffer.device, offset);
}

MemoryManager* MemoryManager::Global() {
  // NOTE: explicitly use new to avoid exit-time destruction of global state
  // Global state will be recycled by OS as the process exits.
  static auto* inst = new MemoryManager();
  return inst;
}

std::string DeviceTypeStr(DLDeviceType type) {
  switch (type) {
    case kDLOpenCL:
      return "opencl";
      break;
    case kDLVulkan:
      return "vulkan";
      break;
    default:
      return "";
  }
}

Allocator* GetDeviceSpecificAllocator(Device dev, AllocatorType type) {
  std::string dev_str = DeviceTypeStr(dev.device_type);
  auto device_alloc_helper = tvm::ffi::Function::GetGlobal("DeviceAllocator." + dev_str);
  void* valloc;
  Allocator* allocator = nullptr;
  if (device_alloc_helper) {
    valloc = (*device_alloc_helper)(dev, static_cast<int>(type)).cast<void*>();
    allocator = static_cast<Allocator*>(valloc);
  }
  if (nullptr == allocator) {
    switch (type) {
      case kNaive: {
        VLOG(1) << "New naive allocator for " << dev;
        allocator = new NaiveAllocator();
        break;
      }
      case kPooled: {
        VLOG(1) << "New pooled allocator for " << dev;
        allocator = new PooledAllocator();
        break;
      }
      default:
        LOG(FATAL) << "Unknown allocator type: " << type;
    }
  }
  return allocator;
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
    LOG(FATAL) << "Allocator for " << dev << " has not been created yet.";
  }
  if (it->second.find(type) == it->second.end()) {
    LOG(FATAL) << "Allocator for " << dev << " of type " << type << " has not been created yet.";
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

NDArray Allocator::Empty(ffi::Shape shape, DLDataType dtype, DLDevice dev,
                         Optional<String> mem_scope) {
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
  if (!mem_scope.defined() || mem_scope.value().empty() || mem_scope.value() == "global") {
    buffer = this->Alloc(dev, size, alignment, dtype);
  } else {
    buffer = this->Alloc(dev, shape, dtype, mem_scope.value());
  }
  return NDArray::FromNDAlloc(BufferAlloc(buffer), shape, dtype, dev);
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
  LOG(FATAL) << "Allocator cannot allocate data space with "
             << "specified memory scope: " << mem_scope;
  return {};
}

void Allocator::Clear() {
  // This function by default does nothing.
  // For naive allocator, no explicit manual clear is needed.
  // Pooled allocator will override this method.
}

TVM_FFI_REGISTER_GLOBAL("vm.builtin.memory_manager.clear").set_body_typed(MemoryManager::Clear);

}  // namespace memory
}  // namespace runtime
}  // namespace tvm
