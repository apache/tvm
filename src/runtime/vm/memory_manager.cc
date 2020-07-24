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
 * \file tvm/runtime/vm/memory_manager.cc
 * \brief Allocate and manage memory for the runtime.
 */
#include <tvm/runtime/vm/memory_manager.h>

#include <memory>
#include <utility>

#include "naive_allocator.h"
#include "pooled_allocator.h"

namespace tvm {
namespace runtime {
namespace vm {

static void BufferDeleter(Object* obj) {
  auto* ptr = static_cast<NDArray::Container*>(obj);
  CHECK(ptr->manager_ctx != nullptr);
  Buffer* buffer = reinterpret_cast<Buffer*>(ptr->manager_ctx);
  MemoryManager::GetAllocator(buffer->ctx)->Free(*(buffer));
  delete buffer;
  delete ptr;
}

void StorageObj::Deleter(Object* obj) {
  auto* ptr = static_cast<NDArray::Container*>(obj);
  // When invoking AllocNDArray we don't own the underlying allocation
  // and should not delete the buffer, but instead let it be reclaimed
  // by the storage object's destructor.
  //
  // We did bump the reference count by 1 to keep alive the StorageObj
  // allocation in case this NDArray is the sole owner.
  //
  // We decrement the object allowing for the buffer to release our
  // reference count from allocation.
  StorageObj* storage = reinterpret_cast<StorageObj*>(ptr->manager_ctx);
  storage->DecRef();
  delete ptr;
}

inline void VerifyDataType(DLDataType dtype) {
  CHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    CHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    CHECK_EQ(dtype.bits % 8, 0);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

NDArray StorageObj::AllocNDArray(size_t offset, std::vector<int64_t> shape, DLDataType dtype) {
  VerifyDataType(dtype);

  // crtical zone: allocate header, cannot throw
  NDArray::Container* container = new NDArray::Container(nullptr, shape, dtype, this->buffer.ctx);

  container->SetDeleter(StorageObj::Deleter);
  size_t needed_size = GetDataSize(container->dl_tensor);
  this->IncRef();
  // The manager context pointer must continue to point to the storage object
  // which owns the backing memory, and keeps track of the reference count.
  //
  // When we free a container we extract the storage object, decrement its
  // reference count, then destroy the container, but leave the underlying
  // buffer intact.
  container->manager_ctx = reinterpret_cast<void*>(this);

  // is this UB?
  // The only change we make w.r.t offset is modifying the data pointer
  // of the backing tensor to point into the buffer instead of its start.
  auto offset_ptr = reinterpret_cast<uint8_t*>(this->buffer.data) + offset;
  container->dl_tensor.data = reinterpret_cast<void*>(offset_ptr);

  NDArray ret(GetObjectPtr<Object>(container));
  // RAII in effect, now run the check.

  CHECK(offset + needed_size <= this->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << this->buffer.size << "bytes";

  return ret;
}

MemoryManager* MemoryManager::Global() {
  static MemoryManager memory_manager;
  return &memory_manager;
}

Allocator* MemoryManager::GetOrCreateAllocator(TVMContext ctx, AllocatorType type) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  if (m->allocators_.find(ctx) == m->allocators_.end()) {
    std::unique_ptr<Allocator> alloc;
    switch (type) {
      case kNaive: {
        DLOG(INFO) << "New naive allocator for " << DeviceName(ctx.device_type) << "("
                   << ctx.device_id << ")";
        alloc.reset(new NaiveAllocator(ctx));
        break;
      }
      case kPooled: {
        DLOG(INFO) << "New pooled allocator for " << DeviceName(ctx.device_type) << "("
                   << ctx.device_id << ")";
        alloc.reset(new PooledAllocator(ctx));
        break;
      }
      default:
        LOG(FATAL) << "Unknown allocator type: " << type;
    }
    auto ret = alloc.get();
    m->allocators_.emplace(ctx, std::move(alloc));
    return ret;
  }
  auto alloc = m->allocators_.at(ctx).get();
  if (alloc->type() != type) {
    LOG(WARNING) << "The type of existing allocator for " << DeviceName(ctx.device_type) << "("
                 << ctx.device_id << ") is different from the request type (" << alloc->type()
                 << " vs " << type << ")";
  }
  return alloc;
}

Allocator* MemoryManager::GetAllocator(TVMContext ctx) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mu_);
  auto it = m->allocators_.find(ctx);
  if (it == m->allocators_.end()) {
    LOG(FATAL) << "Allocator for " << DeviceName(ctx.device_type) << "(" << ctx.device_id
               << ") has not been created yet.";
  }
  return it->second.get();
}

NDArray Allocator::Empty(std::vector<int64_t> shape, DLDataType dtype, DLContext ctx) {
  VerifyDataType(dtype);
  NDArray::Container* container = new NDArray::Container(nullptr, shape, dtype, ctx);
  container->SetDeleter(BufferDeleter);
  size_t size = GetDataSize(container->dl_tensor);
  size_t alignment = GetDataAlignment(container->dl_tensor);
  Buffer* buffer = new Buffer;
  *buffer = this->Alloc(size, alignment, dtype);
  container->manager_ctx = reinterpret_cast<void*>(buffer);
  container->dl_tensor.data = buffer->data;
  return NDArray(GetObjectPtr<Object>(container));
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
