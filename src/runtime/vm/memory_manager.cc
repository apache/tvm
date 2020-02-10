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
#include <utility>
#include <memory>
#include "memory_manager.h"
#include "naive_allocator.h"
#include "pooled_allocator.h"

namespace tvm {
namespace runtime {
namespace vm {

static void BufferDeleter(Object* obj) {
  auto* ptr = static_cast<NDArray::Container*>(obj);
  CHECK(ptr->manager_ctx != nullptr);
  Buffer* buffer = reinterpret_cast<Buffer*>(ptr->manager_ctx);
  MemoryManager::Global()->GetAllocator(buffer->ctx)->
      Free(*(buffer));
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
  // TODO(@jroesch): generalize later to non-overlapping allocations.
  CHECK_EQ(offset, 0u);
  VerifyDataType(dtype);

  // crtical zone: allocate header, cannot throw
  NDArray::Container* container = new NDArray::Container(nullptr, shape, dtype, this->buffer.ctx);

  container->SetDeleter(StorageObj::Deleter);
  size_t needed_size = GetDataSize(container->dl_tensor);
  this->IncRef();
  container->manager_ctx = reinterpret_cast<void*>(this);
  container->dl_tensor.data = this->buffer.data;
  NDArray ret(GetObjectPtr<Object>(container));

  // RAII in effect, now run the check.
  // TODO(@jroesch): generalize later to non-overlapping allocations.
  CHECK(needed_size == this->buffer.size)
    << "size mistmatch required " << needed_size << " found " << this->buffer.size;

  return ret;
}

MemoryManager* MemoryManager::Global() {
  static MemoryManager memory_manager;
  return &memory_manager;
}

Allocator* MemoryManager::GetAllocator(TVMContext ctx) {
  std::lock_guard<std::mutex> lock(mu_);
  if (allocators_.find(ctx) == allocators_.end()) {
    DLOG(INFO) << "New allocator for " << DeviceName(ctx.device_type) << "("
               << ctx.device_id << ")";
    std::unique_ptr<Allocator> alloc(new NaiveAllocator(ctx));
    allocators_.emplace(ctx, std::move(alloc));
  }
  return allocators_.at(ctx).get();
}

NDArray Allocator::Empty(std::vector<int64_t> shape, DLDataType dtype, DLContext ctx) {
  VerifyDataType(dtype);
  NDArray::Container* container = new NDArray::Container(nullptr, shape, dtype, ctx);
  container->SetDeleter(BufferDeleter);
  size_t size = GetDataSize(container->dl_tensor);
  size_t alignment = GetDataAlignment(container->dl_tensor);
  Buffer *buffer = new Buffer;
  *buffer = this->Alloc(size, alignment, dtype);
  container->manager_ctx = reinterpret_cast<void*>(buffer);
  container->dl_tensor.data = buffer->data;
  return NDArray(GetObjectPtr<Object>(container));
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
