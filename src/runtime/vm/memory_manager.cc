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
 *  Copyright (c) 2019 by Contributors
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

static void BufferDeleter(NDArray::Container* ptr) {
  CHECK(ptr->manager_ctx != nullptr);
  Buffer* buffer = reinterpret_cast<Buffer*>(ptr->manager_ctx);
  MemoryManager::Global()->GetAllocator(buffer->ctx)->
      Free(*(buffer));
  delete buffer;
  delete ptr;
}

NDArray Allocator::Empty(std::vector<int64_t> shape, DLDataType dtype, DLContext ctx) {
  VerifyDataType(dtype);
  NDArray::Container* container = new NDArray::Container(nullptr, shape, dtype, ctx);
  container->deleter = BufferDeleter;
  size_t size = GetDataSize(container->dl_tensor);
  size_t alignment = GetDataAlignment(container->dl_tensor);
  Buffer *buffer = new Buffer;
  *buffer = this->Alloc(size, alignment, dtype);
  container->manager_ctx = reinterpret_cast<void*>(buffer);
  container->dl_tensor.data = buffer->data;
  return NDArray(container);
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
