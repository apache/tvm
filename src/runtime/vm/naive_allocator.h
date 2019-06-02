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
 * \file src/runtime/naive_allocator.h
 */
#ifndef TVM_RUNTIME_VM_NAIVE_ALLOCATOR_H_
#define TVM_RUNTIME_VM_NAIVE_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <atomic>

#include "memory_manager.h"

namespace tvm {
namespace runtime {
namespace vm {

class NaiveAllocator final : public Allocator {
 public:
  explicit NaiveAllocator(TVMContext ctx) : Allocator(), used_memory_(0), ctx_(ctx) {}

  Buffer Alloc(size_t nbytes, size_t alignment, TVMType type_hint) override {
    Buffer buf;
    buf.ctx = ctx_;
    buf.size = nbytes;
    buf.data = DeviceAPI::Get(ctx_)->AllocDataSpace(ctx_, nbytes, alignment, type_hint);
    used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
    DLOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  void Free(const Buffer& buffer) override {
    DeviceAPI::Get(ctx_)->FreeDataSpace(buffer.ctx, buffer.data);
    used_memory_.fetch_sub(buffer.size, std::memory_order_relaxed);
    DLOG(INFO) << "free " << buffer.size << " B, used memory " << used_memory_ << " B";
  }

  size_t UsedMemory() const override {
    return used_memory_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<size_t> used_memory_;
  TVMContext ctx_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_NAIVE_ALLOCATOR_H_
