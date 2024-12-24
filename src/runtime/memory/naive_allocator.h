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
 * \file src/runtime/memory/naive_allocator.h
 */
#ifndef TVM_RUNTIME_MEMORY_NAIVE_ALLOCATOR_H_
#define TVM_RUNTIME_MEMORY_NAIVE_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/memory/memory_manager.h>

#include <atomic>
#include <string>

namespace tvm {
namespace runtime {
namespace memory {

class NaiveAllocator final : public Allocator {
 public:
  explicit NaiveAllocator() : Allocator(kNaive) { used_memory_ = 0; }

  Buffer Alloc(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    Buffer buf;
    buf.device = dev;
    buf.size = nbytes;
    buf.alloc_type = kNaive;
    buf.data = DeviceAPI::Get(dev)->AllocDataSpace(dev, nbytes, alignment, type_hint);
    used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
    DLOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  void Free(const Buffer& buffer) override {
    DeviceAPI::Get(buffer.device)->FreeDataSpace(buffer.device, buffer.data);
    used_memory_.fetch_sub(buffer.size, std::memory_order_relaxed);
    DLOG(INFO) << "free " << buffer.size << " B, used memory " << used_memory_ << " B";
  }

  size_t UsedMemory() const override { return used_memory_.load(std::memory_order_relaxed); }
};

}  // namespace memory
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_MEMORY_NAIVE_ALLOCATOR_H_
