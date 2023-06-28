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
 * \file tvm/runtime/relax_vm/naive_allocator.h
 */
#ifndef TVM_RUNTIME_RELAX_VM_NAIVE_ALLOCATOR_H_
#define TVM_RUNTIME_RELAX_VM_NAIVE_ALLOCATOR_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/relax_vm/memory_manager.h>

#include <atomic>

namespace tvm {
namespace runtime {
namespace relax_vm {

class NaiveAllocator final : public Allocator {
 public:
  explicit NaiveAllocator(Device dev) : Allocator(kNaive), used_memory_(0), device_(dev) {}

  Buffer Alloc(size_t nbytes, size_t alignment, DLDataType type_hint) override {
    Buffer buf;
    buf.device = device_;
    buf.size = nbytes;
    buf.data =
        runtime::DeviceAPI::Get(device_)->AllocDataSpace(device_, nbytes, alignment, type_hint);
    used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
    DLOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  Buffer Alloc(ShapeTuple shape, DLDataType dtype, String mem_scope) override {
    DLTensor temp;
    temp.data = nullptr;
    temp.device = device_;
    temp.ndim = shape.size();
    temp.dtype = dtype;
    temp.shape = const_cast<int64_t*>(shape.data());
    temp.strides = nullptr;
    temp.byte_offset = 0;
    size_t nbytes = GetDataSize(temp);

    Buffer buf;
    buf.device = device_;
    buf.size = nbytes;
    buf.data = runtime::DeviceAPI::Get(device_)->AllocDataSpace(device_, shape.size(), shape.data(),
                                                                dtype, mem_scope);
    used_memory_.fetch_add(nbytes, std::memory_order_relaxed);
    DLOG(INFO) << "allocate " << nbytes << " B, used memory " << used_memory_ << " B";
    return buf;
  }

  void Free(const Buffer& buffer) override {
    runtime::DeviceAPI::Get(device_)->FreeDataSpace(buffer.device, buffer.data);
    used_memory_.fetch_sub(buffer.size, std::memory_order_relaxed);
    DLOG(INFO) << "free " << buffer.size << " B, used memory " << used_memory_ << " B";
  }

 private:
  std::atomic<size_t> used_memory_;
  Device device_;
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_NAIVE_ALLOCATOR_H_
