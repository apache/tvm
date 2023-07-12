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
 * \file tvm/runtime/relax_vm/memory_manager.cc
 * \brief Allocate and manage memory for the Relay VM.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/memory_manager.h>

#include <memory>
#include <utility>

#include "naive_allocator.h"
#include "pooled_allocator.h"
#include "tvm/runtime/memory.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

static void BufferDeleter(Object* obj) {
  auto* ptr = static_cast<runtime::NDArray::Container*>(obj);
  ICHECK(ptr->manager_ctx != nullptr);
  Buffer* buffer = reinterpret_cast<Buffer*>(ptr->manager_ctx);
  MemoryManager::GetAllocator(buffer->device)->Free(*(buffer));
  delete buffer;
  delete ptr;
}

void StorageObj::Deleter(Object* obj) {
  auto* ptr = static_cast<runtime::NDArray::Container*>(obj);
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

Storage::Storage(Buffer buffer) {
  auto n = make_object<StorageObj>();
  n->buffer = std::move(buffer);
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

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < runtime::kAllocAlignment) return runtime::kAllocAlignment;
  return align;
}

runtime::NDArray StorageObj::AllocNDArray(uint64_t offset, ShapeTuple shape, DLDataType dtype) {
  VerifyDataType(dtype);

  // critical zone: allocate header, cannot throw
  runtime::NDArray::Container* container =
      new runtime::NDArray::Container(nullptr, shape, dtype, this->buffer.device);

  container->SetDeleter(StorageObj::Deleter);
  size_t needed_size = runtime::GetDataSize(container->dl_tensor);
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

  runtime::NDArray ret(runtime::GetObjectPtr<Object>(container));
  // RAII in effect, now run the check.

  ICHECK(offset + needed_size <= this->buffer.size)
      << "storage allocation failure, attempted to allocate " << needed_size << " at offset "
      << offset << " in region that is " << this->buffer.size << "bytes";

  return ret;
}

MemoryManager* MemoryManager::Global() {
  // NOTE: explicitly use new to avoid exit-time destruction of global state
  // Global state will be recycled by OS as the process exits.
  static auto* inst = new MemoryManager();
  return inst;
}

Allocator* MemoryManager::GetOrCreateAllocator(Device dev, AllocatorType type) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mutex_);
  if (m->allocators_.find(dev) == m->allocators_.end()) {
    std::unique_ptr<Allocator> alloc;
    switch (type) {
      case kNaive: {
        DLOG(INFO) << "New naive allocator for " << runtime::DeviceName(dev.device_type) << "("
                   << dev.device_id << ")";
        alloc.reset(new NaiveAllocator(dev));
        break;
      }
      case kPooled: {
        DLOG(INFO) << "New pooled allocator for " << runtime::DeviceName(dev.device_type) << "("
                   << dev.device_id << ")";
        alloc.reset(new PooledAllocator(dev));
        break;
      }
      default:
        LOG(FATAL) << "Unknown allocator type: " << type;
    }
    auto ret = alloc.get();
    m->allocators_.emplace(dev, std::move(alloc));
    return ret;
  }
  auto alloc = m->allocators_.at(dev).get();
  if (alloc->type() != type) {
    LOG(WARNING) << "The type of existing allocator for " << runtime::DeviceName(dev.device_type)
                 << "(" << dev.device_id << ") is different from the request type ("
                 << alloc->type() << " vs " << type << ")";
  }
  return alloc;
}

Allocator* MemoryManager::GetAllocator(Device dev) {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mutex_);
  auto it = m->allocators_.find(dev);
  if (it == m->allocators_.end()) {
    LOG(FATAL) << "Allocator for " << runtime::DeviceName(dev.device_type) << "(" << dev.device_id
               << ") has not been created yet.";
  }
  return it->second.get();
}

void MemoryManager::Clear() {
  MemoryManager* m = MemoryManager::Global();
  std::lock_guard<std::mutex> lock(m->mutex_);
  m->allocators_.clear();
}

Buffer Allocator::Alloc(ShapeTuple shape, DLDataType dtype, String mem_scope) {
  ICHECK_EQ(shape.size(), 1) << "Allocator of type (" << type_
                             << ") does not support nD allocation. Please use allocator type ("
                             << AllocatorType::kNaive << ")";
  CHECK_EQ(mem_scope, "global") << "Allocator of type (" << type_
                                << ") does not support memory scope " << mem_scope
                                << ". Please use allocator type (" << AllocatorType::kNaive << ")";

  DLTensor temp;
  temp.ndim = shape.size();
  temp.dtype = dtype;
  temp.shape = const_cast<int64_t*>(shape.data());
  temp.strides = nullptr;
  temp.byte_offset = 0;
  size_t nbytes = GetDataSize(temp);

  return Alloc(nbytes, runtime::kAllocAlignment, dtype);
}

runtime::NDArray Allocator::Empty(ShapeTuple shape, DLDataType dtype, DLDevice dev) {
  VerifyDataType(dtype);
  runtime::NDArray::Container* container =
      new runtime::NDArray::Container(nullptr, shape, dtype, dev);
  container->SetDeleter(BufferDeleter);
  size_t size = runtime::GetDataSize(container->dl_tensor);
  size_t alignment = GetDataAlignment(container->dl_tensor);
  Buffer* buffer = new Buffer;
  *buffer = this->Alloc(size, alignment, dtype);
  container->manager_ctx = reinterpret_cast<void*>(buffer);
  container->dl_tensor.data = buffer->data;
  return runtime::NDArray(runtime::GetObjectPtr<Object>(container));
}

TVM_REGISTER_GLOBAL("vm.builtin.memory_manager.clear").set_body_typed(MemoryManager::Clear);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
