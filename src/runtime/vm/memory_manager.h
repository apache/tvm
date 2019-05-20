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
 * \file src/runtime/memory_manager.h
 * \brief Abstract device memory management API
 */
#ifndef TVM_RUNTIME_VM_MEMORY_MANAGER_H_
#define TVM_RUNTIME_VM_MEMORY_MANAGER_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace std {
template <>
struct hash<TVMContext> {
  std::size_t operator()(const TVMContext& ctx) const {
    return ((ctx.device_id << 8) | ctx.device_type);
  }
};

template <>
struct equal_to<TVMContext> {
  bool operator()(const TVMContext& lhs, const TVMContext& rhs) const {
    return (lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id);
  }
};

}  // namespace std

namespace tvm {
namespace runtime {
namespace vm {

struct Buffer {
  /*! \brief The pointer to the allocated block of memory. */
  void* data{nullptr};
  /*! \brief The size of the block. */
  size_t size{0};
  /*! \brief The context of the allocated buffers. */
  TVMContext ctx;
};

class Allocator {
 public:
  Allocator() {}

  /*! \brief Allocate an empty NDArray using from the allocator.
   *  \param shape The shape of the NDArray.
   *  \param alignment The datatype of the NDArray.
   *  \param ctx The context where the array is allocated.
   *  \return The empty NDArray.
   */
  NDArray Empty(std::vector<int64_t> shape,
                DLDataType dtype,
                DLContext ctx);
  /*! \brief Allocate a buffer given a size, alignment and type.
   *  \param nbytes The size of the buffer.
   *  \param alignment The alignment of the buffer.
   *  \param type_hint A type hint to the allocator.
   *  \return A sized allocation in the form of a buffer.
  */
  virtual Buffer Alloc(size_t nbytes, size_t alignment, TVMType type_hint) = 0;
  /*! \brief Free a buffer allocated by the allocator.
   *  \param buffer The buffer to free.
   */
  virtual void Free(const Buffer& buffer) = 0;
  /*! \brief The amount of memory currently allocated.
   *  \return The amount of memory currently allocated.
   */
  virtual size_t UsedMemory() const = 0;
  virtual ~Allocator() = default;
};

class MemoryManager {
 public:
  static MemoryManager* Global();

  Allocator* GetAllocator(TVMContext ctx);

 private:
  MemoryManager() {}

 private:
  std::mutex mu_;
  std::unordered_map<TVMContext, std::unique_ptr<Allocator> > allocators_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_MEMORY_MANAGER_H_
