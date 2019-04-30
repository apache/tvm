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
 * \file tvm/runtime/memory_manager.cc
 * \brief Allocate and manage memory for the runtime.
 */
#include <utility>
#include <memory>
#include "memory_manager.h"
#include "naive_allocator.h"
#include "pooled_allocator.h"

namespace tvm {
namespace runtime {

MemoryManager* MemoryManager::Global() {
  static MemoryManager memory_manager;
  return &memory_manager;
}

Allocator* MemoryManager::GetAllocator(TVMContext ctx) {
  std::lock_guard<std::mutex> lock(mu_);
  if (allocators_.find(ctx) == allocators_.end()) {
    // LOG(INFO) << "New allocator for " << DeviceName(ctx.device_type) << "("
    //           << ctx.device_id << ")";
    std::unique_ptr<Allocator> alloc(new NaiveAllocator(ctx));
    allocators_.emplace(ctx, std::move(alloc));
  }
  return allocators_.at(ctx).get();
}

}  // namespace runtime
}  // namespace tvm
