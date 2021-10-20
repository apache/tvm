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
 * \file src/runtime/contrib/arm_compute_lib/acl_allocator.cc
 * \brief ACL Allocator implementation that requests memory from TVM.
 */

#include "acl_allocator.h"

namespace tvm {
namespace runtime {
namespace contrib {

void* ACLAllocator::allocate(size_t size, size_t alignment) {
  ICHECK_GT(size, 0) << "Cannot allocate size less than or equal to zero";
  return this->device_api_->AllocWorkspace(this->device_, size, {});
}

void ACLAllocator::free(void* ptr) { this->device_api_->FreeWorkspace(this->device_, ptr); }

std::unique_ptr<arm_compute::IMemoryRegion> ACLAllocator::make_region(size_t size,
                                                                      size_t alignment) {
  return std::make_unique<ACLMemoryRegion>(size, alignment);
}

ACLMemoryRegion::ACLMemoryRegion(size_t size, size_t alignment)
    : IMemoryRegion(size), ptr_(nullptr) {
  if (size != 0) {
    this->ptr_ = this->device_api_->AllocDataSpace(this->device_, size, alignment, {});
  }
}

ACLMemoryRegion::ACLMemoryRegion(void* ptr, size_t size)
    : IMemoryRegion(size), ptr_(nullptr), is_subregion_(true) {
  if (size != 0) {
    this->ptr_ = ptr;
  }
}

ACLMemoryRegion::~ACLMemoryRegion() {
  if (this->ptr_ != nullptr && !is_subregion_) {
    this->device_api_->FreeDataSpace(this->device_, this->ptr_);
  }
}

std::unique_ptr<arm_compute::IMemoryRegion> ACLMemoryRegion::extract_subregion(size_t offset,
                                                                               size_t size) {
  if (this->ptr_ != nullptr && (offset < _size) && (_size - offset >= size)) {
    return std::make_unique<ACLMemoryRegion>(static_cast<uint8_t*>(this->ptr_) + offset, size);
  } else {
    return nullptr;
  }
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
