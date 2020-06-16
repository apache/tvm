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

#include "target_data_layout_encoder.h"

namespace tvm {
namespace runtime {

TargetDataLayoutEncoder::Alloc::Alloc(TargetDataLayoutEncoder* parent, size_t start_offset,
                                      size_t size, TargetPtr start_addr)
    : parent_(parent),
      start_offset_(start_offset),
      curr_offset_(0),
      size_(size),
      start_addr_(start_addr) {
  parent_->live_unchecked_allocs_.insert(this);
}

TargetDataLayoutEncoder::Alloc::~Alloc() {
  auto it = parent_->live_unchecked_allocs_.find(this);
  if (it != parent_->live_unchecked_allocs_.end()) {
    // alloc was not already checked
    parent_->live_unchecked_allocs_.erase(it);
    if (curr_offset_ != size_) {
      parent_->unchecked_alloc_start_offsets_.push_back(start_addr_.value().uint64());
    }
  }
}

void TargetDataLayoutEncoder::Alloc::CheckUnfilled() {
  CHECK(curr_offset_ == size_) << "unwritten space in alloc 0x" << std::hex
                               << start_addr_.value().uint64() << "; curr_offset=0x" << curr_offset_
                               << ", size=0x" << size_;
}

TargetPtr TargetDataLayoutEncoder::Alloc::start_addr() { return start_addr_; }

size_t TargetDataLayoutEncoder::Alloc::size() { return size_; }

void TargetDataLayoutEncoder::CheckUnfilledAllocs() {
  CHECK(live_unchecked_allocs_.size() > 0) << "No allocs to check";
  if (unchecked_alloc_start_offsets_.size() > 0) {
    LOG(ERROR) << "Unchecked allocs were found:";
    for (size_t alloc_start_addr : unchecked_alloc_start_offsets_) {
      LOG(ERROR) << " * 0x" << std::hex << alloc_start_addr;
    }
    CHECK(false) << "Unchecked allocs found during CheckUnfilledAllocs";
  }

  for (class Alloc* s : live_unchecked_allocs_) {
    s->CheckUnfilled();
  }
  live_unchecked_allocs_.clear();
}

}  // namespace runtime
}  // namespace tvm
