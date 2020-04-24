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
 * \file micro_section_allocator.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_SECTION_ALLOCATOR_H_
#define TVM_RUNTIME_MICRO_MICRO_SECTION_ALLOCATOR_H_

#include <string>
#include <unordered_map>
#include "micro_common.h"

namespace tvm {
namespace runtime {

/*!
 * \brief allocator for an on-device memory section
 */
class MicroSectionAllocator {
 public:
  /*!
   * \brief constructor that specifies section boundaries
   * \param region location and size of the section on the device
   */
  explicit MicroSectionAllocator(std::string section_name,
                                 DevMemRegion region,
                                 TargetWordSize word_size)
    : section_name_(section_name),
      start_addr_(region.start),
      size_(0),
      capacity_(region.size),
      word_size_(word_size) {
      CHECK_EQ(start_addr_.value().uint64() % word_size.bytes(), 0)
        << "micro section start not aligned to " << word_size.bytes() << " bytes";
      CHECK_EQ(capacity_ % word_size.bytes(), 0)
        << "micro section end not aligned to " << word_size.bytes() << " bytes";
    }

  /*!
   * \brief destructor
   */
  ~MicroSectionAllocator() {}

  /*!
   * \brief memory allocator
   * \param alloc_size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  TargetPtr Allocate(size_t size) {
    size_ = UpperAlignValue(size_, word_size_.bytes());
    CHECK(size_ + size < capacity_)
      << "cannot alloc " << size << " bytes in section \""
      << section_name_ << "\" (start_addr=" << start_addr_.cast_to<void*>()
      << ", used=" << size_ << ", capacity=" << capacity_ << ")";
    TargetPtr alloc_addr = start_addr_ + size_;
    size_ += size;
    alloc_map_[alloc_addr.value().uint64()] = size;
    return alloc_addr;
  }

  /*!
   * \brief free prior allocation from section
   * \param offs offset to allocated memory
   * \note simple allocator scheme, more complex versions will be implemented later
   */
  void Free(TargetPtr addr) {
    CHECK(alloc_map_.find(addr.value().uint64()) != alloc_map_.end())
      << "freed pointer was never allocated";
    alloc_map_.erase(addr.value().uint64());
    if (alloc_map_.empty()) {
      size_ = 0;
    }
  }

  /*!
   * \brief start offset of the memory region managed by this allocator
   */
  TargetPtr start_addr() const { return start_addr_; }

  /*!
   * \brief current end addr of the space being used in this memory region
   */
  TargetPtr curr_end_addr() const { return start_addr_ + size_; }

  /*!
   * \brief end addr of the memory region managed by this allocator
   */
  TargetPtr max_addr() const { return start_addr_ + capacity_; }

  /*!
   * \brief size of the section
   */
  size_t size() const { return size_; }

  /*!
   * \brief capacity of the section
   */
  size_t capacity() const { return capacity_; }

 private:
  /*! \brief name of the section (for debugging) */
  std::string section_name_;
  /*! \brief start address of the section */
  TargetPtr start_addr_;
  /*! \brief current size of the section */
  size_t size_;
  /*! \brief total storage capacity of the section */
  size_t capacity_;
  /*! \brief number of bytes in a word on the target device */
  TargetWordSize word_size_;
  /*! \brief allocation map for allocation sizes */
  std::unordered_map<uint64_t, size_t> alloc_map_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_SECTION_ALLOCATOR_H_
