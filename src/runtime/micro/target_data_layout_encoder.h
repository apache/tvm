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
 * \file target_data_layout_encoder.h
 * \brief uTVM data layout encoder
 */
#ifndef TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_
#define TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_

#include <memory>
#include <set>
#include <vector>

#include "host_driven/utvm_runtime_enum.h"
#include "micro_common.h"

namespace tvm {
namespace runtime {

// TODO(weberlo, areusch): Handle endianness.

/*!
 * \brief data encoder for uTVM that builds a host-side buffer
 */
class TargetDataLayoutEncoder {
 public:
  /*!
   * \brief helper class for writing into `TargetDataLayoutEncoder`
   */
  class Alloc {
   public:
    /*!
     * \brief constructor
     * \param parent pointer to parent encoder
     * \param start_offset start byte offset of the alloc in the backing buffer
     * \param size size (in bytes) of the memory region allocated for this alloc
     * \param start_addr start address of the alloc in the device's memory
     */
    Alloc(TargetDataLayoutEncoder* parent, size_t start_offset, size_t size, TargetPtr start_addr);

    ~Alloc();

    /*!
     * \brief writes `sizeof(T) * num_elems` bytes of data from `arr`
     * \param arr array to be read from
     * \param num_elems number of elements in array
     */
    template <typename T>
    void WriteArray(const T* arr, size_t num_elems);

    /*!
     * \brief writes `val`
     * \param val value to be written
     */
    template <typename T>
    void WriteValue(const T& val);

    /*!
     * \brief returns start address of the alloc in device memory
     * \return device start address
     */
    TargetPtr start_addr();

    /*!
     * \brief returns number of bytes allocated for this alloc
     * \return size of this alloc
     */
    size_t size();

    size_t curr_offset() const { return curr_offset_; }

    void CheckUnfilled();

   private:
    /*! \brief pointer to parent encoder */
    TargetDataLayoutEncoder* parent_;
    /*! \brief start offset of the alloc in the parent's backing parent_buffer */
    size_t start_offset_;
    /*! \brief current offset relative to the start offset of this alloc */
    size_t curr_offset_;
    /*! \brief size (in bytes) of the memory region allocated for this alloc */
    size_t size_;
    /*! \brief start address of the alloc in the device's memory */
    TargetPtr start_addr_;
  };

  /*!
   * \brief constructor
   * \param start_addr start address of the encoder in device memory
   */
  explicit TargetDataLayoutEncoder(size_t capacity, TargetWordSize word_size)
      : buf_(std::vector<uint8_t>()),
        curr_offset_(0),
        start_addr_(word_size, nullptr),
        capacity_(capacity),
        word_size_(word_size) {}

  /*!
   * \brief allocates a alloc for `sizeof(T) * num_elems` bytes of data
   * \param num_elems number of elements of type `T` being allocated (defaults to 1)
   * \return alloc of size `sizeof(T) * num_elems` bytes
   */
  template <typename T>
  std::unique_ptr<class Alloc> Alloc(size_t num_elems = 1) {
    curr_offset_ = UpperAlignValue(curr_offset_, word_size_.bytes());
    size_t size = sizeof(T) * num_elems;
    if (curr_offset_ + size > buf_.size()) {
      buf_.resize(curr_offset_ + size);
    }
    CHECK(buf_.size() < capacity_) << "out of space in data encoder";
    size_t alloc_start_offset = curr_offset_;
    curr_offset_ += size;
    class Alloc* alloc =
        new class Alloc(this, alloc_start_offset, size, start_addr() + alloc_start_offset);
    return std::unique_ptr<class Alloc>(alloc);
  }

  void Clear() {
    buf_.clear();
    curr_offset_ = 0;
  }

  /*!
   * \brief returns the array backing the encoder's buffer
   * \return array backing the encoder's buffer
   */
  uint8_t* data() { return buf_.data(); }

  /*!
   * \brief returns current size of the encoder's buffer
   * \return buffer size
   */
  size_t buf_size() const { return buf_.size(); }

  TargetPtr start_addr() const {
    CHECK_NE(start_addr_.value().uint64(), 0) << "start addr uninitialized";
    return start_addr_;
  }

  void set_start_addr(TargetPtr start_addr) {
    CHECK_EQ(buf_.size(), 0) << "cannot change encoder start addr unless empty";
    start_addr_ =
        TargetPtr(word_size_, UpperAlignValue(start_addr.value().uint64(), word_size_.bytes()));
  }

  void CheckUnfilledAllocs();

 private:
  /*! \brief in-memory backing buffer */
  std::vector<uint8_t> buf_;
  /*! \brief current offset */
  size_t curr_offset_;
  /*! \brief start address of the encoder in device memory */
  TargetPtr start_addr_;
  /*! \brief number of bytes available in device memory */
  size_t capacity_;
  /*! \brief number of bytes in a word on the target device */
  TargetWordSize word_size_;
  /*! \brief Alloc instances allocated now but not yet checked by CheckUnfilledAllocs */
  std::set<class Alloc*> live_unchecked_allocs_;
  /*! \brief start offsets Alloc instances that were dealloated before CheckUnfilledAllocs ran */
  std::vector<size_t> unchecked_alloc_start_offsets_;
  friend Alloc::~Alloc();
};

template <typename T>
void TargetDataLayoutEncoder::Alloc::WriteArray(const T* arr, size_t num_elems) {
  if (num_elems == 0) return;
  size_t size = sizeof(T) * num_elems;
  CHECK(curr_offset_ + size <= size_) << "not enough space in alloc";
  uint8_t* curr_ptr = &(parent_->data())[start_offset_ + curr_offset_];
  std::memcpy(curr_ptr, arr, size);
  curr_offset_ += size;
}

template <typename T>
void TargetDataLayoutEncoder::Alloc::WriteValue(const T& val) {
  WriteArray(&val, 1);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_
