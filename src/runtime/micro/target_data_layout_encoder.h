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

#include <vector>
#include "host_driven/utvm_runtime.h"

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
  template <typename T>
  class Slot {
   public:
    /*!
     * \brief constructor
     * \param parent pointer to parent encoder
     * \param start_offset start byte offset of the slot in the backing buffer
     * \param size size (in bytes) of the memory region allocated for this slot
     * \param start_addr start address of the slot in the device's memory
     */
    Slot(TargetDataLayoutEncoder* parent, size_t start_offset, size_t size, TargetPtr start_addr);

    ~Slot();

    /*!
     * \brief writes `sizeof(T) * num_elems` bytes of data from `arr`
     * \param arr array to be read from
     * \param num_elems number of elements in array
     */
    void WriteArray(const T* arr, size_t num_elems);

    /*!
     * \brief writes `val`
     * \param val value to be written
     */
    void WriteValue(const T& val);

    /*!
     * \brief returns start address of the slot in device memory
     * \return device start address
     */
    TargetPtr start_addr();

    /*!
     * \brief returns number of bytes allocated for this slot
     * \return size of this slot
     */
    size_t size();

   private:
    /*! \brief pointer to parent encoder */
    TargetDataLayoutEncoder* parent_;
    /*! \brief start offset of the slot in the parent's backing parent_buffer */
    size_t start_offset_;
    /*! \brief current offset relative to the start offset of this slot */
    size_t curr_offset_;
    /*! \brief size (in bytes) of the memory region allocated for this slot */
    size_t size_;
    /*! \brief start address of the slot in the device's memory */
    TargetPtr start_addr_;
  };

  /*!
   * \brief constructor
   * \param start_addr start address of the encoder in device memory
   */
  explicit TargetDataLayoutEncoder(size_t capacity, TargetWordSize word_size)
      : buf_(std::vector<uint8_t>()), curr_offset_(0),
        start_addr_(word_size, nullptr),
        capacity_(capacity), word_size_(word_size) {
  }

  /*!
   * \brief allocates a slot for `sizeof(T) * num_elems` bytes of data
   * \param num_elems number of elements of type `T` being allocated (defaults to 1)
   * \return slot of size `sizeof(T) * num_elems` bytes
   */
  template <typename T>
  Slot<T> Alloc(size_t num_elems = 1) {
    curr_offset_ = UpperAlignValue(curr_offset_, word_size_.bytes());
    size_t size = sizeof(T) * num_elems;
    if (curr_offset_ + size > buf_.size()) {
      buf_.resize(curr_offset_ + size);
    }
    CHECK(buf_.size() < capacity_) << "out of space in data encoder";
    size_t slot_start_offset = curr_offset_;
    curr_offset_ += size;
    return Slot<T>(this, slot_start_offset, size, start_addr() + slot_start_offset);
  }

  void Clear() {
    buf_.clear();
    curr_offset_ = 0;
  }

  /*!
   * \brief returns the array backing the encoder's buffer
   * \return array backing the encoder's buffer
   */
  uint8_t* data() {
    return buf_.data();
  }

  /*!
   * \brief returns current size of the encoder's buffer
   * \return buffer size
   */
  size_t buf_size() const {
    return buf_.size();
  }

  TargetPtr start_addr() const {
    CHECK_NE(start_addr_.value().uint64(), 0) << "start addr uninitialized";
    return start_addr_;
  }

  void set_start_addr(TargetPtr start_addr) {
    CHECK_EQ(buf_.size(), 0) << "cannot change encoder start addr unless empty";
    start_addr_ = TargetPtr(word_size_,
                            UpperAlignValue(start_addr.value().uint64(), word_size_.bytes()));
  }

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
};

template <typename T>
TargetDataLayoutEncoder::Slot<T>::Slot(TargetDataLayoutEncoder* parent,
                                       size_t start_offset,
                                       size_t size,
                                       TargetPtr start_addr)
    : parent_(parent),
      start_offset_(start_offset),
      curr_offset_(0),
      size_(size),
      start_addr_(start_addr) {}

template <typename T>
TargetDataLayoutEncoder::Slot<T>::~Slot() {
  // TODO(weberlo, areusch): this can mask the exception thrown by slot allocation... even though
  // that doesn't make sense.
  CHECK(curr_offset_ == size_) << "unwritten space in slot; curr_offset="
                               << curr_offset_ << ", size=" << size_;
}

template <typename T>
void TargetDataLayoutEncoder::Slot<T>::WriteArray(const T* arr, size_t num_elems) {
  if (num_elems == 0) return;
  size_t size = sizeof(T) * num_elems;
  CHECK(curr_offset_ + size <= size_) << "not enough space in slot";
  uint8_t* curr_ptr = &(parent_->data())[start_offset_ + curr_offset_];
  std::memcpy(curr_ptr, arr, size);
  curr_offset_ += size;
}

template <typename T>
void TargetDataLayoutEncoder::Slot<T>::WriteValue(const T& val) {
  WriteArray(&val, 1);
}

template <typename T>
TargetPtr TargetDataLayoutEncoder::Slot<T>::start_addr() {
  return start_addr_;
}

template <typename T>
size_t TargetDataLayoutEncoder::Slot<T>::size() {
  return size_;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_
