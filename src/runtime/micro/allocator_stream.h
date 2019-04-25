/*!
 *  Copyright (c) 2019 by Contributors
 * \file allocator_stream.h
 * \brief allocator stream utility
 */
#ifndef TVM_RUNTIME_MICRO_ALLOCATOR_STREAM_H_
#define TVM_RUNTIME_MICRO_ALLOCATOR_STREAM_H_

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <dmlc/memory_io.h>

namespace tvm {
namespace runtime {

/*!
 * \brief helper class for writing into `AllocatorStream`
 */
class Slot {
 public:
  /*!
   * \brief constructor
   * \param buf shared pointer to parent backing buffer
   * \param start_offs start byte offset of the slot in the backing buffer
   * \param size size (in bytes) of the memory region allocated for this slot
   * \param dev_start_addr start address of the slot in the device's memory
   */
  Slot(std::shared_ptr<std::vector<uint8_t>> buf, size_t start_offs, size_t size, void* dev_start_addr)
    : buf_(buf)
    , start_offs_(start_offs)
    , curr_offs_(0)
    , size_(size)
    , dev_start_addr_(dev_start_addr) {
  }

  /*!
   * \brief writes `sizeof(T)` bytes of data from `src_ptr`
   * \param src_ptr address of the buffer to be read from
   */
  template <typename T>
  void Write(const T* src_ptr) {
    Write(src_ptr, sizeof(T));
  }

  /*!
   * \brief writes `sizeof(T) * length` bytes of data from `src_ptr`
   * \param src_ptr address of the buffer to be read from
   * \param length address of the buffer to be read from
   */
  template <typename T>
  void WriteArray(const T* src_ptr, size_t length) {
    Write(src_ptr, sizeof(T) * length);
  }

  /*!
   * \brief fills this slot with data from `src_ptr`
   * \param src_ptr address of the buffer to be read from
   * \param length address of the buffer to be read from
   */
  template <typename T>
  void WriteEntire(const T* src_ptr) {
    CHECK(curr_offs_ == 0);
    Write(src_ptr, size_);
  }

  /*!
   * \brief writes `size` bytes of data from `src_ptr` into the backing buffer
   * \param src_ptr address of the buffer to be read from
   * \param size number of bytes to be written
   */
  void Write(const void* src_ptr, size_t size) {
    if (size == 0) return;
    CHECK(curr_offs_ + size <= size_);
    uint8_t* curr_ptr = &(*buf_)[start_offs_ + curr_offs_];
    std::memcpy(curr_ptr, src_ptr, size);
    curr_offs_ += size;
  }

  /*!
   * \brief returns start address of the slot in device memory
   * \return device start address
   */
  void* dev_start_addr() {
    return dev_start_addr_;
  }

  /*!
   * \brief returns number of bytes allocated for this slot
   * \return size of this slot
   */
  size_t size() {
    return size_;
  }

 private:
  // We store a pointer to the backing buffer and a byte offset, instead of just
  // a pointer at the offset into the buffer, in order to prevent stale
  // references on vector resize.

  /*! \brief shared pointer to parent backing buffer */
  std::shared_ptr<std::vector<uint8_t>> buf_;
  /*! \brief start offset of the slot in the backing buffer */
  size_t start_offs_;
  /*! \brief current offset relative to the start offset of this slot */
  size_t curr_offs_;
  /*! \brief size (in bytes) of the memory region allocated for this slot */
  size_t size_;
  /*! \brief start address of the slot in the device's memory */
  void* dev_start_addr_;
};

/*!
 * \brief allocation-based stream for uTVM args allocation
 */
class AllocatorStream {
 public:
  /*!
   * \brief constructor
   * \param dev_start_addr start address of the stream in the device's memory
   */
  explicit AllocatorStream(void* dev_start_addr)
      : buf_(std::make_shared<std::vector<uint8_t>>())
      , curr_offs_(0)
      , dev_start_addr_(dev_start_addr) {}

  /*!
   * \brief allocates a slot for `sizeof(T)` bytes of data
   * \return slot of size `sizeof(T)` bytes
   */
  template <typename T>
  Slot Alloc() {
    return Alloc(sizeof(T));
  }

  /*!
   * \brief allocates a slot for `sizeof(T) * length` bytes of data
   * \param length number of elements in the array being allocated for
   * \return slot of size `sizeof(T) * length` bytes
   */
  template <typename T>
  Slot AllocArray(size_t length) {
    return Alloc(sizeof(T) * length);
  }

  /*!
   * \brief allocates a slot for `size` bytes of data
   * \param size number of bytes to allocate
   * \return slot of size `size` bytes
   */
  Slot Alloc(size_t size) {
    if (curr_offs_ + size > buf_->size()) {
      buf_->resize(curr_offs_ + size);
    }
    size_t slot_start_offs = curr_offs_;
    curr_offs_ += size;
    return Slot(buf_, slot_start_offs, size, GetDevAddr(slot_start_offs));
  }

  /*!
   * \brief returns the corresponding device address for the offset `offset`
   * \param offset byte offset from the beginning of the backing buffer
   * \return device address
   */
  void* GetDevAddr(size_t offset) {
    return reinterpret_cast<uint8_t*>(dev_start_addr_) + offset;
  }

  /*!
   * \brief returns the array backing the stream's buffer
   * \return array backing the stream's buffer
   */
  const uint8_t* data() {
    return buf_->data();
  }

  /*!
   * \brief returns current size of the stream buffer
   * \return buffer size
   */
  size_t size() {
    return buf_->size();
  }

 private:
  /*! \brief in-memory backing buffer */
  std::shared_ptr<std::vector<uint8_t>> buf_;
  /*! \brief current offset */
  size_t curr_offs_;
  /*! \brief on-device start address */
  void* dev_start_addr_;
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_ALLOCATOR_STREAM_H_
