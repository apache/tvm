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
 * \file ring_buffer.h
 * \brief this file aims to provide a wrapper of sockets
 */
#ifndef TVM_SUPPORT_RING_BUFFER_H_
#define TVM_SUPPORT_RING_BUFFER_H_

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace tvm {
namespace support {
/*!
 * \brief Ring buffer class for data buffering in IO.
 *  Enables easy usage for sync and async mode.
 */
class RingBuffer {
 public:
  /*! \brief Initial capacity of ring buffer. */
  static const int kInitCapacity = 4 << 10;
  /*! \brief constructor */
  RingBuffer() : ring_(kInitCapacity) {}
  /*! \return number of bytes available in buffer. */
  size_t bytes_available() const { return bytes_available_; }
  /*! \return Current capacity of buffer. */
  size_t capacity() const { return ring_.size(); }
  /*!
   *  Reserve capacity to be at least n.
   *  Will only increase capacity if n is bigger than current capacity.
   *
   *  The effect of Reserve only lasts before the next call to Reserve.
   *  Other functions in the ring buffer can also call into the reserve.
   *
   * \param n The size of capacity.
   */
  void Reserve(size_t n) {
    if (ring_.size() < n) {
      size_t old_size = ring_.size();
      size_t new_size = static_cast<size_t>(n * 1.2);
      ring_.resize(new_size);
      if (head_ptr_ + bytes_available_ > old_size) {
        // copy the ring overflow part into the tail.
        size_t ncopy = head_ptr_ + bytes_available_ - old_size;
        if (old_size + ncopy > ring_.size()) {
          ring_.resize(old_size + ncopy);
        }
        memcpy(&ring_[0] + old_size, &ring_[0], ncopy);
      }
    } else if (ring_.size() > n * 8 && ring_.size() > kInitCapacity) {
      // shrink too large temporary buffer to
      // avoid out of memory on some embedded devices
      if (bytes_available_ != 0) {
        // move existing bytes to the head.
        size_t old_bytes = bytes_available_;
        std::vector<char> tmp(old_bytes);
        Read(&tmp[0], old_bytes);

        memcpy(&ring_[0], &tmp[0], old_bytes);
        bytes_available_ = old_bytes;
      }
      // shrink the ring.
      size_t new_size = kInitCapacity;
      new_size = std::max(new_size, n);
      new_size = std::max(new_size, bytes_available_);

      ring_.resize(new_size);
      ring_.shrink_to_fit();
      head_ptr_ = 0;
    }
  }

  /*!
   * \brief Perform a non-blocking read from buffer
   *  size must be smaller than this->bytes_available()
   * \param data the data pointer.
   * \param size The number of bytes to read.
   */
  void Read(void* data, size_t size) {
    ICHECK_GE(bytes_available_, size);
    size_t ncopy = std::min(size, ring_.size() - head_ptr_);
    memcpy(data, &ring_[0] + head_ptr_, ncopy);
    if (ncopy < size) {
      memcpy(reinterpret_cast<char*>(data) + ncopy, &ring_[0], size - ncopy);
    }
    head_ptr_ = (head_ptr_ + size) % ring_.size();
    bytes_available_ -= size;
    if (bytes_available_ == 0) {
      head_ptr_ = 0;
    }
  }
  /*!
   * \brief Read data from buffer with and put them to non-blocking send function.
   *
   * \param fsend A send function handle to put the data to.
   * \param max_nbytes Maximum number of bytes can to read.
   * \tparam FSend A non-blocking function with signature size_t (const void* data, size_t size);
   */
  template <typename FSend>
  size_t ReadWithCallback(FSend fsend, size_t max_nbytes) {
    size_t size = std::min(max_nbytes, bytes_available_);
    ICHECK_NE(size, 0U);
    size_t ncopy = std::min(size, ring_.size() - head_ptr_);
    size_t nsend = fsend(&ring_[0] + head_ptr_, ncopy);
    if (ncopy == nsend && ncopy < size) {
      size_t nsend2 = fsend(&ring_[0], size - ncopy);
      nsend += nsend2;
    }
    head_ptr_ = (head_ptr_ + nsend) % ring_.size();
    bytes_available_ -= nsend;
    if (bytes_available_ == 0) {
      head_ptr_ = 0;
    }
    return nsend;
  }
  /*!
   * \brief Write data into buffer, always ensures all data is written.
   * \param data The data pointer
   * \param size The size of data to be written.
   */
  void Write(const void* data, size_t size) {
    this->Reserve(bytes_available_ + size);
    size_t tail = head_ptr_ + bytes_available_;
    if (tail >= ring_.size()) {
      memcpy(&ring_[0] + (tail - ring_.size()), data, size);
    } else {
      size_t ncopy = std::min(ring_.size() - tail, size);
      memcpy(&ring_[0] + tail, data, ncopy);
      if (ncopy < size) {
        memcpy(&ring_[0], reinterpret_cast<const char*>(data) + ncopy, size - ncopy);
      }
    }
    bytes_available_ += size;
  }
  /*!
   * \brief Written data into the buffer by give it a non-blocking callback function.
   *
   * \param frecv A receive function handle
   * \param max_nbytes Maximum number of bytes can write.
   * \tparam FRecv A non-blocking function with signature size_t (void* data, size_t size);
   */
  template <typename FRecv>
  size_t WriteWithCallback(FRecv frecv, size_t max_nbytes) {
    this->Reserve(bytes_available_ + max_nbytes);
    size_t nbytes = max_nbytes;
    size_t tail = head_ptr_ + bytes_available_;
    if (tail >= ring_.size()) {
      size_t nrecv = frecv(&ring_[0] + (tail - ring_.size()), nbytes);
      bytes_available_ += nrecv;
      return nrecv;
    } else {
      size_t ncopy = std::min(ring_.size() - tail, nbytes);
      size_t nrecv = frecv(&ring_[0] + tail, ncopy);
      bytes_available_ += nrecv;
      if (nrecv == ncopy && ncopy < nbytes) {
        size_t nrecv2 = frecv(&ring_[0], nbytes - ncopy);
        bytes_available_ += nrecv2;
        nrecv += nrecv2;
      }
      return nrecv;
    }
  }

 private:
  // buffer head
  size_t head_ptr_{0};
  // number of bytes occupied in the buffer.
  size_t bytes_available_{0};
  // The internal data ring.
  std::vector<char> ring_;
};
}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_RING_BUFFER_H_
