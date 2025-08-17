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
 * \file buffer_stream.h
 * \brief Internal minimal stream helper to read from a buffer.
 */
#ifndef TVM_FFI_EXTRA_BUFFER_STREAM_H_
#define TVM_FFI_EXTRA_BUFFER_STREAM_H_

#include <algorithm>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace tvm {
namespace ffi {

/*!
 * \brief Lightweight stream helper to read from a buffer.
 */
class BufferInStream {
 public:
  /*!
   * \brief constructor
   * \param p_buffer the head pointer of the memory region.
   * \param buffer_size the size of the memorybuffer
   */
  BufferInStream(const void* data, size_t size)
      : data_(reinterpret_cast<const char*>(data)), size_(size) {}
  /*!
   * \brief Reads raw from stream.
   * \param ptr pointer to the data to be read
   * \param size the size of the data to be read
   * \return the number of bytes read
   */
  size_t Read(void* ptr, size_t size) {
    size_t nread = std::min(size_ - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, data_ + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  /*!
   * \brief Reads arithmetic data from stream in endian-aware manner.
   * \param data data to be read
   * \tparam T the data type to be read
   * \return whether the read was successful
   */
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
  bool Read(T* data) {
    bool ret = Read(static_cast<void*>(data), sizeof(T)) == sizeof(T);  // NOLINT(*)
    if (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
      ByteSwap(&data, sizeof(T), 1);
    }
    return ret;
  }
  /*!
   * \brief Reads an array of data from stream in endian-aware manner.
   * \param data data to be read
   * \param size the size of the data to be read
   * \return whether the read was successful
   */
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
  bool ReadArray(T* data, size_t size) {
    bool ret =
        this->Read(static_cast<void*>(data), sizeof(T) * size) == sizeof(T) * size;  // NOLINT(*)
    if (!TVM_FFI_IO_NO_ENDIAN_SWAP) {
      ByteSwap(data, sizeof(T), size);
    }
    return ret;
  }
  /*!
   * \brief Reads a string from stream.
   * \param data data to be read
   * \return whether the read was successful
   */
  bool Read(std::string* data) {
    // use uint64_t to ensure platform independent size
    uint64_t size = 0;
    if (!this->Read<uint64_t>(&size)) return false;
    data->resize(size);
    if (!this->Read(data->data(), size)) return false;
    return true;
  }
  /*!
   * \brief Reads a vector of data from stream in endian-aware manner.
   * \param data data to be read
   * \return whether the read was successful
   */
  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
  bool Read(std::vector<T>* data) {
    uint64_t size = 0;
    if (!this->Read<uint64_t>(&size)) return false;
    data->resize(size);
    return this->ReadArray(data->data(), size);
  }

 private:
  /*! \brief in memory buffer */
  const char* data_;
  /*! \brief size of the buffer */
  size_t size_;
  /*! \brief current pointer */
  size_t curr_ptr_{0};
};  // class BytesInStream

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_BUFFER_STREAM_H_
