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
 * \file tvm/support/io.h
 * \brief Binary stream I/O interface.
 */
#ifndef TVM_SUPPORT_IO_H_
#define TVM_SUPPORT_IO_H_

#include <cstddef>
#include <cstdint>

namespace tvm {
namespace support {

/*!
 * \brief Primary Serializer template. Specialize for each serializable type.
 *
 * A valid specialization must provide:
 *   static constexpr bool enabled = true;
 *   static void Write(Stream* strm, const T& data);
 *   static bool Read(Stream* strm, T* data);
 */
template <typename T, typename = void>
struct Serializer {
  static constexpr bool enabled = false;
};

/*!
 * \brief Abstract binary stream for serialization.
 *
 * Subclasses implement the raw Read/Write byte methods.
 * The template Write/Read methods delegate to Serializer
 * for endian-aware, type-safe binary I/O.
 *
 * \note Subclasses that override Read(void*, size_t) or Write(const void*, size_t)
 *       must add `using Stream::Read;` and `using Stream::Write;` to make the
 *       template overloads visible (C++ name-hiding rule).
 */
class Stream {
 public:
  /*!
   * \brief Read raw bytes from the stream.
   * \param ptr  Destination buffer.
   * \param size Number of bytes to read.
   * \return Number of bytes actually read.
   */
  virtual size_t Read(void* ptr, size_t size) = 0;

  /*!
   * \brief Write raw bytes to the stream.
   * \param ptr  Source buffer.
   * \param size Number of bytes to write.
   * \return Number of bytes actually written.
   */
  virtual size_t Write(const void* ptr, size_t size) = 0;

  /*! \brief Virtual destructor. */
  virtual ~Stream() = default;

  /*!
   * \brief Write a typed value using Serializer<T>.
   * \tparam T The data type (must have a Serializer<T> specialization).
   * \param data The value to write.
   */
  template <typename T>
  void Write(const T& data) {
    static_assert(Serializer<T>::enabled, "No Serializer<T> specialization for this type");
    Serializer<T>::Write(this, data);
  }

  /*!
   * \brief Read a typed value using Serializer<T>.
   * \tparam T The data type (must have a Serializer<T> specialization).
   * \param out_data Pointer to receive the value.
   * \return true on success.
   */
  template <typename T>
  bool Read(T* out_data) {
    static_assert(Serializer<T>::enabled, "No Serializer<T> specialization for this type");
    return Serializer<T>::Read(this, out_data);
  }

  /*!
   * \brief Write an array of typed values element by element.
   * \param data Pointer to the first element.
   * \param num_elems Number of elements.
   */
  template <typename T>
  void WriteArray(const T* data, size_t num_elems) {
    for (size_t i = 0; i < num_elems; ++i) Write<T>(data[i]);
  }

  /*!
   * \brief Read an array of typed values element by element.
   * \param data Pointer to the first element.
   * \param num_elems Number of elements.
   * \return true on success.
   */
  template <typename T>
  bool ReadArray(T* data, size_t num_elems) {
    for (size_t i = 0; i < num_elems; ++i) {
      if (!Read<T>(data + i)) return false;
    }
    return true;
  }
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_IO_H_
