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
 * \file support/bytes_io.h
 * \brief In-memory byte stream implementations.
 *
 *   BytesInStream  — sequential read from a fixed memory region
 *   BytesOutStream — sequential write to a caller-provided std::string
 */
#ifndef TVM_SUPPORT_BYTES_IO_H_
#define TVM_SUPPORT_BYTES_IO_H_

#include <tvm/ffi/error.h>
#include <tvm/ffi/string.h>
#include <tvm/support/io.h>

#include <algorithm>
#include <cstring>
#include <string>

namespace tvm {
namespace support {

/*!
 * \brief Read-only stream over a fixed memory region.
 */
class BytesInStream : public Stream {
 public:
  using Stream::Read;
  using Stream::Write;

  BytesInStream(const void* data, size_t size)
      : data_(reinterpret_cast<const char*>(data)), size_(size), pos_(0) {}

  explicit BytesInStream(const ffi::Bytes& bytes)
      : data_(bytes.data()), size_(bytes.size()), pos_(0) {}

  explicit BytesInStream(const std::string& str) : data_(str.data()), size_(str.size()), pos_(0) {}

  size_t Read(void* ptr, size_t size) override {
    size_t nread = std::min(size_ - pos_, size);
    if (nread != 0) std::memcpy(ptr, data_ + pos_, nread);
    pos_ += nread;
    return nread;
  }

  size_t Write(const void*, size_t) override {
    TVM_FFI_THROW(InternalError) << "write to input stream";
    return 0;
  }

 private:
  const char* data_;
  size_t size_;
  size_t pos_;
};

/*!
 * \brief Write-only stream that appends to a caller-provided std::string.
 */
class BytesOutStream : public Stream {
 public:
  using Stream::Read;
  using Stream::Write;

  explicit BytesOutStream(std::string* buffer) : buffer_(buffer) {}

  size_t Write(const void* ptr, size_t size) override {
    if (size == 0) return 0;
    buffer_->append(reinterpret_cast<const char*>(ptr), size);
    return size;
  }

  size_t Read(void*, size_t) override {
    TVM_FFI_THROW(InternalError) << "read from output stream";
    return 0;
  }

 private:
  std::string* buffer_;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_BYTES_IO_H_
