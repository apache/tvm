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
 * \file file_io.h
 * \brief Defines a simple DMLC stream for file io.
 */
#include <dmlc/io.h>

#ifndef SUPPORT_FILE_IO_H_
#define SUPPORT_FILE_IO_H_

namespace tvm {
namespace support {

/*!
 * \brief A dmlc stream which wraps standard file operations.
 */
struct SimpleBinaryFileStream : public dmlc::Stream {
 public:
  SimpleBinaryFileStream(const std::string& path, bool read) {
    const char* fname = path.c_str();
    if (read) {
      fp_ = std::fopen(fname, "rb");
    } else {
      fp_ = std::fopen(fname, "wb");
    }
    CHECK(fp_) << "Unable to open file " << path;
    read_ = read;
  }
  virtual ~SimpleBinaryFileStream(void) { this->Close(); }
  virtual size_t Read(void* ptr, size_t size) {
    CHECK(read_) << "File opened in write-mode, cannot read.";
    return std::fread(ptr, 1, size, fp_);
  }
  virtual void Write(const void* ptr, size_t size) {
    CHECK(!read_) << "File opened in read-mode, cannot write.";
    CHECK(std::fwrite(ptr, 1, size, fp_) == size) << "SimpleBinaryFileStream.Write incomplete";
  }
  inline void Close(void) {
    if (fp_ != NULL) {
      std::fclose(fp_);
      fp_ = NULL;
    }
  }

 private:
  std::FILE* fp_ = nullptr;
  bool read_;  // if false, then in write mode.
};             // class SimpleBinaryFileStream
}  // namespace support
}  // namespace tvm
#endif  // SUPPORT_FILE_IO_H_
