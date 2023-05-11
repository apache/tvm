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
 *
 * \file base64.h
 * \brief data stream support to input and output from/to base64 stream
 *   base64 is easier to store and pass as text format in mapreduce
 */
#ifndef TVM_SUPPORT_BASE64_H_
#define TVM_SUPPORT_BASE64_H_

#include <tvm/runtime/logging.h>

#include <cctype>
#include <cstdio>
#include <string>

namespace tvm {
namespace support {
/*! \brief namespace of base64 decoding and encoding table */
namespace base64 {
// decoding table
const char DecodeTable[] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    62,  // '+'
    0,  0,  0,
    63,                                      // '/'
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61,  // '0'-'9'
    0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  // 'A'-'Z'
    0,  0,  0,  0,  0,  0,  26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50, 51,  // 'a'-'z'
};
// encoding table
static const char EncodeTable[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
}  // namespace base64

/*!
 * \brief Buffer reader from stream to avoid
 *  virtual call overhead on each read.
 */
class StreamBufferReader {
 public:
  explicit StreamBufferReader(size_t buffer_size) { buffer_.resize(buffer_size); }
  /*!
   * \brief set input stream
   * \param stream The stream to be set
   */
  void set_stream(dmlc::Stream* stream) {
    stream_ = stream;
    read_len_ = read_ptr_ = 1;
  }
  /*!
   * \return allows quick read using get char
   */
  int GetChar() {
    while (true) {
      if (read_ptr_ < read_len_) {
        return static_cast<int>(buffer_[read_ptr_++]);
      } else {
        read_len_ = stream_->Read(&buffer_[0], buffer_.length());
        if (read_len_ == 0) return EOF;
        read_ptr_ = 0;
      }
    }
  }
  /*! \return whether we are reaching the end of file */
  bool AtEnd() const { return read_len_ == 0; }

 private:
  /*! \brief the underlying stream */
  dmlc::Stream* stream_{nullptr};
  /*! \brief buffer to hold data */
  std::string buffer_;
  /*! \brief length of valid data in buffer */
  size_t read_len_{1};
  /*! \brief pointer in the buffer */
  size_t read_ptr_{1};
};

/*!
 * \brief Input stream from base64 encoding
 */
class Base64InStream : public dmlc::Stream {
 public:
  explicit Base64InStream(dmlc::Stream* fs) : reader_(256) { reader_.set_stream(fs); }
  /*!
   * \brief initialize the stream position to beginning of next base64 stream
   * \note call this function before actually start read
   */
  void InitPosition(void) {
    // get a character
    do {
      temp_ch_ = reader_.GetChar();
    } while (isspace(temp_ch_));
  }
  /*! \brief whether current position is end of a base64 stream */
  bool IsEOF(void) const { return num_prev_ == 0 && (temp_ch_ == EOF || isspace(temp_ch_)); }

  using dmlc::Stream::Read;
  // override read function.
  size_t Read(void* ptr, size_t size) final {
    using base64::DecodeTable;
    if (size == 0) return 0;
    // use tlen to record left size
    size_t tlen = size;
    unsigned char* cptr = static_cast<unsigned char*>(ptr);
    // if anything left, load from previous buffered result
    if (num_prev_ != 0) {
      if (num_prev_ == 2) {
        if (tlen >= 2) {
          *cptr++ = buf_prev[0];
          *cptr++ = buf_prev[1];
          tlen -= 2;
          num_prev_ = 0;
        } else {
          // assert tlen == 1
          *cptr++ = buf_prev[0];
          --tlen;
          buf_prev[0] = buf_prev[1];
          num_prev_ = 1;
        }
      } else {
        // assert num_prev_ == 1
        *cptr++ = buf_prev[0];
        --tlen;
        num_prev_ = 0;
      }
    }
    if (tlen == 0) return size;
    int nvalue;
    // note: everything goes with 4 bytes in Base64
    // so we process 4 bytes a unit
    while (tlen && temp_ch_ != EOF && !isspace(temp_ch_)) {
      // first byte
      nvalue = DecodeTable[temp_ch_] << 18;
      {
        // second byte
        temp_ch_ = reader_.GetChar();
        ICHECK(temp_ch_ != EOF && !isspace(temp_ch_)) << "invalid base64 format";
        nvalue |= DecodeTable[temp_ch_] << 12;
        *cptr++ = (nvalue >> 16) & 0xFF;
        --tlen;
      }
      {
        // third byte
        temp_ch_ = reader_.GetChar();
        ICHECK(temp_ch_ != EOF && !isspace(temp_ch_)) << "invalid base64 format";
        // handle termination
        if (temp_ch_ == '=') {
          temp_ch_ = reader_.GetChar();
          ICHECK(temp_ch_ == '=') << "invalid base64 format";
          temp_ch_ = reader_.GetChar();
          ICHECK(temp_ch_ == EOF || isspace(temp_ch_)) << "invalid base64 format";
          break;
        }
        nvalue |= DecodeTable[temp_ch_] << 6;
        if (tlen) {
          *cptr++ = (nvalue >> 8) & 0xFF;
          --tlen;
        } else {
          buf_prev[num_prev_++] = (nvalue >> 8) & 0xFF;
        }
      }
      {
        // fourth byte
        temp_ch_ = reader_.GetChar();
        ICHECK(temp_ch_ != EOF && !isspace(temp_ch_)) << "invalid base64 format";
        if (temp_ch_ == '=') {
          temp_ch_ = reader_.GetChar();
          ICHECK(temp_ch_ == EOF || isspace(temp_ch_)) << "invalid base64 format";
          break;
        }
        nvalue |= DecodeTable[temp_ch_];
        if (tlen) {
          *cptr++ = nvalue & 0xFF;
          --tlen;
        } else {
          buf_prev[num_prev_++] = nvalue & 0xFF;
        }
      }
      // get next char
      temp_ch_ = reader_.GetChar();
    }
    if (kStrictCheck) {
      ICHECK_EQ(tlen, 0) << "Base64InStream: read incomplete";
    }
    return size - tlen;
  }
  virtual void Write(const void* ptr, size_t size) {
    LOG(FATAL) << "Base64InStream do not support write";
  }

 private:
  // internal reader
  StreamBufferReader reader_;
  int temp_ch_{0};
  int num_prev_{0};
  unsigned char buf_prev[2];
  // whether we need to do strict check
  static const bool kStrictCheck = false;
};

/*!
 * \brief Stream to write to base64 format.
 */
class Base64OutStream : public dmlc::Stream {
 public:
  explicit Base64OutStream(dmlc::Stream* fp) : fp_(fp) {}

  using dmlc::Stream::Write;

  void Write(const void* ptr, size_t size) final {
    using base64::EncodeTable;
    size_t tlen = size;
    const unsigned char* cptr = static_cast<const unsigned char*>(ptr);
    while (tlen) {
      while (buf__top_ < 3 && tlen != 0) {
        buf_[++buf__top_] = *cptr++;
        --tlen;
      }
      if (buf__top_ == 3) {
        // flush 4 bytes out
        PutChar(EncodeTable[buf_[1] >> 2]);
        PutChar(EncodeTable[((buf_[1] << 4) | (buf_[2] >> 4)) & 0x3F]);
        PutChar(EncodeTable[((buf_[2] << 2) | (buf_[3] >> 6)) & 0x3F]);
        PutChar(EncodeTable[buf_[3] & 0x3F]);
        buf__top_ = 0;
      }
    }
  }
  virtual size_t Read(void* ptr, size_t size) {
    LOG(FATAL) << "Base64OutStream do not support read";
  }
  /*!
   * \brief finish writing of all current base64 stream, do some post processing
   * \param endch character to put to end of stream, if it is EOF, then nothing will be appended.
   */
  void Finish(int endch = EOF) {
    using base64::EncodeTable;
    if (buf__top_ == 1) {
      PutChar(EncodeTable[buf_[1] >> 2]);
      PutChar(EncodeTable[(buf_[1] << 4) & 0x3F]);
      PutChar('=');
      PutChar('=');
    }
    if (buf__top_ == 2) {
      PutChar(EncodeTable[buf_[1] >> 2]);
      PutChar(EncodeTable[((buf_[1] << 4) | (buf_[2] >> 4)) & 0x3F]);
      PutChar(EncodeTable[(buf_[2] << 2) & 0x3F]);
      PutChar('=');
    }
    buf__top_ = 0;
    if (endch != EOF) PutChar(endch);
    this->Flush();
  }

 private:
  static constexpr size_t kBufferSize = 256;

  dmlc::Stream* fp_{nullptr};
  int buf__top_{0};
  unsigned char buf_[4];
  std::string out_buf_;

  void PutChar(char ch) {
    out_buf_ += ch;
    if (out_buf_.length() >= kBufferSize) Flush();
  }
  void Flush(void) {
    if (out_buf_.length() != 0) {
      fp_->Write(&out_buf_[0], out_buf_.length());
      out_buf_.clear();
    }
  }
};
}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_BASE64_H_
