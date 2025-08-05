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
 * \file tvm/ffi/extra/base64.h
 * \brief Base64 encoding and decoding utilities
 */
#ifndef TVM_FFI_EXTRA_BASE64_H_
#define TVM_FFI_EXTRA_BASE64_H_

#include <tvm/ffi/string.h>

#include <string>

namespace tvm {
namespace ffi {
/*!
 * \brief Encode a byte array into a base64 string
 * \param bytes The byte array to encode
 * \return The base64 encoded string
 */
inline String Base64Encode(TVMFFIByteArray bytes) {
  // encoding every 3 bytes into 4 characters
  constexpr const char kEncodeTable[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string encoded;
  encoded.reserve(4 * (bytes.size + 2) / 3);

  for (size_t i = 0; i < (bytes.size / 3) * 3; i += 3) {
    int32_t buf[3];
    buf[0] = static_cast<int32_t>(bytes.data[i]);
    buf[1] = static_cast<int32_t>(bytes.data[i + 1]);
    buf[2] = static_cast<int32_t>(bytes.data[i + 2]);
    encoded.push_back(kEncodeTable[buf[0] >> 2]);
    encoded.push_back(kEncodeTable[((buf[0] << 4) | (buf[1] >> 4)) & 0x3F]);
    encoded.push_back(kEncodeTable[((buf[1] << 2) | (buf[2] >> 6)) & 0x3F]);
    encoded.push_back(kEncodeTable[buf[2] & 0x3F]);
  }
  if (bytes.size % 3 == 1) {
    int32_t buf[1] = {static_cast<int32_t>(bytes.data[bytes.size - 1])};
    encoded.push_back(kEncodeTable[buf[0] >> 2]);
    encoded.push_back(kEncodeTable[(buf[0] << 4) & 0x3F]);
    encoded.push_back('=');
    encoded.push_back('=');
  } else if (bytes.size % 3 == 2) {
    int32_t buf[2] = {static_cast<int32_t>(bytes.data[bytes.size - 2]),
                      static_cast<int32_t>(bytes.data[bytes.size - 1])};
    encoded.push_back(kEncodeTable[buf[0] >> 2]);
    encoded.push_back(kEncodeTable[((buf[0] << 4) | (buf[1] >> 4)) & 0x3F]);
    encoded.push_back(kEncodeTable[(buf[1] << 2) & 0x3F]);
    encoded.push_back('=');
  }
  return String(encoded);
}

/*!
 * \brief Encode a bytes object into a base64 string
 * \param data The bytes object to encode
 * \return The base64 encoded string
 */
inline String Base64Encode(const Bytes& data) {
  return Base64Encode(TVMFFIByteArray{data.data(), data.size()});
}

/*!
 * \brief Decode a base64 string into a byte array
 * \param data The base64 encoded string to decode
 * \return The decoded byte array
 */
inline Bytes Base64Decode(TVMFFIByteArray bytes) {
  constexpr const char kDecodeTable[] = {
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
  std::string decoded;
  decoded.reserve(bytes.size * 3 / 4);
  if (bytes.size == 0) return Bytes();
  TVM_FFI_ICHECK(bytes.size % 4 == 0) << "invalid base64 encoding";
  // leverage this property to simplify decoding
  static_assert('=' < sizeof(kDecodeTable) && kDecodeTable[static_cast<size_t>('=')] == 0);
  // base64 is always multiple of 4 bytes
  for (size_t i = 0; i < bytes.size; i += 4) {
    // decode every 4 characters into 24bits, each character contains 6 bits
    // note that = is also decoded as 0, which is safe to skip
    int32_t buf[4] = {
        static_cast<int32_t>(bytes.data[i]),
        static_cast<int32_t>(bytes.data[i + 1]),
        static_cast<int32_t>(bytes.data[i + 2]),
        static_cast<int32_t>(bytes.data[i + 3]),
    };
    int32_t value_i24 = (static_cast<int32_t>(kDecodeTable[buf[0]]) << 18) |
                        (static_cast<int32_t>(kDecodeTable[buf[1]]) << 12) |
                        (static_cast<int32_t>(kDecodeTable[buf[2]]) << 6) |
                        static_cast<int32_t>(kDecodeTable[buf[3]]);
    // unpack 24bits into 3 bytes, each contains 8 bits
    decoded.push_back(static_cast<char>((value_i24 >> 16) & 0xFF));
    if (buf[2] != '=') {
      decoded.push_back(static_cast<char>((value_i24 >> 8) & 0xFF));
    }
    if (buf[3] != '=') {
      decoded.push_back(static_cast<char>(value_i24 & 0xFF));
    }
  }
  return Bytes(decoded);
}

/*!
 * \brief Decode a base64 string into a byte array
 * \param data The base64 encoded string to decode
 * \return The decoded byte array
 */
inline Bytes Base64Decode(const String& data) {
  return Base64Decode(TVMFFIByteArray{data.data(), data.size()});
}

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_EXTRA_BASE64_H_
