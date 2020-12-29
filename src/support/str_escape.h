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
 * \file support/str_escape.h
 * \brief Print escape sequence of a string.
 */
#ifndef TVM_SUPPORT_STR_ESCAPE_H_
#define TVM_SUPPORT_STR_ESCAPE_H_

#include <sstream>
#include <string>

namespace tvm {
namespace support {

/*!
 * \brief Create a stream with escape.
 * \param data The data
 * \param size The size of the string.
 * \param use_octal_escape True to use octal escapes instead of hex. If producing C
 *      strings, use octal escapes to avoid ambiguously-long hex escapes.
 * \return the Result string.
 */
inline std::string StrEscape(const char* data, size_t size, bool use_octal_escape = false) {
  std::ostringstream stream;
  for (size_t i = 0; i < size; ++i) {
    unsigned char c = data[i];
    if (c >= ' ' && c <= '~' && c != '\\' && c != '"') {
      stream << c;
    } else {
      stream << '\\';
      switch (c) {
        case '"':
          stream << '"';
          break;
        case '\\':
          stream << '\\';
          break;
        case '\t':
          stream << 't';
          break;
        case '\r':
          stream << 'r';
          break;
        case '\n':
          stream << 'n';
          break;
        default:
          if (use_octal_escape) {
            stream << static_cast<unsigned char>('0' + ((c >> 6) & 0x03))
                   << static_cast<unsigned char>('0' + ((c >> 3) & 0x07))
                   << static_cast<unsigned char>('0' + (c & 0x07));
          } else {
            const char* hex_digits = "0123456789ABCDEF";
            stream << 'x' << hex_digits[c >> 4] << hex_digits[c & 0xf];
          }
      }
    }
  }
  return stream.str();
}

/*!
 * \brief Create a stream with escape.
 * \param data The data
 * \param size The size of the string.
 * \return the Result string.
 */
inline std::string StrEscape(const std::string& val) { return StrEscape(val.data(), val.length()); }

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_STR_ESCAPE_H_
