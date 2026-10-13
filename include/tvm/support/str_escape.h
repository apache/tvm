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
 *
 * \param data The data
 *
 * \param size The size of the string.
 *
 * \param use_octal_escape True to use octal escapes instead of hex. If producing C
 *      strings, use octal escapes to avoid ambiguously-long hex escapes.
 *
 * \param escape_whitespace_special_chars If True (default), escape
 * any tab, newline, and carriage returns that occur in the string.
 * If False, do not escape these characters.
 *
 * \return the Result string.
 */
inline std::string StrEscape(const char* data, size_t size, bool use_octal_escape = false,
                             bool escape_whitespace_special_chars = true) {
  std::ostringstream stream;
  for (size_t i = 0; i < size; ++i) {
    unsigned char c = data[i];
    if (c >= ' ' && c <= '~' && c != '\\' && c != '"') {
      stream << c;
    } else {
      switch (c) {
        case '"':
          stream << '\\' << '"';
          break;
        case '\\':
          stream << '\\' << '\\';
          break;
        case '\t':
          if (escape_whitespace_special_chars) {
            stream << '\\' << 't';
          } else {
            stream << c;
          }
          break;
        case '\r':
          if (escape_whitespace_special_chars) {
            stream << '\\' << 'r';
          } else {
            stream << c;
          }
          break;
        case '\n':
          if (escape_whitespace_special_chars) {
            stream << '\\' << 'n';
          } else {
            stream << c;
          }
          break;
        default:
          if (use_octal_escape) {
            stream << '\\' << static_cast<unsigned char>('0' + ((c >> 6) & 0x03))
                   << static_cast<unsigned char>('0' + ((c >> 3) & 0x07))
                   << static_cast<unsigned char>('0' + (c & 0x07));
          } else {
            const char* hex_digits = "0123456789ABCDEF";
            stream << '\\' << 'x' << hex_digits[c >> 4] << hex_digits[c & 0xf];
          }
      }
    }
  }
  return stream.str();
}

/*!
 * \brief Create a stream with escape.
 *
 * \param val The C++ string
 *
 * \param use_octal_escape True to use octal escapes instead of hex. If producing C
 *      strings, use octal escapes to avoid ambiguously-long hex escapes.
 *
 * \param escape_whitespace_special_chars If True (default), escape
 * any tab, newline, and carriage returns that occur in the string.
 * If False, do not escape these characters.  If producing python
 * strings with """triple quotes""", do not escape these characters.
 *
 * \return the Result string.
 */
inline std::string StrEscape(const std::string& val, bool use_octal_escape = false,
                             bool escape_whitespace_special_chars = true) {
  return StrEscape(val.data(), val.length(), use_octal_escape, escape_whitespace_special_chars);
}

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_STR_ESCAPE_H_
