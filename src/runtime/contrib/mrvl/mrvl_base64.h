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
 * \file mrvl_base64.h
 * \brief Util functions for converting plain bytes back to plain bytes
 */

#ifndef TVM_RUNTIME_CONTRIB_MRVL_MRVL_BASE64_H_
#define TVM_RUNTIME_CONTRIB_MRVL_MRVL_BASE64_H_

#include <tvm/runtime/logging.h>

#include <cstddef>
#include <string>

#include "../../../../src/support/base64.h"

namespace tvm {
namespace runtime {
namespace contrib {
namespace mrvl {

inline size_t b64strlen(const std::string& b64str) {
  ICHECK(b64str.size() % 4 == 0) << "invalid base64 encoding";
  size_t length = b64str.size() / 4 * 3;
  if (b64str[b64str.size() - 2] == '=') {
    length -= 2;
  } else if (b64str[b64str.size() - 1] == '=') {
    length -= 1;
  }
  return length;
}

inline void b64decode(const std::string& b64str, uint8_t* ret) {
  size_t index = 0;
  const auto length = b64str.size();
  for (size_t i = 0; i < length; i += 4) {
    int8_t ch0 = tvm::support::base64::DecodeTable[(int32_t)b64str[i]];
    int8_t ch1 = tvm::support::base64::DecodeTable[(int32_t)b64str[i + 1]];
    int8_t ch2 = tvm::support::base64::DecodeTable[(int32_t)b64str[i + 2]];
    int8_t ch3 = tvm::support::base64::DecodeTable[(int32_t)b64str[i + 3]];
    uint8_t st1 = (ch0 << 2) + (ch1 >> 4);
    ret[index++] = st1;
    if (b64str[i + 2] != '=') {
      uint8_t st2 = ((ch1 & 0b1111) << 4) + (ch2 >> 2);
      ret[index++] = st2;
      if (b64str[i + 3] != '=') {
        uint8_t st3 = ((ch2 & 0b11) << 6) + ch3;
        ret[index++] = st3;
      }
    }
  }
  ICHECK(b64strlen(b64str) == index) << "base64 decoding fails";
}

}  // namespace mrvl
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MRVL_MRVL_BASE64_H_
