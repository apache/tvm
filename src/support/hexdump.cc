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
 * \file support/hexdump.cc
 * \brief Print hex representation of BLOBs.
 */

#include "hexdump.h"

namespace tvm {
namespace support {

void HexDump(const std::string& s, std::ostream& os) {
  os << std::hex << std::setfill('0') << std::right;

  int addr_width = 4;
  for (size_t addr_bytes = s.size() >> 16; addr_bytes != 0; addr_bytes >>= 4) {
    addr_width++;
  }

  for (size_t cursor = 0; cursor < s.size(); cursor += 0x10) {
    os << std::setw(addr_width) << cursor;
    size_t row_end = cursor + 0x10;
    if (row_end > s.size()) {
      row_end = s.size();
    }

    os << "  ";
    for (size_t j = cursor; j < row_end; j++) {
      os << " " << std::setw(2) << (unsigned int)(s[j] & 0xff);
    }

    for (size_t j = row_end; j < cursor + 0x10; j++) {
      os << "   ";
    }

    os << std::setw(1) << "  ";
    for (size_t j = cursor; j < row_end; j++) {
      os << (isprint(s[j]) ? s[j] : '.');
    }
    os << std::endl;
  }
}

}  // namespace support
}  // namespace tvm
