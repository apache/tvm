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
 * \file support/hexdump.h
 * \brief Print hex representation of BLOBs.
 */
#ifndef TVM_SUPPORT_HEXDUMP_H_
#define TVM_SUPPORT_HEXDUMP_H_

#include <iomanip>
#include <sstream>
#include <string>

namespace tvm {
namespace support {

/*! \brief generate a hexdump of some binary data.
 * \param s Binary data to print.
 * \param os stream that receives the hexdump.
 */
void HexDump(const std::string& s, std::ostream& os);

/*! \brief return a string containing a hexdump of the data in s */
inline std::string HexDump(const std::string& s) {
  std::stringstream ss;
  HexDump(s, ss);
  return ss.str();
}

}  // namespace support
}  // namespace tvm
#endif  // TVM_SUPPORT_HEXDUMP_H_
