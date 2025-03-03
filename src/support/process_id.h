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
 * \file pipe.h
 * \brief Platform independent pipe, used for IPC.
 */
#ifndef TVM_SUPPORT_PROCESS_ID_H_
#define TVM_SUPPORT_PROCESS_ID_H_

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

namespace tvm {
namespace support {

/*! \brief Returns the PID of the current process as an 64-bit signed integer. */
inline int64_t GetProcessId() {
  int64_t result;
#ifdef _WIN32
  DWORD pid = GetCurrentProcessId();
  result = static_cast<int64_t>(pid);
#else
  pid_t pid = getpid();
  result = static_cast<int64_t>(pid);
#endif
  return result;
}

/*! \brief Returns the PID and TIR of the current process/thread as a formatted string */
inline std::string GetProcessIdAndThreadIdHeader() {
  std::ostringstream os;
  os << "[PID " << GetProcessId() << " TID 0x" << std::setw(16) << std::setfill('0') << std::hex
     << std::this_thread::get_id() << "]";
  return os.str();
}

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_PROCESS_ID_H_
