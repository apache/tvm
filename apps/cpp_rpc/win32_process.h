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
 * \file win32_process.h
 * \brief Win32 process code to mimic a POSIX fork()
 */
#ifndef TVM_APPS_CPP_RPC_WIN32_PROCESS_H_
#define TVM_APPS_CPP_RPC_WIN32_PROCESS_H_

#include <chrono>
#include <string>

#include "../../src/support/socket.h"

namespace tvm {
namespace runtime {
/*!
 * \brief SpawnRPCChild Spawns a child process with a given timeout to run
 * \param fd The client socket to duplicate in the child
 * \param timeout The time in seconds to wait for the child to complete before termination
 */
void SpawnRPCChild(SOCKET fd, std::chrono::seconds timeout);
/*!
 * \brief ChildProcSocketHandler Ran from the child process and runs server to handle the client
 * socket \param mmap_path The memory mapped file path that will contain the information to
 * duplicate the client socket from the parent
 */
void ChildProcSocketHandler(const std::string& mmap_path);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_CPP_RPC_WIN32_PROCESS_H_
