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
 * \file rpc_server.h
 * \brief RPC Server implementation.
 */
#ifndef TVM_APPS_CPP_RPC_SERVER_H_
#define TVM_APPS_CPP_RPC_SERVER_H_

#include <string>
#include "tvm/runtime/c_runtime_api.h"

namespace tvm {
namespace runtime {

#if defined(WIN32)
/*!
 * \brief ServerLoopFromChild The Server loop process.
 * \param sock The socket information
 * \param addr The socket address information
 */
void ServerLoopFromChild(SOCKET socket);
#endif

/*!
 * \brief RPCServerCreate Creates the RPC Server.
 * \param host The hostname of the server, Default=0.0.0.0
 * \param port The port of the RPC, Default=9090
 * \param port_end The end search port of the RPC, Default=9199
 * \param tracker The address of RPC tracker in host:port format e.g. 10.77.1.234:9190 Default=""
 * \param key The key used to identify the device type in tracker. Default=""
 * \param custom_addr Custom IP Address to Report to RPC Tracker. Default=""
 * \param silent Whether run in silent mode. Default=True
 */
void RPCServerCreate(std::string host = "",
                     int port = 9090,
                     int port_end = 9099,
                     std::string tracker_addr = "",
                     std::string key = "",
                     std::string custom_addr = "",
                     bool silent = true);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_APPS_CPP_RPC_SERVER_H_
