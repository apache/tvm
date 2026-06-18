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
 * \file rpc_tracker.h
 * \brief RPC Tracker implementation.
 */
#ifndef TVM_APPS_CPP_RPC_TRACKER_H_
#define TVM_APPS_CPP_RPC_TRACKER_H_

#include <string>

#include "tvm/runtime/base.h"

namespace tvm {
namespace runtime {

/*!
 * \brief RPCTrackerCreate Creates the RPC Tracker.
 * \param host The listen address of tracker, Default=0.0.0.0
 * \param port The port of the RPC tracker, Default=9190
 * \param port_end The end search port of the RPC tracker, Default=9199
 * \param silent Whether run in silent mode. Default=True
 */
void RPCTrackerCreate(std::string host = "", int port = 9190, int port_end = 9199,
                      bool silent = true);
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_APPS_CPP_RPC_TRACKER_H_
