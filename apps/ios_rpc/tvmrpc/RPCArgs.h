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

#ifndef TVM_APPS_IOS_RPC_ARGS_H_
#define TVM_APPS_IOS_RPC_ARGS_H_

#import "RPCServer.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Struct representing arguments of iOS RPC app
 */
typedef struct RPCArgs_t {
  /// Tracker or Proxy address (actually ip)
  const char* host_url;

  /// Tracker or Proxy port
  int host_port;

  /// device key to report
  const char* key;

  /// custom adress to report into Tracker. Ignored for other server modes.
  const char* custom_addr;

  /// Verbose mode. Will print status messages to std out.
  /// 0 - no prints , 1 - print state to output
  bool verbose;

  /// Immediate server launch. No UI interaction.
  /// 0 - UI interaction, 1 - automatically connect on launch
  bool immediate_connect;

  /// Server mode
  RPCServerMode server_mode;
} RPCArgs;

/*!
 * \brief Get current global RPC args
 */
RPCArgs get_current_rpc_args(void);

/*!
 * \brief Set current global RPC args and update values in app cache
 */
void set_current_rpc_args(RPCArgs args);

/*!
 * \brief Pars command line args and update current global RPC args
 * Also updates values in app cache
 */
void update_rpc_args(int argc, char* argv[]);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_APPS_IOS_RPC_ARGS_H_
