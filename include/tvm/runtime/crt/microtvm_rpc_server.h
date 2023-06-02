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
 * \file microtvm_rpc_server.h
 * \brief MicroTVM RPC Server
 */

#ifndef TVM_RUNTIME_CRT_MICROTVM_RPC_SERVER_H_
#define TVM_RUNTIME_CRT_MICROTVM_RPC_SERVER_H_

#include <stdlib.h>
#include <sys/types.h>
#include <tvm/runtime/crt/error_codes.h>

#include "../../../../src/support/ssize.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief TVM RPC channel write function.
 *
 * Tries to write `num_bytes` from `data` to the underlying channel.
 * \param context The context.
 * \param data Pointer to data to write.
 * \param num_bytes Number of bytes avaiable in data.
 * \return The number of bytes written.
 */
typedef ssize_t (*microtvm_rpc_channel_write_t)(void* context, const uint8_t* data,
                                                size_t num_bytes);

/*! \brief Opaque pointer type to TVM RPC Server. */
typedef void* microtvm_rpc_server_t;

/*! \brief Initialize the TVM RPC Server.
 *
 * Call this on device startup before calling anyother microtvm_rpc_server_ functions.
 *
 * \param write_func A callback function invoked by the TVM RPC Server to write data back to the
 *                   host. Internally, the TVM RPC Server will block until all data in a reply
 *                   packet has been written.
 * \param write_func_ctx An opaque pointer passed to write_func when it is called.
 * \return A pointer to the TVM RPC Server. The pointer is allocated in the same memory space as
 *         the TVM workspace.
 */
microtvm_rpc_server_t MicroTVMRpcServerInit(microtvm_rpc_channel_write_t write_func,
                                            void* write_func_ctx);

/*! \brief Do any tasks suitable for the main thread, and maybe process new incoming data.
 *
 * \param server The TVM RPC Server pointer.
 * \param new_data If not nullptr, a pointer to a buffer pointer, which should point at new input
 *     data to process. On return, updated to point past data that has been consumed.
 * \param new_data_size_bytes Points to the number of valid bytes in `new_data`. On return,
 *     updated to the number of unprocessed bytes remaining in `new_data` (usually 0).
 * \return An error code indicating the outcome of the server main loop iteration.
 */
tvm_crt_error_t MicroTVMRpcServerLoop(microtvm_rpc_server_t server, uint8_t** new_data,
                                      size_t* new_data_size_bytes);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_CRT_MICROTVM_RPC_SERVER_H_
