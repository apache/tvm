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
 * \file utvm_rpc_server.h
 * \brief MicroTVM RPC Server
 */

#ifndef TVM_RUNTIME_CRT_UTVM_RPC_SERVER_H_
#define TVM_RUNTIME_CRT_UTVM_RPC_SERVER_H_

#include <stdlib.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief TVM RPC channel write function.
 *
 * Tries to write `num_bytes` from `data` to the underlying channel.
 * \param data Pointer to data to write.
 * \param num_bytes Number of bytes avaiable in data.
 * \return The number of bytes written.
 */
typedef ssize_t (*utvm_rpc_channel_write_t)(void* context, const uint8_t* data, size_t num_bytes);

/*! \brief Opaque pointer type to TVM RPC Server. */
typedef void* utvm_rpc_server_t;

/*! \brief Initialize the TVM RPC Server.
 *
 * Call this on device startup before calling anyother utvm_rpc_server_ functions.
 *
 * \param memory A memory block used by the runtime as dynamic memory, primarily to allocate
 *               tensors.
 * \param memory_size_bytes Size of the memory block, in bytes. Should be a multiple of
 *                          (1 << page_size_bytes_log2)
 * \param page_size_bytes_log2 Log2 of the size of each memory page. The internal allocator
 *                             allocates one page at a time; more pages reduces waste but
 *                             increases overhead.
 * \param write_func A callback function invoked by the TVM RPC Server to write data back to the
 *                   host. Internally, the TVM RPC Server will block until all data in a reply
 *                   packet has been written.
 * \param write_func_ctx An opaque pointer passed to write_func when it is called.
 * \return A pointer to the TVM RPC Server. The pointer is allocated in the same memory space as
 *         the TVM workspace.
 */
utvm_rpc_server_t UTvmRpcServerInit(uint8_t* memory, size_t memory_size_bytes,
                                    size_t page_size_bytes_log2,
                                    utvm_rpc_channel_write_t write_func, void* write_func_ctx);

/*! \brief Copy received data into an internal buffer for processing.
 *
 * Currently only handles 1 byte of data. In the future, the goal of this function is to be safe to
 * invoke from an ISR. At that time, this function will just append to an internal buffer.
 *
 * \param server The TVM RPC Server pointer.
 * \param byte The received byte of data.
 * \return The number of bytes copied to the internal buffer. May be less than data_size_bytes when
 * the internal buffer fills.
 */
size_t UTvmRpcServerReceiveByte(utvm_rpc_server_t server, uint8_t byte);

/*! \brief Perform normal processing of received data.
 *
 * \param server The TVM RPC Server pointer.
 * \return true while the server is still running. false when it shuts down gracefully.
 */
bool UTvmRpcServerLoop(utvm_rpc_server_t server);

#ifdef __cplusplus
}
#endif

#endif  // TVM_RUNTIME_CRT_UTVM_RPC_SERVER_H_
