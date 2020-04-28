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
 * \file rpc_channel.h
 * \brief Communication endpoints to connect local and remote RPC sessions.
 */
#ifndef TVM_RUNTIME_RPC_RPC_CHANNEL_H_
#define TVM_RUNTIME_RPC_RPC_CHANNEL_H_

#include <tvm/runtime/packed_func.h>
#include <utility>

namespace tvm {
namespace runtime {

/*!
 * \brief Abstract channel interface used to create RPCEndpoint.
 */
class RPCChannel {
 public:
  /*! \brief virtual destructor */
  virtual ~RPCChannel() {}
  /*!
   * \brief Send data over to the channel.
   * \param data The data pointer.
   * \param size The size fo the data.
   * \return The actual bytes sent.
   */
  virtual size_t Send(const void* data, size_t size) = 0;
  /*!
   * \brief Recv data from channel.
   *
   * \param data The data pointer.
   * \param size The size fo the data.
   * \return The actual bytes received.
   */
  virtual size_t Recv(void* data, size_t size) = 0;
};

/*!
 * \brief RPC channel which callback
 * frontend (Python/Java/etc.)'s send & recv function
 */
class CallbackChannel final : public RPCChannel {
 public:
  /*!
   * \brief Constructor.
   *
   * \param fsend The send function, takes in a TVMByteArray and returns the
   *              number of bytes sent in that array. Returns -1 if error happens.
   * \param frecv The recv function, takes an expected maximum size, and return
   *              a byte array with the actual amount of data received.
   */
  explicit CallbackChannel(PackedFunc fsend, PackedFunc frecv)
      : fsend_(std::move(fsend)), frecv_(std::move(frecv)) {}

  ~CallbackChannel() {}
  /*!
   * \brief Send data over to the channel.
   * \param data The data pointer.
   * \param size The size fo the data.
   * \return The actual bytes sent.
   */
  size_t Send(const void* data, size_t size) final;
  /*!
   * \brief Recv data from channel.
   *
   * \param data The data pointer.
   * \param size The size fo the data.
   * \return The actual bytes received.
   */
  size_t Recv(void* data, size_t size) final;

 private:
  PackedFunc fsend_;
  PackedFunc frecv_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_CHANNEL_H_
