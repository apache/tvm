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
 *  Copyright (c) 2017 by Contributors
 * \file rpc_session.h
 * \brief Base RPC session interface.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SESSION_H_
#define TVM_RUNTIME_RPC_RPC_SESSION_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <mutex>
#include <string>
#include <memory>
#include <utility>
#include "../../common/ring_buffer.h"

namespace tvm {
namespace runtime {

const int kRPCMagic = 0xff271;

/*! \brief The remote functio handle */
using RPCFuncHandle = void*;

struct RPCArgBuffer;

/*! \brief The RPC code */
enum class RPCCode : int {
  kNone,
  kCallFunc,
  kReturn,
  kException,
  kShutdown,
  kCopyFromRemote,
  kCopyToRemote,
  kCopyAck,
  // The following are code that can send over CallRemote
  kSystemFuncStart,
  kGetGlobalFunc,
  kGetTimeEvaluator,
  kFreeFunc,
  kDevSetDevice,
  kDevGetAttr,
  kDevAllocData,
  kDevFreeData,
  kDevStreamSync,
  kCopyAmongRemote,
  kModuleLoad,
  kModuleImport,
  kModuleFree,
  kModuleGetFunc,
  kModuleGetSource,
  kNDArrayFree
};

/*!
 * \brief Abstract channel interface used to create RPCSession.
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
e   * \brief Recv data from channel.
   *
   * \param data The data pointer.
   * \param size The size fo the data.
   * \return The actual bytes received.
   */
  virtual size_t Recv(void* data, size_t size) = 0;
};

// Bidirectional Communication Session of PackedRPC
class RPCSession {
 public:
  /*! \brief virtual destructor */
  ~RPCSession();
  /*!
   *  \brief The server loop that server runs to handle RPC calls.
   */
  void ServerLoop();
  /*!
   * \brief Message handling function for event driven server.
   *  Called when the server receives a message.
   *  Event driven handler will never call recv on the channel
   *  and always relies on the ServerEventHandler.
   *  to receive the data.
   *
   * \param in_bytes The incoming bytes.
   * \param event_flag  1: read_available, 2: write_avaiable.
   * \return State flag.
   *     1: continue running, no need to write,
   *     2: need to write
   *     0: shutdown
   */
  int ServerEventHandler(const std::string& in_bytes,
                         int event_flag);
  /*!
   * \brief Call into remote function
   * \param handle The function handle
   * \param args The arguments
   * \param rv The return value.
   * \param fwrap Wrapper function to turn Function/Module handle into real return.
   */
  void CallFunc(RPCFuncHandle handle,
                TVMArgs args,
                TVMRetValue* rv,
                const PackedFunc* fwrap);
  /*!
   * \brief Copy bytes into remote array content.
   * \param from The source host data.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param nbytes The size of the memory in bytes.
   * \param ctx_to The target context.
   * \param type_hint Hint of content data type.
   */
  void CopyToRemote(void* from,
                    size_t from_offset,
                    void* to,
                    size_t to_offset,
                    size_t nbytes,
                    TVMContext ctx_to,
                    TVMType type_hint);
  /*!
   * \brief Copy bytes from remote array content.
   * \param from The source host data.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param nbytes The size of the memory in bytes.
   * \param ctx_from The source context.
   * \param type_hint Hint of content data type.
   */
  void CopyFromRemote(void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t nbytes,
                      TVMContext ctx_from,
                      TVMType type_hint);
  /*!
   * \brief Get a remote timer function on ctx.
   *  This function consumes fhandle, caller should not call Free on fhandle.
   *
   * \param fhandle The function handle.
   * \param ctx The ctx to run measurement on.
   * \param number The number of times to run this function for taking average.
          We call these runs as one `repeat` of measurement.
   * \param repeat The number of times to repeat the measurement.
          In total, the function will be invoked (1 + number x repeat) times,
          where the first one is warm up and will be discarded.
          The returned result contains `repeat` costs,
          each of which is an average of `number` costs.
   * \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
          By default, one `repeat` contains `number` runs. If this parameter is set,
          the parameters `number` will be dynamically adjusted to meet the
          minimum duration requirement of one `repeat`.
          i.e., When the run time of one `repeat` falls below this time,
          the `number` parameter will be automatically increased.
   * \return A remote timer function
   */
  RPCFuncHandle GetTimeEvaluator(RPCFuncHandle fhandle,
                                 TVMContext ctx,
                                 int number,
                                 int repeat,
                                 int min_repeat_ms);
  /*!
   * \brief Call a remote defined system function with arguments.
   * \param fcode The function code.
   * \param args The arguments
   * \return The returned remote value.
   */
  template<typename... Args>
  inline TVMRetValue CallRemote(RPCCode fcode, Args&& ...args);
  /*!
   * \return The session table index of the session.
   */
  int table_index() const {
    return table_index_;
  }
  /*!
   * \brief Create a RPC session with given channel.
   * \param channel The communication channel.
   * \param name The local name of the session, used for debug
   * \param remote_key The remote key of the session
   *   if remote_key equals "%toinit", we need to re-intialize
   *   it by event handler.
   */
  static std::shared_ptr<RPCSession> Create(
      std::unique_ptr<RPCChannel> channel,
      std::string name,
      std::string remote_key);
  /*!
   * \brief Try get session from the global session table by table index.
   * \param table_index The table index of the session.
   * \return The shared_ptr to the session, can be nullptr.
   */
  static std::shared_ptr<RPCSession> Get(int table_index);

 private:
  class EventHandler;
  // Handle events until receives a return
  // Also flushes channels so that the function advances.
  RPCCode HandleUntilReturnEvent(
      TVMRetValue* rv, bool client_mode, const PackedFunc* fwrap);
  // Initalization
  void Init();
  // Shutdown
  void Shutdown();
  // Internal channel.
  std::unique_ptr<RPCChannel> channel_;
  // Internal mutex
  std::recursive_mutex mutex_;
  // Internal ring buffer.
  common::RingBuffer reader_, writer_;
  // Event handler.
  std::shared_ptr<EventHandler> handler_;
  // call remote with specified function code.
  PackedFunc call_remote_;
  // The index of this session in RPC session table.
  int table_index_{0};
  // The name of the session.
  std::string name_;
  // The remote key
  std::string remote_key_;
};

/*!
 * \brief Wrap a timer function to measure the time cost of a given packed function.
 * \param f The function argument.
 * \param ctx The context.
 * \param number The number of times to run this function for taking average.
          We call these runs as one `repeat` of measurement.
 * \param repeat The number of times to repeat the measurement.
          In total, the function will be invoked (1 + number x repeat) times,
          where the first one is warm up and will be discarded.
          The returned result contains `repeat` costs,
          each of which is an average of `number` costs.
 * \param min_repeat_ms The minimum duration of one `repeat` in milliseconds.
          By default, one `repeat` contains `number` runs. If this parameter is set,
          the parameters `number` will be dynamically adjusted to meet the
          minimum duration requirement of one `repeat`.
          i.e., When the run time of one `repeat` falls below this time,
          the `number` parameter will be automatically increased.
 * \return f_timer A timer function.
 */
PackedFunc WrapTimeEvaluator(PackedFunc f,
                             TVMContext ctx,
                             int number,
                             int repeat,
                             int min_repeat_ms);

/*!
 * \brief Create a Global RPC module that refers to the session.
 * \param sess The RPC session of the global module.
 * \return The created module.
 */
Module CreateRPCModule(std::shared_ptr<RPCSession> sess);

// Remote space pointer.
struct RemoteSpace {
  void* data;
  std::shared_ptr<RPCSession> sess;
};

// implementation of inline functions
template<typename... Args>
inline TVMRetValue RPCSession::CallRemote(RPCCode code, Args&& ...args) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  writer_.Write(&code, sizeof(code));
  return call_remote_(std::forward<Args>(args)...);
}
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SESSION_H_
