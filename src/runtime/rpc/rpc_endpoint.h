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
 * \file rpc_endpoint.h
 * \brief Communication endpoints to connect local and remote RPC sessions.
 */
#ifndef TVM_RUNTIME_RPC_RPC_ENDPOINT_H_
#define TVM_RUNTIME_RPC_RPC_ENDPOINT_H_

#include <tvm/runtime/packed_func.h>

#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "../../support/ring_buffer.h"
#include "../minrpc/rpc_reference.h"
#include "rpc_channel.h"
#include "rpc_channel_logger.h"
#include "rpc_session.h"

namespace tvm {
namespace runtime {

// Magic header for RPC data plane
const int kRPCMagic = 0xff271;
// magic header for RPC tracker(control plane)
const int kRPCTrackerMagic = 0x2f271;
// sucess response
const int kRPCSuccess = kRPCMagic + 0;
// cannot found matched key in server
const int kRPCMismatch = kRPCMagic + 2;

/*! \brief Enumeration code for the RPC tracker */
enum class TrackerCode : int {
  kFail = -1,
  kSuccess = 0,
  kPing = 1,
  kStop = 2,
  kPut = 3,
  kRequest = 4,
  kUpdateInfo = 5,
  kSummary = 6,
  kGetPendingMatchKeys = 7
};

/*!
 * \brief Communication endpoints to connect local and remote RPC sessions.
 *        An endpoint can either be a client or a server.
 */
class RPCEndpoint {
 public:
  /*! \brief virtual destructor
   * Closes the connection if the connection hasn't already been closed.
   */
  ~RPCEndpoint();

  /*!
   *  \brief Shutdown RPC connection.
   *
   *  Shutdown has no effect if the connection has already been shut down.
   *  Shutdown will wait for all output currently queued from the RPC connection (i.e. The user
   * doesn't need to wait for completion before calling Shutdown.) Any further use of objects that
   * depended on the endpoint (e.g. A tvm.nd.array allocated on the remote RPC session) may throw an
   * exception when used.
   */
  void Shutdown();

  /*!
   *  \brief The server loop that server runs to handle RPC calls.
   */
  void ServerLoop();
  /*!
   * \brief Message handling function for an async IO event driven server.
   *
   *  Called when the server receives a message or an IO event update.
   *  Event driven handler will never call recv on the channel
   *  and always relies on the ServerIOEventHandler to receive the data.
   *
   * \param in_bytes The incoming bytes.
   * \param event_flag  1: read_available, 2: write_avaiable.
   * \return State flag.
   *     1: continue running, no need to write,
   *     2: need to write
   *     0: shutdown
   */
  int ServerAsyncIOEventHandler(const std::string& in_bytes, int event_flag);

  /*!
   * \brief Initalize the session on the remote that will be used to back all the RPC requests.
   *
   *  If no session constructor arguments is passed, LocalSession will be used in the remote.
   *  Otherwise the remote serving session will be constructed using the arguments
   *  specified in the session_constructor_args.
   *
   *  The construction rule can be summarized as follows:
   *
   * \code
   *
   *  auto args = session_constructor_args;
   *  int n = args.size();
   *  if (n != 0) {
   *    std::string constructor = args[0];
   *    server.serving_session_ = GetGlobalFunc(constructor)(
   *        args[1], args[2] ... args[n - 1])
   *  } else {
   *    server.serving_session_ = LocalSession();
   *  }
   * \endcode
   *
   * \param session_constructor_args Optional sequence of the remote sesssion constructor.
   */
  void InitRemoteSession(TVMArgs session_constructor_args);

  /*!
   * \brief Call into remote function
   * \param handle The function handle
   * \param arg_values The argument values.
   * \param arg_type_codes the type codes of the argument.
   * \param num_args Number of arguments.
   * \param fencode_return The function to receive return value encodings.
   */
  void CallFunc(RPCSession::PackedFuncHandle handle, const TVMValue* arg_values,
                const int* arg_type_codes, int num_args, RPCSession::FEncodeReturn encode_return);
  /*!
   * \brief Copy bytes into remote array content.
   * \param from The source host data.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param nbytes The size of the memory in bytes.
   * \param dev_to The target device.
   * \param type_hint Hint of content data type.
   */
  void CopyToRemote(void* from_bytes, DLTensor* to, uint64_t nbytes);
  /*!
   * \brief Copy bytes from remote array content.
   * \param from The source host data.
   * \param from_offset The byte offeset in the from.
   * \param to The target array.
   * \param to_offset The byte offset in the to.
   * \param nbytes The size of the memory in bytes.
   * \param dev_from The source device.
   * \param type_hint Hint of content data type.
   */
  void CopyFromRemote(DLTensor* from, void* to_bytes, uint64_t nbytes);

  /*!
   * \brief Call a remote defined system function with arguments.
   * \param fcode The function code.
   * \param args The arguments
   * \return The returned remote value.
   */
  template <typename... Args>
  inline TVMRetValue SysCallRemote(RPCCode fcode, Args&&... args);
  /*!
   * \brief Create a RPC session with given channel.
   * \param channel The communication channel.
   * \param name The local name of the session, used for debug
   * \param remote_key The remote key of the session
   *   if remote_key equals "%toinit", we need to re-intialize
   *   it by event handler.
   * \param fcleanup The cleanup Packed function.
   */
  static std::shared_ptr<RPCEndpoint> Create(std::unique_ptr<RPCChannel> channel, std::string name,
                                             std::string remote_key,
                                             TypedPackedFunc<void()> fcleanup = nullptr);

 private:
  class EventHandler;
  // Handle events until receives a return
  // Also flushes channels so that the function advances.
  RPCCode HandleUntilReturnEvent(bool client_mode, RPCSession::FEncodeReturn setreturn);
  // Initalization
  void Init();
  // Internal channel.
  std::unique_ptr<RPCChannel> channel_;

  // Internal mutex
  std::mutex mutex_;
  // Internal ring buffer.
  support::RingBuffer reader_, writer_;
  // Event handler.
  std::shared_ptr<EventHandler> handler_;
  // syscall remote with specified function code.
  PackedFunc syscall_remote_;
  // The name of the session.
  std::string name_;
  // The remote key
  std::string remote_key_;
  // Invoked when the RPC session is terminated
  TypedPackedFunc<void()> fcleanup_;
};

/*!
 * \brief Create an RPC client session from an RPC client endpoint.
 * \param endpoint The endpoint.
 * \return The created session.
 */
std::shared_ptr<RPCSession> CreateClientSession(std::shared_ptr<RPCEndpoint> endpoint);

// implementation of inline functions
template <typename... Args>
inline TVMRetValue RPCEndpoint::SysCallRemote(RPCCode code, Args&&... args) {
  return syscall_remote_(static_cast<int>(code), std::forward<Args>(args)...);
}

/*!
 * \brief Calculates overhead size of a CopyToRemote packet.
 * \param to DLTensor to copy.
 * \param code RPCCode for this transfer.
 * \param nbytes Number of bytes to transfer.
 * \return The remote-copy packet overhead size.
 */
uint64_t RemoteCopyCalculatePacketOverheadSize(DLTensor* tensor, RPCCode code, uint64_t nbytes);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_ENDPOINT_H_
