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
 * \file rpc_session.h
 * \brief Base RPC session interface.
 */
#ifndef TVM_RUNTIME_RPC_RPC_SESSION_H_
#define TVM_RUNTIME_RPC_RPC_SESSION_H_

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>

#include <functional>
#include <memory>
#include <string>

#include "../minrpc/rpc_reference.h"

namespace tvm {
namespace runtime {

/*!
 * \brief The interface of all remote RPC sessions.
 *
 *  It contains all the necessary interface to implement
 *  remote call and resource management.
 *
 *  The interface is designed to allow easy proxy-chaining
 *  by forward requests to another RPCSession.
 */
class RPCSession {
 public:
  /*! \brief PackedFunc Handle in the remote. */
  using PackedFuncHandle = void*;

  /*! \brief Module handle in the remote. */
  using ModuleHandle = void*;

  /*! \brief NDArray handle in the remote. */
  using NDArrayHandle = void*;

  /*!
   * \brief Callback to send an encoded return values via encode_args.
   *
   * \param encode_args The arguments that we can encode the return values into.
   *
   * Encoding convention (as list of arguments):
   * - str/float/int/byte: [tcode: int, value: TVMValue] value follows PackedFunc convention.
   * - PackedFunc/Module: [tcode: int, handle: void*]
   * - NDArray: [tcode: int,  meta: DLTensor*, nd_handle: void*]
   *            DLTensor* contains the meta-data as well as handle into the remote data.
   *            nd_handle can be used for deletion.
   */
  using FEncodeReturn = std::function<void(TVMArgs encoded_args)>;

  /*!
   * \brief Callback to send an encoded return values via encode_args.
   *
   * \param status The return status, can be RPCCode::kReturn or RPCCode::kException.
   * \param encode_args The arguments that we can encode the return values into.
   */
  using FAsyncCallback = std::function<void(RPCCode status, TVMArgs encoded_args)>;

  /*! \brief Destructor.*/
  virtual ~RPCSession() {}

  /*!
   * \brief Get function in the session.
   * \param name The name of the function.
   * \return The function handle.
   */
  virtual PackedFuncHandle GetFunction(const std::string& name) = 0;

  /*!
   * \brief Call into a remote Packed function.
   *
   *  Calling convention:
   *
   *  - type_code is follows the PackedFunc convention.
   *  - int/float/string/bytes follows the PackedFunc convention, all data are local.
   *  - PackedFunc/Module and future remote objects: pass remote handle instead.
   *  - NDArray/DLTensor: pass a DLTensor pointer, the data field of DLTensor
   *                      points to a remote data handle returned by the Device API.
   *                      The meta-data of the DLTensor sits on local.
   *
   *  The caller populates the arguments and manages these arguments.
   *
   *  The callee can change the content of arg_values and arg_type_codes
   *  if they want to do inplace modify and forward.
   *
   *  The callee need to store the return value into ret_value.
   *  - PackedFunc/Module are stored as void*
   *  - NDArray is stored as local NDArray, whose data field is a remote handle.
   *    Notably the NDArray's deleter won't delete remote handle.
   *    It is up to the user of the RPCSession to such wrapping.
   *  - In short, remote handles are "moved" as return values
   *    and the callee needs to explicitly manage them by calling
   *    the deleter functions when they are no longer needed.
   *
   * \param func The function handle.
   * \param arg_values The argument values.
   * \param arg_type_codes the type codes of the argument.
   * \param num_args Number of arguments.
   * \param fencode_return The function to set the return value,
   *                       if not called, return value is null.
   */
  virtual void CallFunc(PackedFuncHandle func, const TVMValue* arg_values,
                        const int* arg_type_codes, int num_args,
                        const FEncodeReturn& fencode_return) = 0;

  /*!
   * \brief Copy bytes into remote array content.
   * \param local_from_bytes The source host data.
   * \param remote_to The target array.
   * \param nbytes The size of the memory in bytes.
   */
  virtual void CopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes) = 0;
  /*!
   * \brief Copy bytes from remote array content.
   * \param remote_from The source host data.
   * \param local_to_bytes The target array.
   * \param nbytes The size of the memory in bytes.
   */
  virtual void CopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes) = 0;

  /*!
   * \brief Free a remote function.
   * \param handle The remote handle, can be NDArray/PackedFunc/Module
   * \param type_code The type code of the underlying type.
   */
  virtual void FreeHandle(void* handle, int type_code) = 0;

  /*!
   * \brief Get device API that represents the remote
   *  actions that can be taken on the remote.
   *
   *  The caller can then call into the Alloc/Free functions
   *  to allocate free spaces and taking the pointer as the handle.
   *
   *  The device API is guaranteed to be alive during the
   *  lifetime of the Session.
   *
   * \param dev The remote device.
   * \param allow_missing Whether can we return nullptr if it is not available.
   *
   * \return The device API.
   */
  virtual DeviceAPI* GetDeviceAPI(Device dev, bool allow_missing = false) = 0;

  /*!
   * \brief Whether the session is a local session and we can directly
   *        the data handle returned by the session and treat it as pointer
   *        to the local memory.
   *
   * This information is useful for RPC server to directly copy into the
   * local memory without creating a temporary buffer.
   *
   * \return Whether it is a local session.
   */
  virtual bool IsLocalSession() const = 0;

  // Asynchrous variant of API
  // These APIs are used by the RPC server to allow sessions that
  // have special implementations for the async functions.
  //
  // In the async APIs, an exception is returned by the passing
  // async_error=true, encode_args=[error_msg].

  /*!
   * \brief Whether the session is async.
   *
   * If the session is not async, its Aync implementations
   * simply calls into the their synchronize counterparts,
   * and the callback is guaranteed to be called before the async function finishes.
   *
   * \return the async state.
   *
   * \note We can only use async session in an Event driven RPC server.
   */
  virtual bool IsAsync() const;

  /*!
   * \brief Asynchrously call func.
   * \param func The function handle.
   * \param arg_values The argument values.
   * \param arg_type_codes the type codes of the argument.
   * \param num_args Number of arguments.
   *
   * \param callback The callback to pass the return value or exception.
   */
  virtual void AsyncCallFunc(PackedFuncHandle func, const TVMValue* arg_values,
                             const int* arg_type_codes, int num_args, FAsyncCallback callback);

  /*!
   * \brief Asynchrous version of CopyToRemote.
   *
   * \param local_from_bytes The source host data.
   * \param remote_to The target array.
   * \param nbytes The size of the memory in bytes.
   * \param on_complete The callback to signal copy complete.
   * \note All the allocated memory in local_from, and remote_to
   *       must stay alive until on_compelete is called.
   */
  virtual void AsyncCopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes,
                                 FAsyncCallback on_complete);

  /*!
   * \brief Asynchrous version of CopyFromRemote.
   *
   * \param remote_from The source host data.
   * \param local_to_bytes The target array.
   * \param nbytes The size of the memory in bytes.
   * \param on_complete The callback to signal copy complete.
   * \note All the allocated memory in remote_from, and local_to
   *       must stay alive until on_compelete is called.
   */
  virtual void AsyncCopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes,
                                   FAsyncCallback on_complete);
  /*!
   * \brief Asynchrously wait for all events in dev, stream compeletes.
   * \param dev The device.
   * \param stream The stream to wait on.
   * \param on_complete The callback to signal copy complete.
   */
  virtual void AsyncStreamWait(Device dev, TVMStreamHandle stream, FAsyncCallback on_compelte);

  /*!
   * \return The session table index of the session.
   */
  int table_index() const { return table_index_; }

  /*!
   * \brief Try get session from the global session table by table index.
   * \param table_index The table index of the session.
   * \return The shared_ptr to the session, can be nullptr.
   */
  static std::shared_ptr<RPCSession> Get(int table_index);

  /*!
   * \brief Shutdown RPC connection.
   */
  virtual void Shutdown() {}

 protected:
  /*!
   * \brief Send an exception to the callback.
   * \param msg The exception message.
   */
  void SendException(FAsyncCallback callback, const char* msg);

 private:
  /*! \brief index of this session in RPC session table */
  int table_index_{0};
  /*! \brief Insert the current session to the session table.*/
  static void InsertToSessionTable(std::shared_ptr<RPCSession> sess);
  // friend declaration
  friend Module CreateRPCSessionModule(std::shared_ptr<RPCSession> sess);
};

/*!
 * \brief Remote space handle cell used by the RPC runtime API.
 *
 *  When we allocate space using a rpc device, the data pointer
 *  points to an allocated RemoteSpace.
 */
struct RemoteSpace {
  /*! \brief The remote data handle. */
  void* data;
  /*! \brief Reference to the underlying RPC session. */
  std::shared_ptr<RPCSession> sess;
};

/*!
 * \brief Create a Global RPC module that refers to the session.
 * \param sess The RPC session of the global module.
 * \return The created module.
 */
Module CreateRPCSessionModule(std::shared_ptr<RPCSession> sess);

/*!
 * \brief Get the session module from a RPC session Module.
 * \param mod The input module(must be an RPCModule).
 * \return The internal RPCSession.
 */
std::shared_ptr<RPCSession> RPCModuleGetSession(Module mod);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_SESSION_H_
