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

#ifndef TVM_RUNTIME_MINRPC_MINRPC_INTERFACES_H_
#define TVM_RUNTIME_MINRPC_MINRPC_INTERFACES_H_

#include <tvm/runtime/c_runtime_api.h>

#include "rpc_reference.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Return interface used in ExecInterface to generate and send the responses.
 */
class MinRPCReturnInterface {
 public:
  virtual ~MinRPCReturnInterface() {}
  /*! * \brief sends a response to the client with kTVMNullptr in payload. */
  virtual void ReturnVoid() = 0;

  /*! * \brief sends a response to the client with one kTVMOpaqueHandle in payload. */
  virtual void ReturnHandle(void* handle) = 0;

  /*! * \brief sends an exception response to the client with a kTVMStr in payload. */
  virtual void ReturnException(const char* msg) = 0;

  /*! * \brief sends a packed argument sequnce to the client. */
  virtual void ReturnPackedSeq(const TVMValue* arg_values, const int* type_codes, int num_args) = 0;

  /*! * \brief sends a copy of the requested remote data to the client. */
  virtual void ReturnCopyFromRemote(uint8_t* data_ptr, uint64_t num_bytes) = 0;

  /*! * \brief sends an exception response to the client with the last TVM erros as the message. */
  virtual void ReturnLastTVMError() = 0;

  /*! * \brief internal error. */
  virtual void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone) = 0;
};

/*!
 * \brief Execute interface used in MinRPCServer to process different received commands
 */
class MinRPCExecInterface {
 public:
  virtual ~MinRPCExecInterface() {}

  /*! * \brief Execute an Initilize server command. */
  virtual void InitServer(int num_args) = 0;

  /*! * \brief calls a function specified by the call_handle. */
  virtual void NormalCallFunc(uint64_t call_handle, TVMValue* values, int* tcodes,
                              int num_args) = 0;

  /*! * \brief Execute a copy from remote command by sending the data described in arr to the client
   */
  virtual void CopyFromRemote(DLTensor* arr, uint64_t num_bytes, uint8_t* data_ptr) = 0;

  /*! * \brief Execute a copy to remote command by receiving the data described in arr from the
   * client */
  virtual int CopyToRemote(DLTensor* arr, uint64_t num_bytes, uint8_t* data_ptr) = 0;

  /*! * \brief calls a system function specified by the code. */
  virtual void SysCallFunc(RPCCode code, TVMValue* values, int* tcodes, int num_args) = 0;

  /*! * \brief internal error. */
  virtual void ThrowError(RPCServerStatus code, RPCCode info = RPCCode::kNone) = 0;

  /*! * \brief return the ReturnInterface pointer that is used to generate and send the responses.
   */
  virtual MinRPCReturnInterface* GetReturnInterface() = 0;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MINRPC_MINRPC_INTERFACES_H_
