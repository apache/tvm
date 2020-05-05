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
 * \file rpc_local_session.h
 * \brief Local session that directs all request to the local runtime API.
 */
#ifndef TVM_RUNTIME_RPC_RPC_LOCAL_SESSION_H_
#define TVM_RUNTIME_RPC_RPC_LOCAL_SESSION_H_

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/device_api.h>
#include <functional>
#include <string>
#include "rpc_session.h"

namespace tvm {
namespace runtime {

/*!
 * \brief A local session that directly use the handle repr of the
 *        local tvm runtime objects on the same process.
 */
class LocalSession : public RPCSession {
 public:
  // function overrides
  PackedFuncHandle GetFunction(const std::string& name) final;

  void CallFunc(PackedFuncHandle func,
                const TVMValue* arg_values,
                const int* arg_type_codes,
                int num_args,
                const FEncodeReturn& fencode_return) final;

  void CopyToRemote(void* from,
                    size_t from_offset,
                    void* to,
                    size_t to_offset,
                    size_t nbytes,
                    TVMContext ctx_to,
                    DLDataType type_hint) final;

  void CopyFromRemote(void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t nbytes,
                      TVMContext ctx_from,
                      DLDataType type_hint) final;

  void FreeHandle(void* handle, int type_code) final;

  DeviceAPI* GetDeviceAPI(TVMContext ctx, bool allow_missing = false) final;

 protected:
  /*!
   * \brief Internal implementation of GetFunction.
   * \param name The name of the function.
   * \return The corresponding PackedFunc.
   */
  virtual PackedFunc GetFunctionInternal(const std::string& name);
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_LOCAL_SESSION_H_
