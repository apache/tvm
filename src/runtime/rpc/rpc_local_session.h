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

#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>

#include <functional>
#include <string>
#include <utility>

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
  PackedFuncHandle GetFunction(const std::string& name) override;

  void CallFunc(PackedFuncHandle func, const TVMValue* arg_values, const int* arg_type_codes,
                int num_args, const FEncodeReturn& fencode_return) override;

  void CopyToRemote(void* from_bytes, DLTensor* to, uint64_t nbytes) override;

  void CopyFromRemote(DLTensor* from, void* to_bytes, uint64_t nbytes) override;

  void FreeHandle(void* handle, int type_code) override;

  DeviceAPI* GetDeviceAPI(Device dev, bool allow_missing = false) override;

  bool IsLocalSession() const override { return true; }

 protected:
  /*!
   * \brief internal encode return function.
   * \param rv The return value.
   * \param encode_return The encoding function.
   */
  void EncodeReturn(TVMRetValue rv, const FEncodeReturn& encode_return);
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_RPC_RPC_LOCAL_SESSION_H_
