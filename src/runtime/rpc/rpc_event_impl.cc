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
 * \file rpc_event_impl.cc
 * \brief Event driven RPC server implementation.
 */
#include <tvm/ffi/function.h>

#include <memory>

#include "rpc_endpoint.h"
#include "rpc_local_session.h"

namespace tvm {
namespace runtime {

ffi::Function CreateEventDrivenServer(ffi::Function fsend, std::string name,
                                      std::string remote_key) {
  static ffi::Function frecv(
      [](ffi::PackedArgs args, ffi::Any* rv) { LOG(FATAL) << "Do not allow explicit receive"; });

  auto ch = std::make_unique<CallbackChannel>(fsend, frecv);
  std::shared_ptr<RPCEndpoint> sess = RPCEndpoint::Create(std::move(ch), name, remote_key);
  return ffi::Function([sess](ffi::PackedArgs args, ffi::Any* rv) {
    int ret = sess->ServerAsyncIOEventHandler(args[0].cast<std::string>(), args[1].cast<int>());
    *rv = ret;
  });
}

TVM_FFI_REGISTER_GLOBAL("rpc.CreateEventDrivenServer").set_body_typed(CreateEventDrivenServer);
}  // namespace runtime
}  // namespace tvm
