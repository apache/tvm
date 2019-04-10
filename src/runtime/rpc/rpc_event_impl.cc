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
 * \file rpc_event_impl.cc
 * \brief Event based RPC server implementation.
 */
#include <tvm/runtime/registry.h>
#include <memory>
#include "rpc_session.h"

namespace tvm {
namespace runtime {

class CallbackChannel final : public RPCChannel {
 public:
  explicit CallbackChannel(PackedFunc fsend)
      : fsend_(fsend) {}

  size_t Send(const void* data, size_t size) final {
    TVMByteArray bytes;
    bytes.data = static_cast<const char*>(data);
    bytes.size = size;
    uint64_t ret = fsend_(bytes);
    return static_cast<size_t>(ret);
  }

  size_t Recv(void* data, size_t size) final {
    LOG(FATAL) << "Do not allow explicit receive for";
    return 0;
  }

 private:
  PackedFunc fsend_;
};

PackedFunc CreateEventDrivenServer(PackedFunc fsend,
                                   std::string name,
                                   std::string remote_key) {
  std::unique_ptr<CallbackChannel> ch(new CallbackChannel(fsend));
  std::shared_ptr<RPCSession> sess =
      RPCSession::Create(std::move(ch), name, remote_key);
  return PackedFunc([sess](TVMArgs args, TVMRetValue* rv) {
      int ret = sess->ServerEventHandler(args[0], args[1]);
      *rv = ret;
    });
}

TVM_REGISTER_GLOBAL("rpc._CreateEventDrivenServer")
.set_body_typed(CreateEventDrivenServer);
}  // namespace runtime
}  // namespace tvm
