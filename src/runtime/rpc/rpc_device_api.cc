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
 * \file rpc_device_api.cc
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/logging.h>

#include <utility>

#include "rpc_session.h"

namespace tvm {
namespace runtime {

class RPCDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {
    auto remote_ctx = RemoveRPCSessionMask(ctx);
    GetSess(ctx)->GetDeviceAPI(remote_ctx)->SetDevice(remote_ctx);
  }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    auto remote_ctx = RemoveRPCSessionMask(ctx);
    GetSess(ctx)->GetDeviceAPI(remote_ctx)->GetAttr(remote_ctx, kind, rv);
  }

  void* AllocDataSpace(TVMContext ctx, size_t nbytes, size_t alignment,
                       DLDataType type_hint) final {
    auto sess = GetSess(ctx);
    auto remote_ctx = RemoveRPCSessionMask(ctx);
    void* data =
        sess->GetDeviceAPI(remote_ctx)->AllocDataSpace(remote_ctx, nbytes, alignment, type_hint);

    RemoteSpace* space = new RemoteSpace();
    space->data = data;
    space->sess = std::move(sess);
    return space;
  }
  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    RemoteSpace* space = static_cast<RemoteSpace*>(ptr);
    auto remote_ctx = RemoveRPCSessionMask(ctx);
    try {
      GetSess(ctx)->GetDeviceAPI(remote_ctx)->FreeDataSpace(remote_ctx, space->data);
    } catch (const dmlc::Error& e) {
      // fault tolerance to remote close.
    }
    delete space;
  }
  void CopyDataFromTo(const void* from, size_t from_offset, void* to, size_t to_offset, size_t size,
                      TVMContext ctx_from, TVMContext ctx_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    if (IsRPCSessionContext(ctx_from) && IsRPCSessionContext(ctx_to)) {
      ICHECK(ctx_from.device_type == ctx_to.device_type)
          << "Cannot copy across two different remote session";
      auto remote_ctx_from = RemoveRPCSessionMask(ctx_from);
      auto remote_ctx_to = RemoveRPCSessionMask(ctx_to);
      auto remote_ctx = remote_ctx_from;
      if (remote_ctx.device_type == kDLCPU) remote_ctx = remote_ctx_to;
      GetSess(ctx_from)
          ->GetDeviceAPI(remote_ctx)
          ->CopyDataFromTo(static_cast<const RemoteSpace*>(from)->data, from_offset,
                           static_cast<const RemoteSpace*>(to)->data, to_offset, size,
                           remote_ctx_from, remote_ctx_to, type_hint, stream);
    } else if (IsRPCSessionContext(ctx_from) && ctx_to.device_type == kDLCPU) {
      auto remote_ctx_from = RemoveRPCSessionMask(ctx_from);
      GetSess(ctx_from)->CopyFromRemote(static_cast<const RemoteSpace*>(from)->data, from_offset,
                                        to, to_offset, size, remote_ctx_from, type_hint);
    } else if (ctx_from.device_type == kDLCPU && IsRPCSessionContext(ctx_to)) {
      auto remote_ctx_to = RemoveRPCSessionMask(ctx_to);
      GetSess(ctx_to)->CopyToRemote(const_cast<void*>(from), from_offset,
                                    static_cast<const RemoteSpace*>(to)->data, to_offset, size,
                                    remote_ctx_to, type_hint);
    } else {
      LOG(FATAL) << "expect copy from/to remote or between remote";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    auto remote_ctx = RemoveRPCSessionMask(ctx);
    GetSess(ctx)->GetDeviceAPI(remote_ctx)->StreamSync(remote_ctx, stream);
  }

 private:
  std::shared_ptr<RPCSession> GetSess(TVMContext ctx) {
    int tbl_index = GetRPCSessionIndex(ctx);
    return RPCSession::Get(tbl_index);
  }
};

TVM_REGISTER_GLOBAL("device_api.rpc").set_body([](TVMArgs args, TVMRetValue* rv) {
  static RPCDeviceAPI inst;
  DeviceAPI* ptr = &inst;
  *rv = static_cast<void*>(ptr);
});
}  // namespace runtime
}  // namespace tvm
