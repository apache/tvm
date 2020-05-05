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
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <utility>
#include "rpc_session.h"

namespace tvm {
namespace runtime {

class RPCDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {
    auto remote_ctx = RemoveSessMask(ctx);
    GetSess(ctx)->GetDeviceAPI(remote_ctx)->SetDevice(remote_ctx);
  }

  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    auto remote_ctx = RemoveSessMask(ctx);
    GetSess(ctx)->GetDeviceAPI(remote_ctx)->GetAttr(remote_ctx, kind, rv);
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       DLDataType type_hint) final {
    auto sess = GetSess(ctx);
    auto remote_ctx = RemoveSessMask(ctx);
    void *data = sess->GetDeviceAPI(remote_ctx)->AllocDataSpace(
        remote_ctx, nbytes, alignment, type_hint);

    RemoteSpace* space = new RemoteSpace();
    space->data = data;
    space->sess = std::move(sess);
    return space;
  }
  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    RemoteSpace* space = static_cast<RemoteSpace*>(ptr);
    auto remote_ctx = RemoveSessMask(ctx);
    try {
      GetSess(ctx)->GetDeviceAPI(remote_ctx)->FreeDataSpace(
          remote_ctx, space->data);
    } catch (const dmlc::Error& e) {
      // fault tolerance to remote close.
    }
    delete space;
  }
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      DLDataType type_hint,
                      TVMStreamHandle stream) final {
    int from_dev_type = ctx_from.device_type;
    int to_dev_type = ctx_to.device_type;
    if (from_dev_type > kRPCSessMask &&
        to_dev_type > kRPCSessMask) {
      CHECK(ctx_from.device_type == ctx_to.device_type)
          << "Cannot copy across two different remote session";
      auto remote_ctx_from = RemoveSessMask(ctx_from);
      auto remote_ctx_to = RemoveSessMask(ctx_to);
      auto remote_ctx = remote_ctx_from;
      if (remote_ctx.device_type == kDLCPU) remote_ctx = remote_ctx_to;
      GetSess(ctx_from)->GetDeviceAPI(remote_ctx)
          ->CopyDataFromTo(static_cast<const RemoteSpace*>(from)->data, from_offset,
                           static_cast<const RemoteSpace*>(to)->data, to_offset,
                           size, remote_ctx_from, remote_ctx_to, type_hint, stream);
    } else if (from_dev_type > kRPCSessMask &&
               to_dev_type == kDLCPU) {
      auto remote_ctx_from = RemoveSessMask(ctx_from);
      GetSess(ctx_from)->CopyFromRemote(
          static_cast<const RemoteSpace*>(from)->data, from_offset,
          to, to_offset, size, remote_ctx_from, type_hint);
    } else if (from_dev_type == kDLCPU &&
               to_dev_type > kRPCSessMask) {
      auto remote_ctx_to = RemoveSessMask(ctx_to);
      GetSess(ctx_to)->CopyToRemote(
          const_cast<void*>(from), from_offset,
          static_cast<const RemoteSpace*>(to)->data, to_offset,
          size, remote_ctx_to, type_hint);
    } else {
      LOG(FATAL) << "expect copy from/to remote or between remote";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    auto remote_ctx = RemoveSessMask(ctx);
    GetSess(ctx)->GetDeviceAPI(remote_ctx)->StreamSync(ctx, stream);
  }

 private:
  std::shared_ptr<RPCSession> GetSess(TVMContext ctx) {
    int dev_type = ctx.device_type;
    CHECK_GE(dev_type, kRPCSessMask);
    int tbl_index = dev_type / kRPCSessMask -  1;
    return RPCSession::Get(tbl_index);
  }

  static TVMContext RemoveSessMask(TVMContext ctx) {
    ctx.device_type = static_cast<DLDeviceType>(ctx.device_type % kRPCSessMask);
    return ctx;
  }
};

TVM_REGISTER_GLOBAL("device_api.rpc")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    static RPCDeviceAPI inst;
    DeviceAPI* ptr = &inst;
    *rv = static_cast<void*>(ptr);
  });
}  // namespace runtime
}  // namespace tvm
