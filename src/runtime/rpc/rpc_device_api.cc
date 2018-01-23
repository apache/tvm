/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_device_api.cc
 */
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include "./rpc_session.h"

namespace tvm {
namespace runtime {

class RPCDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(TVMContext ctx) final {
    GetSess(ctx)->CallRemote(
        RPCCode::kDevSetDevice, ctx);
  }
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    *rv = GetSess(ctx)->CallRemote(
        RPCCode::kDevGetAttr, ctx, static_cast<int>(kind));
  }
  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    auto sess = GetSess(ctx);
    void *data = sess->CallRemote(
            RPCCode::kDevAllocData, ctx, nbytes, alignment, type_hint);
    RemoteSpace* space = new RemoteSpace();
    space->data = data;
    space->sess = std::move(sess);
    return space;
  }
  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    RemoteSpace* space = static_cast<RemoteSpace*>(ptr);
    GetSess(ctx)->CallRemote(
        RPCCode::kDevFreeData, ctx, space->data);
    delete space;
  }
  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMStreamHandle stream) final {
    int from_dev_type = ctx_from.device_type;
    int to_dev_type = ctx_to.device_type;
    if (from_dev_type > kRPCSessMask &&
        to_dev_type > kRPCSessMask) {
      CHECK(ctx_from.device_type == ctx_to.device_type)
          << "Cannot copy across two different remote session";
      GetSess(ctx_from)->CallRemote(
          RPCCode::kCopyAmongRemote,
          static_cast<const RemoteSpace*>(from)->data, from_offset,
          static_cast<const RemoteSpace*>(to)->data, to_offset,
          size,  ctx_from, ctx_to, stream);
    } else if (from_dev_type > kRPCSessMask &&
               to_dev_type == kDLCPU) {
      GetSess(ctx_from)->CopyFromRemote(
          static_cast<const RemoteSpace*>(from)->data, from_offset,
          to, to_offset, size,
          ctx_from);
    } else if (from_dev_type == kDLCPU &&
               to_dev_type > kRPCSessMask) {
      GetSess(ctx_to)->CopyToRemote(
          (void*)from, from_offset,  // NOLINT(*)
          static_cast<const RemoteSpace*>(to)->data, to_offset,
          size, ctx_to);
    } else {
      LOG(FATAL) << "expect copy from/to remote or between remote";
    }
  }
  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    GetSess(ctx)->CallRemote(
        RPCCode::kDevStreamSync, ctx, stream);
  }

 private:
  std::shared_ptr<RPCSession> GetSess(TVMContext ctx) {
    int dev_type = ctx.device_type;
    CHECK_GE(dev_type, kRPCSessMask);
    int tbl_index = dev_type / kRPCSessMask -  1;
    return RPCSession::Get(tbl_index);
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
