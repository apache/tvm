/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_event_impl.cc
 * \brief Event based RPC server implementation.
 */
#include <tvm/runtime/registry.h>
#include <memory>
#include "./rpc_session.h"

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

PackedFunc CreateEventDrivenServer(PackedFunc fsend, std::string name) {
  std::unique_ptr<CallbackChannel> ch(new CallbackChannel(fsend));
  std::shared_ptr<RPCSession> sess = RPCSession::Create(std::move(ch), name);
  return PackedFunc([sess](TVMArgs args, TVMRetValue* rv) {
      int ret = sess->ServerEventHandler(args[0], args[1]);
      *rv = ret;
    });
}

TVM_REGISTER_GLOBAL("contrib.rpc._CreateEventDrivenServer")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = CreateEventDrivenServer(args[0], args[1]);
  });
}  // namespace runtime
}  // namespace tvm
