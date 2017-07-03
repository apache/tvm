/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_device_api.cc
 * \brief RPC module.
 */
#include <tvm/runtime/registry.h>
#include <memory>
#include "./rpc_session.h"

namespace tvm {
namespace runtime {

const int kRPCMagic = 0xff271;

// Wrapped remote function to packed func.
struct RPCWrappedFunc {
 public:
  RPCWrappedFunc(void* handle, std::shared_ptr<RPCSession> sess)
      : handle_(handle), sess_(sess) {}
  void operator()(TVMArgs args, TVMRetValue *rv) const {
    sess_->CallFunc(handle_, args, rv);
  }
  ~RPCWrappedFunc() {
    sess_->CallRemote(RPCCode::kFreeFunc, handle_);
  }

 private:
  void* handle_{nullptr};
  std::shared_ptr<RPCSession> sess_;
};

// RPC that represents a remote module session.
class RPCModuleNode final : public ModuleNode {
 public:
  RPCModuleNode(void* module_handle, std::shared_ptr<RPCSession> sess)
      : module_handle_(module_handle), sess_(sess) {
  }
  ~RPCModuleNode() {
    if (module_handle_ != nullptr) {
      sess_->CallRemote(RPCCode::kModuleFree, module_handle_);
    }
  }

  const char* type_key() const final {
    return "rpc";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    RPCFuncHandle handle = GetFuncHandle(name);
    return WrapRemote(handle);
  }

  std::string GetSource(const std::string& format) final {
    if (module_handle_ != nullptr) {
      std::string ret =  sess_->CallRemote(
          RPCCode::kModuleGetSource, module_handle_, format);
    }
    return "";
  }

  std::shared_ptr<RPCSession>& sess() {
    return sess_;
  }

  PackedFunc GetTimeEvaluator(const std::string& name,
                              TVMContext ctx,
                              int nstep) {
    RPCFuncHandle handle = GetFuncHandle(name);
    if (handle == nullptr) return PackedFunc();
    handle = sess_->GetTimeEvaluator(handle, ctx, nstep);
    return WrapRemote(handle);
  }

 private:
  PackedFunc WrapRemote(RPCFuncHandle handle) {
    if (handle == nullptr) return PackedFunc();
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess_);
    return PackedFunc([wf](TVMArgs args, TVMRetValue* rv) {
        return wf->operator()(args, rv);
      });
  }

  RPCFuncHandle GetFuncHandle(const std::string& name) {
    RPCFuncHandle handle = nullptr;
    if (module_handle_ == nullptr) {
      handle = sess_->CallRemote(RPCCode::kGetGlobalFunc, name);
    } else {
      handle = sess_->CallRemote(
          RPCCode::kModuleGetFunc, module_handle_, name);
    }
    return handle;
  }
  // The module handle
  void* module_handle_{nullptr};
  // The local channel
  std::shared_ptr<RPCSession> sess_;
};

Module RPCConnect(std::string url, int port) {
  common::TCPSocket sock;
  common::SockAddr addr(url.c_str(), port);
  sock.Create();
  CHECK(sock.Connect(addr))
      << "Connect to " << addr.AsString() << " failed";
  // hand shake
  int code = kRPCMagic;
  CHECK_EQ(sock.SendAll(&code, sizeof(code)), sizeof(code));
  CHECK_EQ(sock.RecvAll(&code, sizeof(code)), sizeof(code));
  if (code != kRPCMagic) {
    sock.Close();
    LOG(FATAL) << "URL " << url << ":" << port << " is not TVM RPC server";
  }
  std::shared_ptr<RPCModuleNode> n =
      std::make_shared<RPCModuleNode>(nullptr, RPCSession::Create(sock));
  return Module(n);
}

void RPCServerLoop(int sockfd) {
  common::TCPSocket sock(
      static_cast<common::TCPSocket::SockType>(sockfd));
  RPCSession::Create(sock)->ServerLoop();
}

TVM_REGISTER_GLOBAL("contrib.rpc._Connect")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = RPCConnect(args[0], args[1]);
  });

TVM_REGISTER_GLOBAL("module._RPCTimeEvaluator")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    TVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(args[2].operator int());
    ctx.device_id = args[3];
    if (tkey == "rpc") {
      *rv = static_cast<RPCModuleNode*>(m.operator->())
          ->GetTimeEvaluator(args[1], ctx, args[4]);
    } else {
      *rv = WrapTimeEvaluator(
          m.GetFunction(args[1], false), ctx, args[4]);
    }
  });

TVM_REGISTER_GLOBAL("contrib.rpc._LoadRemoteModule")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    CHECK_EQ(tkey, "rpc");
    auto& sess = static_cast<RPCModuleNode*>(m.operator->())->sess();
    void* mhandle = sess->CallRemote(RPCCode::kModuleLoad, args[1]);
    std::shared_ptr<RPCModuleNode> n =
        std::make_shared<RPCModuleNode>(mhandle, sess);
    *rv = Module(n);
  });

TVM_REGISTER_GLOBAL("contrib.rpc._SessTableIndex")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    CHECK_EQ(tkey, "rpc");
    *rv = static_cast<RPCModuleNode*>(m.operator->())->sess()->table_index();
  });

TVM_REGISTER_GLOBAL("contrib.rpc._ServerLoop")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    RPCServerLoop(args[0]);
  });
}  // namespace runtime
}  // namespace tvm
