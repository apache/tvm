/*!
 *  Copyright (c) 2017 by Contributors
 * \file rpc_device_api.cc
 * \brief RPC module.
 */
#include <tvm/runtime/registry.h>
#include <memory>
#include <cstring>
#include "./rpc_session.h"

namespace tvm {
namespace runtime {

// Wrapped remote function to packed func.
struct RPCWrappedFunc {
 public:
  RPCWrappedFunc(void* handle,
                 std::shared_ptr<RPCSession> sess)
      : handle_(handle), sess_(sess) {
    fwrap_ = PackedFunc([sess](TVMArgs args, TVMRetValue* rv) {
        WrapRemote(sess, args.values[0].v_handle, args.type_codes[0], rv);
      });
  }

  void operator()(TVMArgs args, TVMRetValue *rv) const {
    sess_->CallFunc(handle_, args, rv, &fwrap_);
  }
  ~RPCWrappedFunc() {
    sess_->CallRemote(RPCCode::kFreeFunc, handle_);
  }

  static void WrapRemote(std::shared_ptr<RPCSession> sess,
                         void* handle,
                         int tcode,
                         TVMRetValue* rv);

 private:
  PackedFunc fwrap_;
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
                              int number,
                              int repeat) {
    RPCFuncHandle handle = GetFuncHandle(name);
    if (handle == nullptr) return PackedFunc();
    handle = sess_->GetTimeEvaluator(handle, ctx, number, repeat);
    return WrapRemote(handle);
  }

  void* module_handle() const {
    return module_handle_;
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
  // Wrap function to wrap remote module/function.
  PackedFunc fwrap_;
};

void RPCWrappedFunc::WrapRemote(std::shared_ptr<RPCSession> sess,
                                void* handle,
                                int tcode,
                                TVMRetValue *rv) {
  if (handle == nullptr) return;
  if (tcode == kFuncHandle) {
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess);
    *rv = PackedFunc([wf](TVMArgs args, TVMRetValue* rv) {
        return wf->operator()(args, rv);
      });
  } else {
    CHECK_EQ(tcode, kModuleHandle);
    std::shared_ptr<RPCModuleNode> n =
        std::make_shared<RPCModuleNode>(handle, sess);
    *rv = Module(n);
  }
}

Module CreateRPCModule(std::shared_ptr<RPCSession> sess) {
  std::shared_ptr<RPCModuleNode> n =
      std::make_shared<RPCModuleNode>(nullptr, sess);
  return Module(n);
}

TVM_REGISTER_GLOBAL("module._RPCTimeEvaluator")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    TVMContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(args[2].operator int());
    ctx.device_id = args[3];
    if (tkey == "rpc") {
      *rv = static_cast<RPCModuleNode*>(m.operator->())
          ->GetTimeEvaluator(args[1], ctx, args[4], args[5]);
    } else {
      *rv = WrapTimeEvaluator(
          m.GetFunction(args[1], false), ctx, args[4], args[5]);
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

TVM_REGISTER_GLOBAL("contrib.rpc._ImportRemoteModule")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module parent = args[0];
    Module child = args[1];
    CHECK(!std::strcmp(parent->type_key(), "rpc") &&
          !std::strcmp(child->type_key(), "rpc"));
    auto* pmod = static_cast<RPCModuleNode*>(parent.operator->());
    auto* cmod = static_cast<RPCModuleNode*>(child.operator->());
    CHECK(pmod->sess().get() == cmod->sess().get())
        << "Import of remote module need to belong to same session.";
    pmod->sess()->CallRemote(RPCCode::kModuleImport,
                             pmod->module_handle(),
                             cmod->module_handle());
  });

TVM_REGISTER_GLOBAL("contrib.rpc._ModuleHandle")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    CHECK_EQ(tkey, "rpc");
    *rv = static_cast<RPCModuleNode*>(m.operator->())->module_handle();
  });

TVM_REGISTER_GLOBAL("contrib.rpc._SessTableIndex")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    CHECK_EQ(tkey, "rpc");
    *rv = static_cast<RPCModuleNode*>(m.operator->())->sess()->table_index();
  });
}  // namespace runtime
}  // namespace tvm
