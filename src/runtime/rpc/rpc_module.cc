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
 * \file rpc_device_api.cc
 * \brief RPC module.
 */
#include <tvm/runtime/registry.h>
#include <memory>
#include <cstring>
#include "rpc_session.h"

namespace tvm {
namespace runtime {

// Wrapped remote function to packed func.
class RPCWrappedFunc {
 public:
  RPCWrappedFunc(void* handle,
                 std::shared_ptr<RPCSession> sess)
      : handle_(handle), sess_(sess) {
    fwrap_ = PackedFunc([sess](TVMArgs args, TVMRetValue* rv) {
        WrapRemote(sess, args, rv);
      });
  }

  void operator()(TVMArgs args, TVMRetValue *rv) const {
    sess_->CallFunc(handle_, args, rv, &fwrap_);
  }
  ~RPCWrappedFunc() {
    try {
      sess_->CallRemote(RPCCode::kFreeFunc, handle_);
    } catch (const dmlc::Error& e) {
      // fault tolerance to remote close
    }
  }

  static void WrapRemote(std::shared_ptr<RPCSession> sess,
                         TVMArgs args,
                         TVMRetValue* rv);

  // deleter of RPC remote array
  static void RemoteNDArrayDeleter(NDArray::Container* ptr) {
    RemoteSpace* space = static_cast<RemoteSpace*>(ptr->dl_tensor.data);
    space->sess->CallRemote(RPCCode::kNDArrayFree, ptr->manager_ctx);
    delete space;
    delete ptr;
  }
  // wrap return value as remote NDArray.
  static NDArray WrapRemoteNDArray(std::shared_ptr<RPCSession> sess,
                                   DLTensor* tensor,
                                   void* nd_handle) {
    NDArray::Container* data = new NDArray::Container();
    data->manager_ctx = nd_handle;
    data->deleter = RemoteNDArrayDeleter;
    RemoteSpace* space = new RemoteSpace();
    space->sess = sess;
    space->data = tensor->data;
    data->dl_tensor.data = space;
    NDArray ret(data);
    // RAII now in effect
    data->shape_ = std::vector<int64_t>(
        tensor->shape, tensor->shape + tensor->ndim);
    data->dl_tensor.shape = dmlc::BeginPtr(data->shape_);
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup dtype
    data->dl_tensor.dtype = tensor->dtype;
    // setup ctx, encode as remote session
    data->dl_tensor.ctx.device_id = tensor->ctx.device_id;
    data->dl_tensor.ctx.device_type = static_cast<DLDeviceType>(
        static_cast<int>(tensor->ctx.device_type) +
        kRPCSessMask * (sess->table_index() + 1));
    // check strides.
    CHECK(tensor->strides == nullptr);
    // setup byteoffset
    data->dl_tensor.byte_offset = tensor->byte_offset;
    return ret;
  }

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
      try {
        sess_->CallRemote(RPCCode::kModuleFree, module_handle_);
      } catch (const dmlc::Error& e) {
        // fault tolerance to remote close
      }
      module_handle_ = nullptr;
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
                              int repeat,
                              int min_repeat_ms) {
    RPCFuncHandle handle = GetFuncHandle(name);
    if (handle == nullptr) return PackedFunc();
    handle = sess_->GetTimeEvaluator(handle, ctx, number, repeat, min_repeat_ms);
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
                                TVMArgs args,
                                TVMRetValue *rv) {
  void* handle = args.values[0].v_handle;
  int tcode = args.type_codes[0];

  if (handle == nullptr) return;
  if (tcode == kFuncHandle) {
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess);
    *rv = PackedFunc([wf](TVMArgs args, TVMRetValue* rv) {
        return wf->operator()(args, rv);
      });
  } else if (tcode == kModuleHandle) {
    std::shared_ptr<RPCModuleNode> n =
        std::make_shared<RPCModuleNode>(handle, sess);
    *rv = Module(n);
  } else if (tcode == kArrayHandle || tcode == kNDArrayContainer) {
    CHECK_EQ(args.size(), 2);
    DLTensor* tensor = args[0];
    void* nd_handle = args[1];
    *rv = WrapRemoteNDArray(sess, tensor, nd_handle);
  } else {
    LOG(FATAL) << "Cannot wrap tcode=" << tcode;
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
          ->GetTimeEvaluator(args[1], ctx, args[4], args[5], args[6]);
    } else {
      *rv = WrapTimeEvaluator(
          m.GetFunction(args[1], false), ctx, args[4], args[5], args[6]);
    }
  });

TVM_REGISTER_GLOBAL("rpc._LoadRemoteModule")
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

TVM_REGISTER_GLOBAL("rpc._ImportRemoteModule")
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

TVM_REGISTER_GLOBAL("rpc._ModuleHandle")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    CHECK_EQ(tkey, "rpc");
    *rv = static_cast<RPCModuleNode*>(m.operator->())->module_handle();
  });

TVM_REGISTER_GLOBAL("rpc._SessTableIndex")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    Module m = args[0];
    std::string tkey = m->type_key();
    CHECK_EQ(tkey, "rpc");
    *rv = static_cast<RPCModuleNode*>(m.operator->())->sess()->table_index();
  });

}  // namespace runtime
}  // namespace tvm
