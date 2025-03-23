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
 * \file rpc_module.cc
 * \brief RPC runtime module.
 */
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "rpc_endpoint.h"
#include "rpc_session.h"

namespace tvm {
namespace runtime {

// deleter of RPC remote array
static void RemoteNDArrayDeleter(TVMFFIObject* ptr_obj) {
  auto* ptr = ffi::details::ObjectUnsafe::RawObjectPtrFromUnowned<NDArray::Container>(ptr_obj);
  RemoteSpace* space = static_cast<RemoteSpace*>(ptr->dl_tensor.data);
  if (ptr->manager_ctx != nullptr) {
    space->sess->FreeHandle(ptr->manager_ctx);
  }
  delete space;
  delete ptr;
}

/*!
 * \brief Build a local NDArray with remote backing storage.
 * \param sess the RPCSession which owns the given handle.
 * \param handle A pointer valid on the remote end which should form the `data` field of the
 *     underlying DLTensor.
 * \param template_tensor An empty DLTensor whose shape and dtype fields are used to fill the newly
 *     created array. Needed because it's difficult to pass a shape vector as a PackedFunc arg.
 * \param dev Remote device used with this tensor. Must have non-zero RPCSessMask.
 * \param remote_ndarray_handle The handle returned by RPC server to identify the NDArray.
 */
NDArray NDArrayFromRemoteOpaqueHandle(std::shared_ptr<RPCSession> sess, void* handle,
                                      DLTensor* template_tensor, Device dev,
                                      void* remote_ndarray_handle) {
  ICHECK_EQ(sess->table_index(), GetRPCSessionIndex(dev))
      << "The Device given does not belong to the given session";
  RemoteSpace* space = new RemoteSpace();
  space->sess = sess;
  space->data = handle;
  std::vector<int64_t> shape_vec{template_tensor->shape,
                                 template_tensor->shape + template_tensor->ndim};
  NDArray::Container* data = new NDArray::Container(static_cast<void*>(space), std::move(shape_vec),
                                                    template_tensor->dtype, dev);
  data->manager_ctx = remote_ndarray_handle;
  data->SetDeleter(RemoteNDArrayDeleter);
  return NDArray(GetObjectPtr<Object>(data));
}

/*!
 * \brief A wrapped remote function as a PackedFunc.
 */
class RPCWrappedFunc : public Object {
 public:
  RPCWrappedFunc(void* handle, std::shared_ptr<RPCSession> sess) : handle_(handle), sess_(sess) {}

  void operator()(TVMArgs args, TVMRetValue* rv) const {
    std::vector<AnyView> packed_args(args.size());
    std::vector<std::unique_ptr<DLTensor>> temp_dltensors;

    // scan and check whether we need rewrite these arguments
    // to their remote variant.
    for (int i = 0; i < args.size(); ++i) {
      if (auto opt_str = args[i].TryAs<ffi::String>()) {
        packed_args[i] = opt_str.value().c_str();
        continue;
      }
      packed_args[i] = args[i];
      // run a remote translation to translate RPC related objects to
      // their remote counterparts.
      switch (args[i].type_index()) {
        case ffi::TypeIndex::kTVMFFINDArray: {
          // Pass NDArray as DLTensor
          auto dptr = std::make_unique<DLTensor>(*args[i].operator NDArray().operator->());
          dptr->device = RemoveSessMask(dptr->device);
          dptr->data = static_cast<RemoteSpace*>(dptr->data)->data;
          packed_args[i] = dptr.get();
          temp_dltensors.emplace_back(std::move(dptr));
          break;
        }
        case ffi::TypeIndex::kTVMFFIDLTensorPtr: {
          // translate to a remote view of DLTensor
          auto dptr = std::make_unique<DLTensor>(*args[i].operator DLTensor*());
          dptr->device = RemoveSessMask(dptr->device);
          dptr->data = static_cast<RemoteSpace*>(dptr->data)->data;
          packed_args[i] = dptr.get();
          temp_dltensors.emplace_back(std::move(dptr));
          break;
        }
        case ffi::TypeIndex::kTVMFFIDevice: {
          packed_args[i] = RemoveSessMask(args[i].operator DLDevice());
          break;
        }
        case ffi::TypeIndex::kTVMFFIFunc:
        case ffi::TypeIndex::kTVMFFIRuntimeModule: {
          packed_args[i] = UnwrapRemoteValueToHandle(args[i]);
          break;
        }
      }
    }
    auto set_return = [this, rv](TVMArgs args) { this->WrapRemoteReturnToValue(args, rv); };
    sess_->CallFunc(handle_, ffi::PackedArgs(packed_args.data(), packed_args.size()), set_return);
  }

  ~RPCWrappedFunc() {
    try {
      sess_->FreeHandle(handle_);
    } catch (const Error& e) {
      // fault tolerance to remote close
    }
  }

 private:
  // remote function handle
  void* handle_{nullptr};
  // pointer to the session.
  std::shared_ptr<RPCSession> sess_;

  // unwrap a remote value to the underlying handle.
  void* UnwrapRemoteValueToHandle(const TVMArgValue& arg) const;
  // wrap a remote return via Set
  void WrapRemoteReturnToValue(TVMArgs args, TVMRetValue* rv) const;

  // remove a remote session mask
  Device RemoveSessMask(Device dev) const {
    ICHECK(IsRPCSessionDevice(dev)) << "Can not pass in local device";
    ICHECK_EQ(GetRPCSessionIndex(dev), sess_->table_index())
        << "Can not pass in device with a different remote session";
    return RemoveRPCSessionMask(dev);
  }
};

TVM_REGISTER_OBJECT_TYPE(RPCObjectRefObj);

// RPC that represents a remote module session.
class RPCModuleNode final : public ModuleNode {
 public:
  RPCModuleNode(void* module_handle, std::shared_ptr<RPCSession> sess)
      : module_handle_(module_handle), sess_(sess) {}

  ~RPCModuleNode() {
    if (module_handle_ != nullptr) {
      try {
        sess_->FreeHandle(module_handle_);
      } catch (const Error& e) {
        // fault tolerance to remote close
      }
      module_handle_ = nullptr;
    }
  }

  const char* type_key() const final { return "rpc"; }
  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kRunnable; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "CloseRPCConnection") {
      return PackedFunc([this](TVMArgs, TVMRetValue*) { sess_->Shutdown(); });
    }

    if (module_handle_ == nullptr) {
      return WrapRemoteFunc(sess_->GetFunction(name));
    } else {
      InitRemoteFunc(&remote_mod_get_function_, "tvm.rpc.server.ModuleGetFunction");
      return remote_mod_get_function_(GetRef<Module>(this), name, true);
    }
  }

  String GetSource(const String& format) final {
    LOG(FATAL) << "GetSource for rpc Module is not supported";
    throw;
  }

  PackedFunc GetTimeEvaluator(const std::string& name, Device dev, int number, int repeat,
                              int min_repeat_ms, int limit_zero_time_iterations,
                              int cooldown_interval_ms, int repeats_to_cooldown,
                              int cache_flush_bytes, const std::string& f_preproc_name) {
    InitRemoteFunc(&remote_get_time_evaluator_, "runtime.RPCTimeEvaluator");
    // Remove session mask because we pass dev by parts.
    ICHECK_EQ(GetRPCSessionIndex(dev), sess_->table_index())
        << "ValueError: Need to pass the matched remote device to RPCModule.GetTimeEvaluator";
    dev = RemoveRPCSessionMask(dev);

    if (module_handle_ != nullptr) {
      return remote_get_time_evaluator_(
          GetRef<Module>(this), name, static_cast<int>(dev.device_type), dev.device_id, number,
          repeat, min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms,
          repeats_to_cooldown, cache_flush_bytes, f_preproc_name);
    } else {
      return remote_get_time_evaluator_(
          Optional<Module>(nullptr), name, static_cast<int>(dev.device_type), dev.device_id, number,
          repeat, min_repeat_ms, limit_zero_time_iterations, cooldown_interval_ms,
          repeats_to_cooldown, cache_flush_bytes, f_preproc_name);
    }
  }

  Module LoadModule(std::string name) {
    InitRemoteFunc(&remote_load_module_, "tvm.rpc.server.load_module");
    return remote_load_module_(name);
  }

  void ImportModule(Module other) {
    InitRemoteFunc(&remote_import_module_, "tvm.rpc.server.ImportModule");
    remote_import_module_(GetRef<Module>(this), other);
  }

  const std::shared_ptr<RPCSession>& sess() { return sess_; }

  void* module_handle() const { return module_handle_; }

 private:
  template <typename FType>
  void InitRemoteFunc(FType* func, const std::string& name) {
    if (*func != nullptr) return;
    RPCSession::PackedFuncHandle handle = sess_->GetFunction(name);
    ICHECK(handle != nullptr) << "Cannot found remote function " << name;
    *func = WrapRemoteFunc(handle);
  }

  PackedFunc WrapRemoteFunc(RPCSession::PackedFuncHandle handle) {
    if (handle == nullptr) return PackedFunc();
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess_);
    return PackedFunc([wf](TVMArgs args, TVMRetValue* rv) { return wf->operator()(args, rv); });
  }

  // The module handle
  void* module_handle_{nullptr};
  // The local channel
  std::shared_ptr<RPCSession> sess_;
  // remote function to get time evaluator
  TypedPackedFunc<PackedFunc(Optional<Module>, std::string, int, int, int, int, int, int, int, int,
                             int, std::string)>
      remote_get_time_evaluator_;
  // remote function getter for modules.
  TypedPackedFunc<PackedFunc(Module, std::string, bool)> remote_mod_get_function_;
  // remote function getter for load module
  TypedPackedFunc<Module(std::string)> remote_load_module_;
  // remote function getter for load module
  TypedPackedFunc<void(Module, Module)> remote_import_module_;
};

void* RPCWrappedFunc::UnwrapRemoteValueToHandle(const AnyView& arg) const {
  // TODO(tqchen): only support Module unwrapping for now.
  if (arg.type_index() == kTVMModuleHandle) {
    Module mod = arg;
    std::string tkey = mod->type_key();
    ICHECK_EQ(tkey, "rpc") << "ValueError: Cannot pass a non-RPC module to remote";
    auto* rmod = static_cast<RPCModuleNode*>(mod.operator->());
    ICHECK(rmod->sess() == sess_)
        << "ValueError: Cannot pass in module into a different remote session";
    return rmod->module_handle();
  } else {
    LOG(FATAL) << "ValueError: Cannot pass type " << ffi::TypeIndex2TypeKey(arg.type_index())
               << " as an argument to the remote";
    return nullptr;
  }
}

void RPCWrappedFunc::WrapRemoteReturnToValue(TVMArgs args, TVMRetValue* rv) const {
  int tcode = args[0];

  if (tcode == kTVMNullptr) return;
  if (tcode == kTVMPackedFuncHandle) {
    ICHECK_EQ(args.size(), 2);
    void* handle = args[1];
    auto wf = std::make_shared<RPCWrappedFunc>(handle, sess_);
    *rv = PackedFunc([wf](TVMArgs args, TVMRetValue* rv) { return wf->operator()(args, rv); });
  } else if (tcode == kTVMModuleHandle) {
    ICHECK_EQ(args.size(), 2);
    void* handle = args[1];
    auto n = make_object<RPCModuleNode>(handle, sess_);
    *rv = Module(n);
  } else if (tcode == kTVMObjectHandle) {
    ICHECK_EQ(args.size(), 2);
    void* handle = args[1];
    auto n = make_object<RPCObjectRefObj>(handle, sess_);
    *rv = ObjectRef(n);
  } else if (tcode == kTVMDLTensorHandle || tcode == kTVMNDArrayHandle) {
    ICHECK_EQ(args.size(), 3);
    DLTensor* tensor = args[1];
    void* nd_handle = args[2];
    *rv = NDArrayFromRemoteOpaqueHandle(sess_, tensor->data, tensor,
                                        AddRPCSessionMask(tensor->device, sess_->table_index()),
                                        nd_handle);
  } else {
    ICHECK_EQ(args.size(), 2);
    *rv = args[1];
  }
}

Module CreateRPCSessionModule(std::shared_ptr<RPCSession> sess) {
  auto n = make_object<RPCModuleNode>(nullptr, sess);
  RPCSession::InsertToSessionTable(sess);
  return Module(n);
}

std::shared_ptr<RPCSession> RPCModuleGetSession(Module mod) {
  std::string tkey = mod->type_key();
  ICHECK_EQ(tkey, "rpc") << "ValueError: Cannot pass a non-RPC module to remote";
  auto* rmod = static_cast<RPCModuleNode*>(mod.operator->());
  return rmod->sess();
}

/*!
 * \brief Flush the cache.
 * \param addr The address of data we want to flush
 * \param len The length of data
 */
/*
 * When we are in the tuning of TVM, we will make TVM occupy
 * the cache fully and doesn't flush it during iteration.
 * This has problems then in e2e testing, since arrays that
 * we assume exist in cache (ie. weights) are evicted during e2e runs,
 * which leads to lower performance.
 */
inline void CPUCacheFlushImpl(const char* addr, unsigned int len) {
#if (defined(_M_X64) || defined(__x86_64__) || defined(__aarch64__))

#if defined(__aarch64__)
  size_t ctr_el0 = 0;
  asm volatile("mrs %0, ctr_el0" : "=r"(ctr_el0));
  const size_t cache_line = 4 << ((ctr_el0 >> 16) & 15);
#else
  const size_t cache_line = 64;
#endif

  if (addr == nullptr || len <= 0) {
    return;
  }

  for (uintptr_t uptr = (uintptr_t)addr & ~(cache_line - 1); uptr < (uintptr_t)addr + len;
       uptr += cache_line) {
#if defined(__aarch64__)
    asm volatile("dc civac, %0\n\t" : : "r"(reinterpret_cast<const void*>(uptr)) : "memory");
#else
    _mm_clflush(reinterpret_cast<const void*>(uptr));
#endif
  }

#if defined(__aarch64__)
  asm volatile("dmb ishst" : : : "memory");
#endif

#endif
}

inline void CPUCacheFlush(int begin_index, const TVMArgs& args) {
  for (int i = begin_index; i < args.size(); i++) {
    CPUCacheFlushImpl(static_cast<char*>((args[i].operator DLTensor*()->data)),
                      GetDataSize(*(args[i].operator DLTensor*())));
  }
}

TVM_REGISTER_GLOBAL("runtime.RPCTimeEvaluator")
    .set_body_typed([](Optional<Module> opt_mod, std::string name, int device_type, int device_id,
                       int number, int repeat, int min_repeat_ms, int limit_zero_time_iterations,
                       int cooldown_interval_ms, int repeats_to_cooldown, int cache_flush_bytes,
                       std::string f_preproc_name) {
      Device dev;
      dev.device_type = static_cast<DLDeviceType>(device_type);
      dev.device_id = device_id;
      if (opt_mod.defined()) {
        Module m = opt_mod.value();
        std::string tkey = m->type_key();
        if (tkey == "rpc") {
          return static_cast<RPCModuleNode*>(m.operator->())
              ->GetTimeEvaluator(name, dev, number, repeat, min_repeat_ms,
                                 limit_zero_time_iterations, cooldown_interval_ms,
                                 repeats_to_cooldown, cache_flush_bytes, f_preproc_name);
        } else {
          PackedFunc f_preproc;
          if (!f_preproc_name.empty()) {
            auto* pf_preproc = runtime::Registry::Get(f_preproc_name);
            ICHECK(pf_preproc != nullptr)
                << "Cannot find " << f_preproc_name << " in the global function";
            f_preproc = *pf_preproc;
          }
          PackedFunc pf = m.GetFunction(name, true);
          CHECK(pf != nullptr) << "Cannot find " << name << " in the global registry";
          return profiling::WrapTimeEvaluator(pf, dev, number, repeat, min_repeat_ms,
                                              limit_zero_time_iterations, cooldown_interval_ms,
                                              repeats_to_cooldown, cache_flush_bytes, f_preproc);
        }
      } else {
        auto* pf = runtime::Registry::Get(name);
        ICHECK(pf != nullptr) << "Cannot find " << name << " in the global function";
        PackedFunc f_preproc;
        if (!f_preproc_name.empty()) {
          auto* pf_preproc = runtime::Registry::Get(f_preproc_name);
          ICHECK(pf_preproc != nullptr)
              << "Cannot find " << f_preproc_name << " in the global function";
          f_preproc = *pf_preproc;
        }
        return profiling::WrapTimeEvaluator(*pf, dev, number, repeat, min_repeat_ms,
                                            limit_zero_time_iterations, cooldown_interval_ms,
                                            repeats_to_cooldown, cache_flush_bytes, f_preproc);
      }
    });

TVM_REGISTER_GLOBAL("cache_flush_cpu_non_first_arg").set_body([](TVMArgs args, TVMRetValue* rv) {
  CPUCacheFlush(1, args);
});

// server function registration.
TVM_REGISTER_GLOBAL("tvm.rpc.server.ImportModule").set_body_typed([](Module parent, Module child) {
  parent->Import(child);
});

TVM_REGISTER_GLOBAL("tvm.rpc.server.ModuleGetFunction")
    .set_body_typed([](Module parent, std::string name, bool query_imports) {
      return parent->GetFunction(name, query_imports);
    });

// functions to access an RPC module.
TVM_REGISTER_GLOBAL("rpc.LoadRemoteModule").set_body_typed([](Module sess, std::string name) {
  std::string tkey = sess->type_key();
  ICHECK_EQ(tkey, "rpc");
  return static_cast<RPCModuleNode*>(sess.operator->())->LoadModule(name);
});

TVM_REGISTER_GLOBAL("rpc.ImportRemoteModule").set_body_typed([](Module parent, Module child) {
  std::string tkey = parent->type_key();
  ICHECK_EQ(tkey, "rpc");
  static_cast<RPCModuleNode*>(parent.operator->())->ImportModule(child);
});

TVM_REGISTER_GLOBAL("rpc.SessTableIndex").set_body([](TVMArgs args, TVMRetValue* rv) {
  Module m = args[0];
  std::string tkey = m->type_key();
  ICHECK_EQ(tkey, "rpc");
  *rv = static_cast<RPCModuleNode*>(m.operator->())->sess()->table_index();
});

TVM_REGISTER_GLOBAL("tvm.rpc.NDArrayFromRemoteOpaqueHandle")
    .set_body_typed([](Module mod, void* remote_array, DLTensor* template_tensor, Device dev,
                       void* ndarray_handle) -> NDArray {
      return NDArrayFromRemoteOpaqueHandle(RPCModuleGetSession(mod), remote_array, template_tensor,
                                           dev, ndarray_handle);
    });

}  // namespace runtime
}  // namespace tvm
