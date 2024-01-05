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

/*
 * \file tvmjs_support.cc
 * \brief Support functions to be linked with wasm_runtime to provide
 *        PackedFunc callbacks in tvmjs.
 *        We do not need to link this file in standalone wasm.
 */

// configurations for tvm logging
#define TVM_LOG_STACK_TRACE 0
#define TVM_LOG_DEBUG 0
#define TVM_LOG_CUSTOMIZE 1
#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "../../src/runtime/rpc/rpc_local_session.h"

extern "C" {
// --- Additional C API for the Wasm runtime ---
/*!
 * \brief Allocate space aligned to 64 bit.
 * \param size The size of the space.
 * \return The allocated space.
 */
TVM_DLL void* TVMWasmAllocSpace(int size);

/*!
 * \brief Free the space allocated by TVMWasmAllocSpace.
 * \param data The data pointer.
 */
TVM_DLL void TVMWasmFreeSpace(void* data);

/*!
 * \brief Create PackedFunc from a resource handle.
 * \param resource_handle The handle to the resource.
 * \param out The output PackedFunc.
 * \sa TVMWasmPackedCFunc, TVMWasmPackedCFuncFinalizer
3A * \return 0 if success.
 */
TVM_DLL int TVMWasmFuncCreateFromCFunc(void* resource_handle, TVMFunctionHandle* out);

// --- APIs to be implemented by the frontend. ---
/*!
 * \brief Wasm frontend packed function caller.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from fron-end.
 * \return 0 if success, -1 if failure happens, set error via TVMAPISetLastError.
 */
extern int TVMWasmPackedCFunc(TVMValue* args, int* type_codes, int num_args, TVMRetValueHandle ret,
                              void* resource_handle);

/*!
 * \brief Wasm frontend resource finalizer.
 * \param resource_handle The pointer to the external resource.
 */
extern void TVMWasmPackedCFuncFinalizer(void* resource_handle);
}  // extern "C"

void* TVMWasmAllocSpace(int size) {
  int num_count = (size + 7) / 8;
  return new int64_t[num_count];
}

void TVMWasmFreeSpace(void* arr) { delete[] static_cast<int64_t*>(arr); }

int TVMWasmFuncCreateFromCFunc(void* resource_handle, TVMFunctionHandle* out) {
  return TVMFuncCreateFromCFunc(TVMWasmPackedCFunc, resource_handle, TVMWasmPackedCFuncFinalizer,
                                out);
}

namespace tvm {
namespace runtime {

// A special local session that can interact with async
// functions in the JS runtime.
class AsyncLocalSession : public LocalSession {
 public:
  AsyncLocalSession() {}

  PackedFuncHandle GetFunction(const std::string& name) final {
    if (name == "runtime.RPCTimeEvaluator") {
      return get_time_eval_placeholder_.get();
    } else if (auto* fp = tvm::runtime::Registry::Get(name)) {
      // return raw handle because the remote need to explicitly manage it.
      tvm::runtime::TVMRetValue ret;
      ret = *fp;
      TVMValue val;
      int type_code;
      ret.MoveToCHost(&val, &type_code);
      return val.v_handle;
    } else if (auto* fp = tvm::runtime::Registry::Get("__async." + name)) {
      tvm::runtime::TVMRetValue ret;
      ret = *fp;
      TVMValue val;
      int type_code;
      ret.MoveToCHost(&val, &type_code);
      auto* rptr = val.v_handle;
      async_func_set_.insert(rptr);
      return rptr;
    } else {
      return nullptr;
    }
  }

  void FreeHandle(void* handle, int type_code) final {
    if (type_code == kTVMPackedFuncHandle) {
      auto it = async_func_set_.find(handle);
      if (it != async_func_set_.end()) {
        async_func_set_.erase(it);
      }
    }
    if (handle != get_time_eval_placeholder_.get()) {
      LocalSession::FreeHandle(handle, type_code);
    }
  }

  void AsyncCallFunc(PackedFuncHandle func, const TVMValue* arg_values, const int* arg_type_codes,
                     int num_args, FAsyncCallback callback) final {
    auto it = async_func_set_.find(func);
    if (it != async_func_set_.end()) {
      PackedFunc packed_callback([callback, this](TVMArgs args, TVMRetValue*) {
        int code = args[0];
        TVMRetValue rv;
        rv = args[1];
        if (code == static_cast<int>(RPCCode::kReturn)) {
          this->EncodeReturn(std::move(rv), [&](TVMArgs encoded_args) {
            callback(RPCCode::kReturn, encoded_args);
          });
        } else {
          // for exception, we can pass through as since this is just normal encoding.
          ICHECK_EQ(code, static_cast<int>(RPCCode::kException));
          callback(RPCCode::kException, args);
        }
      });

      TVMRetValue temp;
      std::vector<TVMValue> values(arg_values, arg_values + num_args);
      std::vector<int> type_codes(arg_type_codes, arg_type_codes + num_args);
      values.emplace_back(TVMValue());
      type_codes.emplace_back(0);

      TVMArgsSetter setter(&values[0], &type_codes[0]);
      // pass the callback as the last argument.
      setter(num_args, packed_callback);

      auto* pf = static_cast<PackedFuncObj*>(func);
      pf->CallPacked(TVMArgs(values.data(), type_codes.data(), num_args + 1), &temp);
    } else if (func == get_time_eval_placeholder_.get()) {
      // special handle time evaluator.
      try {
        TVMArgs args(arg_values, arg_type_codes, num_args);
        PackedFunc retfunc = this->GetTimeEvaluator(args[0], args[1], args[2], args[3], args[4],
                                                    args[5], args[6], args[7], args[8], args[9]);
        TVMRetValue rv;
        rv = retfunc;
        this->EncodeReturn(std::move(rv), [&](TVMArgs encoded_args) {
          // mark as async.
          async_func_set_.insert(encoded_args.values[1].v_handle);
          callback(RPCCode::kReturn, encoded_args);
        });
      } catch (const std::runtime_error& e) {
        this->SendException(callback, e.what());
      }
    } else {
      LocalSession::AsyncCallFunc(func, arg_values, arg_type_codes, num_args, callback);
    }
  }

  void AsyncCopyToRemote(void* local_from_bytes, DLTensor* remote_to, uint64_t nbytes,
                         FAsyncCallback on_complete) final {
    try {
      DLTensor local_from;
      local_from.data = local_from_bytes;
      local_from.device = Device{kDLCPU, 0};
      local_from.ndim = remote_to->ndim;
      local_from.shape = remote_to->shape;
      local_from.dtype = remote_to->dtype;
      local_from.strides = nullptr;
      local_from.byte_offset = 0;
      this->GetDeviceAPI(remote_to->device)->CopyDataFromTo(&local_from, remote_to, nullptr);
      this->AsyncStreamWait(remote_to->device, nullptr, on_complete);
    } catch (const std::runtime_error& e) {
      this->SendException(on_complete, e.what());
    }
  }

  void AsyncCopyFromRemote(DLTensor* remote_from, void* local_to_bytes, uint64_t nbytes,
                           FAsyncCallback on_complete) final {
    try {
      DLTensor local_to;
      local_to.data = local_to_bytes;
      local_to.device = Device{kDLCPU, 0};
      local_to.ndim = remote_from->ndim;
      local_to.shape = remote_from->shape;
      local_to.dtype = remote_from->dtype;
      local_to.strides = nullptr;
      local_to.byte_offset = 0;
      this->GetDeviceAPI(remote_from->device)->CopyDataFromTo(remote_from, &local_to, nullptr);
      this->AsyncStreamWait(remote_from->device, nullptr, on_complete);
    } catch (const std::runtime_error& e) {
      this->SendException(on_complete, e.what());
    }
  }

  void AsyncStreamWait(Device dev, TVMStreamHandle stream, FAsyncCallback on_complete) final {
    if (dev.device_type == kDLCPU) {
      TVMValue value;
      int32_t tcode = kTVMNullptr;
      value.v_handle = nullptr;
      on_complete(RPCCode::kReturn, TVMArgs(&value, &tcode, 1));
    } else {
      CHECK(dev.device_type == static_cast<DLDeviceType>(kDLWebGPU));
      if (async_wait_ == nullptr) {
        async_wait_ = tvm::runtime::Registry::Get("__async.wasm.WebGPUWaitForTasks");
      }
      CHECK(async_wait_ != nullptr);
      PackedFunc packed_callback([on_complete](TVMArgs args, TVMRetValue*) {
        int code = args[0];
        on_complete(static_cast<RPCCode>(code),
                    TVMArgs(args.values + 1, args.type_codes + 1, args.size() - 1));
      });
      (*async_wait_)(packed_callback);
    }
  }

  bool IsAsync() const final { return true; }

 private:
  std::unordered_set<void*> async_func_set_;
  std::unique_ptr<PackedFunc> get_time_eval_placeholder_ = std::make_unique<PackedFunc>();
  const PackedFunc* async_wait_{nullptr};

  // time evaluator
  PackedFunc GetTimeEvaluator(Optional<Module> opt_mod, std::string name, int device_type,
                              int device_id, int number, int repeat, int min_repeat_ms,
                              int limit_zero_time_iterations, int cooldown_interval_ms,
                              int repeats_to_cooldown) {
    Device dev;
    dev.device_type = static_cast<DLDeviceType>(device_type);
    dev.device_id = device_id;

    if (opt_mod.defined()) {
      Module m = opt_mod.value();
      std::string tkey = m->type_key();
      return WrapWasmTimeEvaluator(m.GetFunction(name, false), dev, number, repeat, min_repeat_ms,
                                   limit_zero_time_iterations, cooldown_interval_ms,
                                   repeats_to_cooldown);
    } else {
      auto* pf = runtime::Registry::Get(name);
      CHECK(pf != nullptr) << "Cannot find " << name << " in the global function";
      return WrapWasmTimeEvaluator(*pf, dev, number, repeat, min_repeat_ms,
                                   limit_zero_time_iterations, cooldown_interval_ms,
                                   repeats_to_cooldown);
    }
  }

  // time evaluator
  PackedFunc WrapWasmTimeEvaluator(PackedFunc pf, Device dev, int number, int repeat,
                                   int min_repeat_ms, int limit_zero_time_iterations,
                                   int cooldown_interval_ms, int repeats_to_cooldown) {
    auto ftimer = [pf, dev, number, repeat, min_repeat_ms, limit_zero_time_iterations,
                   cooldown_interval_ms, repeats_to_cooldown](TVMArgs args, TVMRetValue* rv) {
      // the function is a async function.
      PackedFunc on_complete = args[args.size() - 1];
      // keep argument alive in finvoke so that they
      // can be used throughout the async benchmark
      std::vector<TVMValue> values(args.values, args.values + args.size() - 1);
      std::vector<int> type_codes(args.type_codes, args.type_codes + args.size() - 1);

      auto finvoke = [pf, values, type_codes](int n) {
        TVMRetValue temp;
        TVMArgs invoke_args(values.data(), type_codes.data(), values.size());
        for (int i = 0; i < n; ++i) {
          pf.CallPacked(invoke_args, &temp);
        }
      };
      auto* time_exec = runtime::Registry::Get("__async.wasm.TimeExecution");
      CHECK(time_exec != nullptr) << "Cannot find wasm.GetTimer in the global function";
      (*time_exec)(TypedPackedFunc<void(int)>(finvoke), dev, number, repeat, min_repeat_ms,
                   limit_zero_time_iterations, cooldown_interval_ms, repeats_to_cooldown,
                   /*cache_flush_bytes=*/0, on_complete);
    };
    return PackedFunc(ftimer);
  }
};

TVM_REGISTER_GLOBAL("wasm.LocalSession").set_body_typed([]() {
  return CreateRPCSessionModule(std::make_shared<AsyncLocalSession>());
});

}  // namespace runtime
}  // namespace tvm
