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
 * \file src/ffi/function.cc
 * \brief Function call registry and safecall context
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <unordered_map>

namespace tvm {
namespace ffi {

class SafeCallContext {
 public:
  void SetLastError(const TVMFFIAny* error_view) {
    last_error_ = Any(AnyView::CopyFromTVMFFIAny(error_view[0]));
    // turn string into formal error.
    if (Optional<String> opt_str = last_error_.as<String>()) {
      last_error_ = ::tvm::ffi::Error("RuntimeError", *opt_str, "");
    }
  }

  void MoveFromLastError(TVMFFIAny* result) { last_error_.MoveToTVMFFIAny(result); }

  static SafeCallContext* ThreadLocal() {
    static thread_local SafeCallContext ctx;
    return &ctx;
  }

 private:
  Any last_error_;
};

/*!
 * \brief Global function table.
 *

 * \note We do not use mutex to guard updating of GlobalFunctionTable
 *
 * The assumption is that updating of GlobalFunctionTable will be done
 * in the main thread during initialization or loading, or
 * explicitly locked from the caller.
 *
 * Then the followup code will leverage the information
 */
class GlobalFunctionTable {
 public:
  void Update(const String& name, Function func, bool can_override) {
    if (table_.count(name)) {
      if (!can_override) {
        TVM_FFI_THROW(RuntimeError) << "Global Function `" << name << "` is already registered";
      }
    }
    table_[name] = new Function(func);
  }

  bool Remove(const String& name) {
    auto it = table_.find(name);
    if (it == table_.end()) return false;
    table_.erase(it);
    return true;
  }

  const Function* Get(const String& name) {
    auto it = table_.find(name);
    if (it == table_.end()) return nullptr;
    return it->second;
  }

  Array<String> ListNames() const {
    Array<String> names;
    names.reserve(table_.size());
    for (const auto& kv : table_) {
      names.push_back(kv.first);
    }
    return names;
  }

  static GlobalFunctionTable* Global() {
    // We deliberately create a new instance via raw new
    // This is because GlobalFunctionTable can contain callbacks into
    // the host language (Python) and the resource can become invalid
    // indeterministic order of destruction and forking.
    // The resources will only be recycled during program exit.
    static GlobalFunctionTable* inst = new GlobalFunctionTable();
    return inst;
  }

 private:
  // deliberately track function pointer without recycling
  // to avoid
  std::unordered_map<String, Function*> table_;
};
}  // namespace ffi
}  // namespace tvm

int TVMFFIFuncCreate(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self),
                     TVMFFIObjectHandle* out) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  Function func = Function::FromExternC(self, safe_call, deleter);
  *out = details::ObjectUnsafe::MoveTVMFFIObjectPtrFromObjectRef(&func);
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIFuncSetGlobal(const char* name, TVMFFIObjectHandle f, int override) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  GlobalFunctionTable::Global()->Update(name, GetRef<Function>(static_cast<FunctionObj*>(f)),
                                        override != 0);
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIFuncGetGlobal(const char* name, TVMFFIObjectHandle* out) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  const Function* fp = GlobalFunctionTable::Global()->Get(name);
  if (fp != nullptr) {
    Function func(*fp);
    *out = details::ObjectUnsafe::MoveTVMFFIObjectPtrFromObjectRef(&func);
  } else {
    *out = nullptr;
  }
  TVM_FFI_SAFE_CALL_END();
}

void TVMFFISetLastError(const TVMFFIAny* error_view) {
  tvm::ffi::SafeCallContext::ThreadLocal()->SetLastError(error_view);
}

void TVMFFIMoveFromLastError(TVMFFIAny* result) {
  tvm::ffi::SafeCallContext::ThreadLocal()->MoveFromLastError(result);
}

TVM_FFI_REGISTER_GLOBAL("tvm_ffi.GlobalFunctionRemove")
    .set_body_typed([](const tvm::ffi::String& name) -> bool {
      return tvm::ffi::GlobalFunctionTable::Global()->Remove(name);
    });

TVM_FFI_REGISTER_GLOBAL("tvm_ffi.GlobalFunctionListNames").set_body_typed([]() {
  return tvm::ffi::GlobalFunctionTable::Global()->ListNames();
});
