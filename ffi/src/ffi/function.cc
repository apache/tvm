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
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {

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
  // Note: this class is hidden from the public API, so we just
  // use it as a private class as ObjectRef
  class Entry : public Object, public TVMFFIMethodInfo {
   public:
    String name_data;
    String doc_data;
    String type_schema_data;
    ffi::Function func_data;

    explicit Entry(const TVMFFIMethodInfo* method_info) {
      // make copy of the metadata
      name_data = String(method_info->name.data, method_info->name.size);
      doc_data = String(method_info->doc.data, method_info->doc.size);
      type_schema_data = String(method_info->type_schema.data, method_info->type_schema.size);
      func_data = AnyView::CopyFromTVMFFIAny(method_info->method).cast<ffi::Function>();
      this->SyncMethodInfo(method_info->flags);
      // no need to update method pointer as it would remain the same as func and we retained
    }
    explicit Entry(String name, ffi::Function func) : name_data(name), func_data(func) {
      this->SyncMethodInfo(kTVMFFIFieldFlagBitMaskIsStaticMethod);
    }

   private:
    void SyncMethodInfo(int64_t flags) {
      this->flags = flags;
      this->name = TVMFFIByteArray{name_data.data(), name_data.size()};
      this->doc = TVMFFIByteArray{doc_data.data(), doc_data.size()};
      this->type_schema = TVMFFIByteArray{type_schema_data.data(), type_schema_data.size()};
    }
  };

  void Update(const String& name, Function func, bool can_override) {
    if (table_.count(name)) {
      if (!can_override) {
        TVM_FFI_THROW(RuntimeError) << "Global Function `" << name << "` is already registered";
      }
    }
    table_.Set(name, ObjectRef(make_object<Entry>(name, func)));
  }

  void Update(const TVMFFIMethodInfo* method_info, bool can_override) {
    String name(method_info->name.data, method_info->name.size);
    if (table_.count(name)) {
      if (!can_override) {
        TVM_FFI_LOG_AND_THROW(RuntimeError)
            << "Global Function `" << name << "` is already registered, possible causes:\n"
            << "- Two GlobalDef().def registrations for the same function \n"
            << "Please remove the duplicate registration.";
      }
    }
    table_.Set(name, ObjectRef(make_object<Entry>(method_info)));
  }

  bool Remove(const String& name) {
    auto it = table_.find(name);
    if (it == table_.end()) return false;
    table_.erase(name);
    return true;
  }

  const Entry* Get(const String& name) {
    auto it = table_.find(name);
    if (it == table_.end()) return nullptr;
    const Object* obj = (*it).second.cast<const Object*>();
    return static_cast<const Entry*>(obj);
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
  Map<String, Any> table_;
};
}  // namespace ffi
}  // namespace tvm

int TVMFFIFunctionCreate(void* self, TVMFFISafeCallType safe_call, void (*deleter)(void* self),
                         TVMFFIObjectHandle* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::Function func = tvm::ffi::Function::FromExternC(self, safe_call, deleter);
  *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(func));
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIAnyViewToOwnedAny(const TVMFFIAny* any_view, TVMFFIAny* out) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::Any result(*reinterpret_cast<const tvm::ffi::AnyView*>(any_view));
  *out = tvm::ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(result));
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIFunctionSetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle f, int override) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  String name_str(name->data, name->size);
  GlobalFunctionTable::Global()->Update(name_str, GetRef<Function>(static_cast<FunctionObj*>(f)),
                                        override != 0);
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIFunctionSetGlobalFromMethodInfo(const TVMFFIMethodInfo* method_info, int override) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  GlobalFunctionTable::Global()->Update(method_info, override != 0);
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIFunctionGetGlobal(const TVMFFIByteArray* name, TVMFFIObjectHandle* out) {
  using namespace tvm::ffi;
  TVM_FFI_SAFE_CALL_BEGIN();
  String name_str(name->data, name->size);
  const GlobalFunctionTable::Entry* fp = GlobalFunctionTable::Global()->Get(name_str);
  if (fp != nullptr) {
    tvm::ffi::Function func(fp->func_data);
    *out = tvm::ffi::details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(func));
  } else {
    *out = nullptr;
  }
  TVM_FFI_SAFE_CALL_END();
}

int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args, int32_t num_args,
                       TVMFFIAny* result) {
  using namespace tvm::ffi;
#ifdef _MSC_VER
  // Avoid tail call optimization
  // in MSVC many cases python symbols are hidden, so we need this function symbol
  // to be in the call frame to reliably detect the ffi boundary
  volatile int ret = reinterpret_cast<FunctionObj*>(func)->safe_call(func, args, num_args, result);
  return ret;
#else
  // NOTE: this is a tail call
  return reinterpret_cast<FunctionObj*>(func)->safe_call(func, args, num_args, result);
#endif
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.FunctionRemoveGlobal",
           [](const tvm::ffi::String& name) -> bool {
             return tvm::ffi::GlobalFunctionTable::Global()->Remove(name);
           })
      .def("ffi.FunctionListGlobalNamesFunctor",
           []() {
             // NOTE: we return functor instead of array
             // so list global function names do not need to depend on array
             // this is because list global function names usually is a core api that happens
             // before array ffi functions are available.
             tvm::ffi::Array<tvm::ffi::String> names =
                 tvm::ffi::GlobalFunctionTable::Global()->ListNames();
             auto return_functor = [names](int64_t i) -> tvm::ffi::Any {
               if (i < 0) {
                 return names.size();
               } else {
                 return names[i];
               }
             };
             return tvm::ffi::Function::FromTyped(return_functor);
           })
      .def("ffi.String", [](tvm::ffi::String val) -> tvm::ffi::String { return val; })
      .def("ffi.Bytes", [](tvm::ffi::Bytes val) -> tvm::ffi::Bytes { return val; });
});
