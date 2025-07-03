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
#include <tvm/ffi/reflection/reflection.h>
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
        TVM_FFI_THROW(RuntimeError) << "Global Function `" << name << "` is already registered";
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

/*!
 * \brief Execution environment specific API registry.
 *
 *  This registry stores C API function pointers about
 *  execution environment(e.g. python) specific API function that
 *  we need for specific low-level handling(e.g. signal checking).
 *
 *  We only stores the C API function when absolutely necessary (e.g. when signal handler
 *  cannot trap back into python). Always consider use the Function FFI when possible
 *  in other cases.
 */
class EnvCAPIRegistry {
 public:
  /*!
   * \brief Callback to check if signals have been sent to the process and
   *        if so invoke the registered signal handler in the frontend environment.
   *
   *  When running FFI in another language (Python), the signal handler
   *  may not be immediately executed, but instead the signal is marked
   *  in the interpreter state (to ensure non-blocking of the signal handler).
   *
   * \return 0 if no error happens, -1 if error happens.
   */
  typedef int (*F_PyErr_CheckSignals)();

  /*! \brief Callback to increment/decrement the python ref count */
  typedef void (*F_Py_IncDefRef)(void*);

  /*!
   * \brief PyErr_CheckSignal function
   */
  F_PyErr_CheckSignals pyerr_check_signals = nullptr;

  /*!
    \brief PyGILState_Ensure function
   */
  void* (*py_gil_state_ensure)() = nullptr;

  /*!
    \brief PyGILState_Release function
   */
  void (*py_gil_state_release)(void*) = nullptr;

  static EnvCAPIRegistry* Global() {
    static EnvCAPIRegistry* inst = new EnvCAPIRegistry();
    return inst;
  }

  // register environment(e.g. python) specific api functions
  void Register(const String& symbol_name, void* fptr) {
    if (symbol_name == "PyErr_CheckSignals") {
      Update(symbol_name, &pyerr_check_signals, fptr);
    } else if (symbol_name == "PyGILState_Ensure") {
      Update(symbol_name, &py_gil_state_ensure, fptr);
    } else if (symbol_name == "PyGILState_Release") {
      Update(symbol_name, &py_gil_state_release, fptr);
    } else {
      TVM_FFI_THROW(ValueError) << "Unknown env API " + symbol_name;
    }
  }

  int EnvCheckSignals() {
    // check python signal to see if there are exception raised
    if (pyerr_check_signals != nullptr) {
      // The C++ env comes without gil, so we need to grab gil here
      WithGIL context(this);
      if ((*pyerr_check_signals)() != 0) {
        // The error will let FFI know that the frontend environment
        // already set an error.
        return -1;
      }
    }
    return 0;
  }

 private:
  // update the internal API table
  template <typename FType>
  void Update(const String& symbol_name, FType* target, void* ptr) {
    FType ptr_casted = reinterpret_cast<FType>(ptr);
    target[0] = ptr_casted;
  }

  struct WithGIL {
    explicit WithGIL(EnvCAPIRegistry* self) : self(self) {
      TVM_FFI_ICHECK(self->py_gil_state_ensure);
      TVM_FFI_ICHECK(self->py_gil_state_release);
      gil_state = self->py_gil_state_ensure();
    }
    ~WithGIL() {
      if (self && gil_state) {
        self->py_gil_state_release(gil_state);
      }
    }
    WithGIL(const WithGIL&) = delete;
    WithGIL(WithGIL&&) = delete;
    WithGIL& operator=(const WithGIL&) = delete;
    WithGIL& operator=(WithGIL&&) = delete;

    EnvCAPIRegistry* self = nullptr;
    void* gil_state = nullptr;
  };
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
  // NOTE: this is a tail call
  return reinterpret_cast<FunctionObj*>(func)->safe_call(func, args, num_args, result);
}

int TVMFFIEnvCheckSignals() { return tvm::ffi::EnvCAPIRegistry::Global()->EnvCheckSignals(); }

/*!
 * \brief Register a symbol into the from the surrounding env.
 * \param name The name of the symbol.
 * \param symbol The symbol to register.
 * \return 0 when success, nonzero when failure happens
 */
int TVMFFIEnvRegisterCAPI(const TVMFFIByteArray* name, void* symbol) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::String s_name(name->data, name->size);
  tvm::ffi::EnvCAPIRegistry::Global()->Register(s_name, symbol);
  TVM_FFI_SAFE_CALL_END();
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
