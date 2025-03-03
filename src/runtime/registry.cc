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
 * \file registry.cc
 * \brief The global registry of packed function.
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "runtime_base.h"

namespace tvm {
namespace runtime {

struct Registry::Manager {
  // map storing the functions.
  // We deliberately used raw pointer.
  // This is because PackedFunc can contain callbacks into the host language (Python) and the
  // resource can become invalid because of indeterministic order of destruction and forking.
  // The resources will only be recycled during program exit.
  std::unordered_map<String, Registry*> fmap;
  // mutex
  std::mutex mutex;

  Manager() {}

  static Manager* Global() {
    // We deliberately leak the Manager instance, to avoid leak sanitizers
    // complaining about the entries in Manager::fmap being leaked at program
    // exit.
    static Manager* inst = new Manager();
    return inst;
  }
};

Registry& Registry::set_body(PackedFunc f) {  // NOLINT(*)
  func_ = f;
  return *this;
}

Registry& Registry::Register(const String& name, bool can_override) {  // NOLINT(*)
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  if (m->fmap.count(name)) {
    ICHECK(can_override) << "Global PackedFunc " << name << " is already registered";
  }

  Registry* r = new Registry();
  r->name_ = name;
  m->fmap[name] = r;
  return *r;
}

bool Registry::Remove(const String& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return false;
  m->fmap.erase(it);
  return true;
}

const PackedFunc* Registry::Get(const String& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return nullptr;
  return &(it->second->func_);
}

std::vector<String> Registry::ListNames() {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  std::vector<String> keys;
  keys.reserve(m->fmap.size());
  for (const auto& kv : m->fmap) {
    keys.push_back(kv.first);
  }
  return keys;
}

/*!
 * \brief Execution environment specific API registry.
 *
 *  This registry stores C API function pointers about
 *  execution environment(e.g. python) specific API function that
 *  we need for specific low-level handling(e.g. signal checking).
 *
 *  We only stores the C API function when absolutely necessary (e.g. when signal handler
 *  cannot trap back into python). Always consider use the PackedFunc FFI when possible
 *  in other cases.
 */
class EnvCAPIRegistry {
 public:
  /*!
   * \brief Callback to check if signals have been sent to the process and
   *        if so invoke the registered signal handler in the frontend environment.
   *
   *  When running TVM in another language (Python), the signal handler
   *  may not be immediately executed, but instead the signal is marked
   *  in the interpreter state (to ensure non-blocking of the signal handler).
   *
   * \return 0 if no error happens, -1 if error happens.
   */
  typedef int (*F_PyErr_CheckSignals)();

  /*! \brief Callback to increment/decrement the python ref count */
  typedef void (*F_Py_IncDefRef)(void*);

  // NOTE: the following functions are only registered in a python
  // environment.
  /*!
   * \brief PyErr_CheckSignal function
   */
  F_PyErr_CheckSignals pyerr_check_signals = nullptr;

  /*!
   * \brief Py_IncRef function
   */
  F_Py_IncDefRef py_inc_ref = nullptr;

  /*!
   * \brief Py_IncRef function
   */
  F_Py_IncDefRef py_dec_ref = nullptr;

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
    } else if (symbol_name == "Py_IncRef") {
      Update(symbol_name, &py_inc_ref, fptr);
    } else if (symbol_name == "Py_DecRef") {
      Update(symbol_name, &py_dec_ref, fptr);
    } else if (symbol_name == "PyGILState_Ensure") {
      Update(symbol_name, &py_gil_state_ensure, fptr);
    } else if (symbol_name == "PyGILState_Release") {
      Update(symbol_name, &py_gil_state_release, fptr);
    } else {
      LOG(FATAL) << "Unknown env API " << symbol_name;
    }
  }

  // implementation of tvm::runtime::EnvCheckSignals
  void CheckSignals() {
    // check python signal to see if there are exception raised
    if (pyerr_check_signals != nullptr) {
      // The C++ env comes without gil, so we need to grab gil here
      WithGIL context(this);
      if ((*pyerr_check_signals)() != 0) {
        // The error will let FFI know that the frontend environment
        // already set an error.
        throw EnvErrorAlreadySet("");
      }
    }
  }

  void IncRef(void* python_obj) {
    WithGIL context(this);
    ICHECK(py_inc_ref) << "Attempted to call Py_IncRef through EnvCAPIRegistry, "
                       << "but Py_IncRef wasn't registered";
    (*py_inc_ref)(python_obj);
  }

  void DecRef(void* python_obj) {
    WithGIL context(this);
    ICHECK(py_dec_ref) << "Attempted to call Py_DefRef through EnvCAPIRegistry, "
                       << "but Py_DefRef wasn't registered";
    (*py_dec_ref)(python_obj);
  }

 private:
  // update the internal API table
  template <typename FType>
  void Update(const String& symbol_name, FType* target, void* ptr) {
    FType ptr_casted = reinterpret_cast<FType>(ptr);
    if (target[0] != nullptr && target[0] != ptr_casted) {
      LOG(WARNING) << "tvm.runtime.RegisterEnvCAPI overrides an existing function " << symbol_name;
    }
    target[0] = ptr_casted;
  }

  struct WithGIL {
    explicit WithGIL(EnvCAPIRegistry* self) : self(self) {
      ICHECK(self->py_gil_state_ensure) << "Attempted to acquire GIL through EnvCAPIRegistry, "
                                        << "but PyGILState_Ensure wasn't registered";
      ICHECK(self->py_gil_state_release) << "Attempted to acquire GIL through EnvCAPIRegistry, "
                                         << "but PyGILState_Release wasn't registered";
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

void EnvCheckSignals() { EnvCAPIRegistry::Global()->CheckSignals(); }

WrappedPythonObject::WrappedPythonObject(void* python_obj) : python_obj_(python_obj) {
  if (python_obj_) {
    EnvCAPIRegistry::Global()->IncRef(python_obj_);
  }
}

WrappedPythonObject::~WrappedPythonObject() {
  if (python_obj_) {
    EnvCAPIRegistry::Global()->DecRef(python_obj_);
  }
}

WrappedPythonObject::WrappedPythonObject(WrappedPythonObject&& other) : python_obj_(nullptr) {
  std::swap(python_obj_, other.python_obj_);
}
WrappedPythonObject& WrappedPythonObject::operator=(WrappedPythonObject&& other) {
  std::swap(python_obj_, other.python_obj_);
  return *this;
}

WrappedPythonObject::WrappedPythonObject(const WrappedPythonObject& other)
    : WrappedPythonObject(other.python_obj_) {}
WrappedPythonObject& WrappedPythonObject::operator=(const WrappedPythonObject& other) {
  return *this = WrappedPythonObject(other);
}
WrappedPythonObject& WrappedPythonObject::operator=(std::nullptr_t) {
  return *this = WrappedPythonObject(nullptr);
}

}  // namespace runtime
}  // namespace tvm

/*! \brief entry to easily hold returning information */
struct TVMFuncThreadLocalEntry {
  /*! \brief result holder for returning strings */
  std::vector<tvm::runtime::String> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char*> ret_vec_charp;
};

/*! \brief Thread local store that can be used to hold return values. */
typedef dmlc::ThreadLocalStore<TVMFuncThreadLocalEntry> TVMFuncThreadLocalStore;

int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f, int override) {
  API_BEGIN();
  using tvm::runtime::GetRef;
  using tvm::runtime::PackedFunc;
  using tvm::runtime::PackedFuncObj;
  tvm::runtime::Registry::Register(name, override != 0)
      .set_body(GetRef<PackedFunc>(static_cast<PackedFuncObj*>(f)));
  API_END();
}

int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {
  API_BEGIN();
  const tvm::runtime::PackedFunc* fp = tvm::runtime::Registry::Get(name);
  if (fp != nullptr) {
    tvm::runtime::TVMRetValue ret;
    ret = *fp;
    TVMValue val;
    int type_code;
    ret.MoveToCHost(&val, &type_code);
    *out = val.v_handle;
  } else {
    *out = nullptr;
  }
  API_END();
}

int TVMFuncListGlobalNames(int* out_size, const char*** out_array) {
  API_BEGIN();
  TVMFuncThreadLocalEntry* ret = TVMFuncThreadLocalStore::Get();
  ret->ret_vec_str = tvm::runtime::Registry::ListNames();
  ret->ret_vec_charp.clear();
  for (size_t i = 0; i < ret->ret_vec_str.size(); ++i) {
    ret->ret_vec_charp.push_back(ret->ret_vec_str[i].c_str());
  }
  *out_array = dmlc::BeginPtr(ret->ret_vec_charp);
  *out_size = static_cast<int>(ret->ret_vec_str.size());
  API_END();
}

int TVMFuncRemoveGlobal(const char* name) {
  API_BEGIN();
  tvm::runtime::Registry::Remove(name);
  API_END();
}

int TVMBackendRegisterEnvCAPI(const char* name, void* ptr) {
  API_BEGIN();
  tvm::runtime::EnvCAPIRegistry::Global()->Register(name, ptr);
  API_END();
}
