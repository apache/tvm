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
 * \file library_module.h
 * \brief Module that builds from a libary of symbols.
 */
#ifndef TVM_FFI_EXTRA_MODULE_INTERNAL_H_
#define TVM_FFI_EXTRA_MODULE_INTERNAL_H_

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>

#include <mutex>

namespace tvm {
namespace ffi {

/*!
 * \brief Library is the common interface
 *  for storing data in the form of shared libaries.
 *
 * \sa src/ffi/extra/dso_library.cc
 * \sa src/ffi/extra/system_library.cc
 */
class Library : public Object {
 public:
  // destructor.
  virtual ~Library() {}
  /*!
   * \brief Get the symbol address for a given name.
   * \param name The name of the symbol.
   * \return The symbol.
   */
  virtual void* GetSymbol(const char* name) = 0;
  // NOTE: we do not explicitly create an type index and type_key here for libary.
  // This is because we do not need dynamic type downcasting and only need to use the refcounting
};

struct ModuleObj::InternalUnsafe {
  static Array<Any>* GetImports(ModuleObj* module) { return &(module->imports_); }

  static void* GetFunctionFromImports(ModuleObj* module, const char* name) {
    // backend implementation for TVMFFIEnvModLookupFromImports
    static std::mutex mutex_;
    std::lock_guard<std::mutex> lock(mutex_);
    String s_name(name);
    auto it = module->import_lookup_cache_.find(s_name);
    if (it != module->import_lookup_cache_.end()) {
      return const_cast<FunctionObj*>((*it).second.operator->());
    }

    auto opt_func = [&]() -> std::optional<Function> {
      for (const Any& import : module->imports_) {
        if (auto opt_func = import.cast<Module>()->GetFunction(s_name, true)) {
          return *opt_func;
        }
      }
      // try global at last
      return tvm::ffi::Function::GetGlobal(s_name);
    }();
    if (!opt_func.has_value()) {
      TVM_FFI_THROW(RuntimeError) << "Cannot find function " << name
                                  << " in the imported modules or global registry.";
    }
    module->import_lookup_cache_.Set(s_name, *opt_func);
    return const_cast<FunctionObj*>((*opt_func).operator->());
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ModuleObj>().def_ro("imports_", &ModuleObj::imports_);
  }
};

/*!
 * \brief Create a library module from a given library.
 *
 * \param lib The library.
 *
 * \return The corresponding loaded module.
 */
Module CreateLibraryModule(ObjectPtr<Library> lib);

}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_EXTRA_MODULE_INTERNAL_H_
