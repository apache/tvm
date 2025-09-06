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
 * \file system_library.cc
 * \brief Create library module that directly get symbol from the system lib.
 */
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <mutex>

#include "module_internal.h"

namespace tvm {
namespace ffi {

class SystemLibSymbolRegistry {
 public:
  void RegisterSymbol(const std::string& name, void* ptr) {
    auto it = symbol_table_.find(name);
    if (it != symbol_table_.end() && ptr != (*it).second) {
      std::cerr << "Warning:SystemLib symbol " << name << " get overriden to a different address "
                << ptr << "->" << (*it).second << std::endl;
    }
    symbol_table_.Set(name, ptr);
  }

  void* GetSymbol(const String& name) {
    auto it = symbol_table_.find(name);
    if (it != symbol_table_.end()) {
      return (*it).second;
    } else {
      return nullptr;
    }
  }

  static SystemLibSymbolRegistry* Global() {
    static SystemLibSymbolRegistry* inst = new SystemLibSymbolRegistry();
    return inst;
  }

 private:
  // Internal symbol table
  Map<String, void*> symbol_table_;
};

class SystemLibrary final : public Library {
 public:
  explicit SystemLibrary(const String& symbol_prefix) : symbol_prefix_(symbol_prefix) {}

  void* GetSymbol(const String& name) final {
    String name_with_prefix = symbol_prefix_ + name;
    return reg_->GetSymbol(name_with_prefix);
  }

  void* GetSymbolWithSymbolPrefix(const String& name) final {
    String name_with_prefix = symbol::tvm_ffi_symbol_prefix + symbol_prefix_ + name;
    return reg_->GetSymbol(name_with_prefix);
  }

 private:
  SystemLibSymbolRegistry* reg_ = SystemLibSymbolRegistry::Global();
  String symbol_prefix_;
};

class SystemLibModuleRegistry {
 public:
  Module GetOrCreateModule(String symbol_prefix) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lib_map_.find(symbol_prefix);
    if (it != lib_map_.end()) {
      return (*it).second;
    } else {
      Module mod = CreateLibraryModule(make_object<SystemLibrary>(symbol_prefix));
      lib_map_.Set(symbol_prefix, mod);
      return mod;
    }
  }

  static SystemLibModuleRegistry* Global() {
    static SystemLibModuleRegistry* inst = new SystemLibModuleRegistry();
    return inst;
  }

 private:
  // Internal mutex
  std::mutex mutex_;
  // maps prefix to the library module
  // we need to make sure each lib map have an unique
  // copy through out the entire lifetime of the process
  Map<String, ffi::Module> lib_map_;
};

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("ffi.SystemLib", [](ffi::PackedArgs args, ffi::Any* rv) {
    String symbol_prefix = "";
    if (args.size() != 0) {
      symbol_prefix = args[0].cast<String>();
    }
    *rv = SystemLibModuleRegistry::Global()->GetOrCreateModule(symbol_prefix);
  });
});
}  // namespace ffi
}  // namespace tvm

int TVMFFIEnvModRegisterSystemLibSymbol(const char* name, void* ptr) {
  tvm::ffi::SystemLibSymbolRegistry::Global()->RegisterSymbol(name, ptr);
  return 0;
}
