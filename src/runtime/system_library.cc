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
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/registry.h>

#include <mutex>

#include "library_module.h"

namespace tvm {
namespace runtime {

class SystemLibSymbolRegistry {
 public:
  void RegisterSymbol(const std::string& name, void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = symbol_table_.find(name);
    if (it != symbol_table_.end() && ptr != it->second) {
      LOG(WARNING) << "SystemLib symbol " << name << " get overriden to a different address " << ptr
                   << "->" << it->second;
    }
    symbol_table_[name] = ptr;
  }

  void* GetSymbol(const char* name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = symbol_table_.find(name);
    if (it != symbol_table_.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  static SystemLibSymbolRegistry* Global() {
    static SystemLibSymbolRegistry* inst = new SystemLibSymbolRegistry();
    return inst;
  }

 private:
  // Internal mutex
  std::mutex mutex_;
  // Internal symbol table
  std::unordered_map<std::string, void*> symbol_table_;
};

class SystemLibrary : public Library {
 public:
  explicit SystemLibrary(const std::string& symbol_prefix) : symbol_prefix_(symbol_prefix) {}

  void* GetSymbol(const char* name) {
    if (symbol_prefix_.length() != 0) {
      std::string name_with_prefix = symbol_prefix_ + name;
      void* symbol = reg_->GetSymbol(name_with_prefix.c_str());
      if (symbol != nullptr) return symbol;
    }
    return reg_->GetSymbol(name);
  }

 private:
  SystemLibSymbolRegistry* reg_ = SystemLibSymbolRegistry::Global();
  std::string symbol_prefix_;
};

class SystemLibModuleRegistry {
 public:
  runtime::Module GetOrCreateModule(std::string symbol_prefix) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = lib_map_.find(symbol_prefix);
    if (it != lib_map_.end()) {
      return it->second;
    } else {
      auto mod = CreateModuleFromLibrary(make_object<SystemLibrary>(symbol_prefix));
      lib_map_[symbol_prefix] = mod;
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
  // we need to make sure each lib map have an unique
  // copy through out the entire lifetime of the process
  // so the cached PackedFunc in the system do not get out dated.
  std::unordered_map<std::string, runtime::Module> lib_map_;
};

TVM_REGISTER_GLOBAL("runtime.SystemLib").set_body([](TVMArgs args, TVMRetValue* rv) {
  std::string symbol_prefix = "";
  if (args.size() != 0) {
    symbol_prefix = args[0].operator std::string();
  }
  *rv = SystemLibModuleRegistry::Global()->GetOrCreateModule(symbol_prefix);
});
}  // namespace runtime
}  // namespace tvm

int TVMBackendRegisterSystemLibSymbol(const char* name, void* ptr) {
  tvm::runtime::SystemLibSymbolRegistry::Global()->RegisterSymbol(name, ptr);
  return 0;
}
