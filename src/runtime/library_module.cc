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
 * \file module_util.cc
 * \brief Utilities for module.
 */
#include "library_module.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

// Library module that exposes symbols from a library.
class LibraryModuleNode final : public ModuleNode {
 public:
  explicit LibraryModuleNode(ObjectPtr<Library> lib) : lib_(lib) {}

  const char* type_key() const final { return "library"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    TVMBackendPackedCFunc faddr;
    if (name == runtime::symbol::tvm_module_main) {
      const char* entry_name =
          reinterpret_cast<const char*>(lib_->GetSymbol(runtime::symbol::tvm_module_main));
      CHECK(entry_name != nullptr)
          << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(lib_->GetSymbol(entry_name));
    } else {
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(lib_->GetSymbol(name.c_str()));
    }
    if (faddr == nullptr) return PackedFunc();
    return WrapPackedFunc(faddr, sptr_to_self);
  }

 private:
  ObjectPtr<Library> lib_;
};

/*!
 * \brief Helper classes to get into internal of a module.
 */
class ModuleInternal {
 public:
  // Get mutable reference of imports.
  static std::vector<Module>* GetImportsAddr(ModuleNode* node) { return &(node->imports_); }
};

PackedFunc WrapPackedFunc(TVMBackendPackedCFunc faddr, const ObjectPtr<Object>& sptr_to_self) {
  return PackedFunc([faddr, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
    TVMValue ret_value;
    int ret_type_code = kTVMNullptr;
    int ret = (*faddr)(const_cast<TVMValue*>(args.values), const_cast<int*>(args.type_codes),
                       args.num_args, &ret_value, &ret_type_code, NULL);
    CHECK_EQ(ret, 0) << TVMGetLastError();
    if (ret_type_code != kTVMNullptr) {
      *rv = TVMRetValue::MoveFromCHost(ret_value, ret_type_code);
    }
  });
}

void InitContextFunctions(std::function<void*(const char*)> fgetsymbol) {
#define TVM_INIT_CONTEXT_FUNC(FuncName)                                                \
  if (auto* fp = reinterpret_cast<decltype(&FuncName)*>(fgetsymbol("__" #FuncName))) { \
    *fp = FuncName;                                                                    \
  }
  // Initialize the functions
  TVM_INIT_CONTEXT_FUNC(TVMFuncCall);
  TVM_INIT_CONTEXT_FUNC(TVMAPISetLastError);
  TVM_INIT_CONTEXT_FUNC(TVMBackendGetFuncFromEnv);
  TVM_INIT_CONTEXT_FUNC(TVMBackendAllocWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendFreeWorkspace);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelLaunch);
  TVM_INIT_CONTEXT_FUNC(TVMBackendParallelBarrier);

#undef TVM_INIT_CONTEXT_FUNC
}

/*!
 * \brief Load and append module blob to module list
 * \param mblob The module blob.
 * \param lib The library.
 *
 * \return Root Module.
 */
runtime::Module ProcessModuleBlob(const char* mblob, ObjectPtr<Library> lib) {
  CHECK(mblob != nullptr);
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = mblob[i];
    nbytes |= (c & 0xffUL) << (i * 8);
  }
  dmlc::MemoryFixedSizeStream fs(const_cast<char*>(mblob + sizeof(nbytes)),
                                 static_cast<size_t>(nbytes));
  dmlc::Stream* stream = &fs;
  uint64_t size;
  CHECK(stream->Read(&size));
  std::vector<Module> modules;
  std::vector<uint64_t> import_tree_row_ptr;
  std::vector<uint64_t> import_tree_child_indices;
  for (uint64_t i = 0; i < size; ++i) {
    std::string tkey;
    CHECK(stream->Read(&tkey));
    // Currently, _lib is for DSOModule, but we
    // don't have loadbinary function for it currently
    if (tkey == "_lib") {
      auto dso_module = Module(make_object<LibraryModuleNode>(lib));
      modules.emplace_back(dso_module);
    } else if (tkey == "_import_tree") {
      CHECK(stream->Read(&import_tree_row_ptr));
      CHECK(stream->Read(&import_tree_child_indices));
    } else {
      std::string fkey = "runtime.module.loadbinary_" + tkey;
      const PackedFunc* f = Registry::Get(fkey);
      CHECK(f != nullptr) << "Loader of " << tkey << "(" << fkey << ") is not presented.";
      Module m = (*f)(static_cast<void*>(stream));
      modules.emplace_back(m);
    }
  }
  // if we are using old dll, we don't have import tree
  // so that we can't reconstruct module relationship using import tree
  if (import_tree_row_ptr.empty()) {
    auto n = make_object<LibraryModuleNode>(lib);
    auto module_import_addr = ModuleInternal::GetImportsAddr(n.operator->());
    for (const auto& m : modules) {
      module_import_addr->emplace_back(m);
    }
    return Module(n);
  } else {
    for (size_t i = 0; i < modules.size(); ++i) {
      for (size_t j = import_tree_row_ptr[i]; j < import_tree_row_ptr[i + 1]; ++j) {
        auto module_import_addr = ModuleInternal::GetImportsAddr(modules[i].operator->());
        auto child_index = import_tree_child_indices[j];
        CHECK(child_index < modules.size());
        module_import_addr->emplace_back(modules[child_index]);
      }
    }
  }
  CHECK(!modules.empty());
  // invariance: root module is always at location 0.
  // The module order is collected via DFS
  return modules[0];
}

Module CreateModuleFromLibrary(ObjectPtr<Library> lib) {
  InitContextFunctions([lib](const char* fname) { return lib->GetSymbol(fname); });
  auto n = make_object<LibraryModuleNode>(lib);
  // Load the imported modules
  const char* dev_mblob =
      reinterpret_cast<const char*>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));
  Module root_mod;
  if (dev_mblob != nullptr) {
    root_mod = ProcessModuleBlob(dev_mblob, lib);
  } else {
    // Only have one single DSO Module
    root_mod = Module(n);
  }

  // allow lookup of symbol from root (so all symbols are visible).
  if (auto* ctx_addr = reinterpret_cast<void**>(lib->GetSymbol(runtime::symbol::tvm_module_ctx))) {
    *ctx_addr = root_mod.operator->();
  }

  return root_mod;
}
}  // namespace runtime
}  // namespace tvm
