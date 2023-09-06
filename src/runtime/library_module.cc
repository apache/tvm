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
  explicit LibraryModuleNode(ObjectPtr<Library> lib, PackedFuncWrapper wrapper)
      : lib_(lib), packed_func_wrapper_(wrapper) {}

  const char* type_key() const final { return "library"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  };

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    TVMBackendPackedCFunc faddr;
    if (name == runtime::symbol::tvm_module_main) {
      const char* entry_name =
          reinterpret_cast<const char*>(lib_->GetSymbol(runtime::symbol::tvm_module_main));
      ICHECK(entry_name != nullptr)
          << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(lib_->GetSymbol(entry_name));
    } else {
      faddr = reinterpret_cast<TVMBackendPackedCFunc>(lib_->GetSymbol(name.c_str()));
    }
    if (faddr == nullptr) return PackedFunc();
    return packed_func_wrapper_(faddr, sptr_to_self);
  }

 private:
  ObjectPtr<Library> lib_;
  PackedFuncWrapper packed_func_wrapper_;
};

PackedFunc WrapPackedFunc(TVMBackendPackedCFunc faddr, const ObjectPtr<Object>& sptr_to_self) {
  return PackedFunc([faddr, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
    TVMValue ret_value;
    int ret_type_code = kTVMNullptr;
    int ret = (*faddr)(const_cast<TVMValue*>(args.values), const_cast<int*>(args.type_codes),
                       args.num_args, &ret_value, &ret_type_code, nullptr);
    // NOTE: important to keep the original error message.
    if (ret != 0) {
      LOG(FATAL) << TVMGetLastError();
    }
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

Module LoadModuleFromBinary(const std::string& type_key, dmlc::Stream* stream) {
  std::string loadkey = "runtime.module.loadbinary_";
  std::string fkey = loadkey + type_key;
  const PackedFunc* f = Registry::Get(fkey);
  if (f == nullptr) {
    std::string loaders = "";
    for (auto reg_name : Registry::ListNames()) {
      std::string name = reg_name;
      if (name.find(loadkey, 0) == 0) {
        if (loaders.size() > 0) {
          loaders += ", ";
        }
        loaders += name.substr(loadkey.size());
      }
    }
    LOG(FATAL) << "Binary was created using {" << type_key
               << "} but a loader of that name is not registered. Available loaders are " << loaders
               << ". Perhaps you need to recompile with this runtime enabled.";
  }

  return (*f)(static_cast<void*>(stream));
}

/*!
 * \brief Load and append module blob to module list
 * \param mblob The module blob.
 * \param lib The library.
 * \param root_module the output root module
 * \param dso_ctx_addr the output dso module
 */
void ProcessModuleBlob(const char* mblob, ObjectPtr<Library> lib,
                       PackedFuncWrapper packed_func_wrapper, runtime::Module* root_module,
                       runtime::ModuleNode** dso_ctx_addr = nullptr) {
  ICHECK(mblob != nullptr);
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = mblob[i];
    nbytes |= (c & 0xffUL) << (i * 8);
  }
  dmlc::MemoryFixedSizeStream fs(const_cast<char*>(mblob + sizeof(nbytes)),
                                 static_cast<size_t>(nbytes));
  dmlc::Stream* stream = &fs;
  uint64_t size;
  ICHECK(stream->Read(&size));
  std::vector<Module> modules;
  std::vector<uint64_t> import_tree_row_ptr;
  std::vector<uint64_t> import_tree_child_indices;
  int num_dso_module = 0;

  for (uint64_t i = 0; i < size; ++i) {
    std::string tkey;
    ICHECK(stream->Read(&tkey));
    // "_lib" serves as a placeholder in the module import tree to indicate where
    // to place the DSOModule
    if (tkey == "_lib") {
      auto dso_module = Module(make_object<LibraryModuleNode>(lib, packed_func_wrapper));
      *dso_ctx_addr = dso_module.operator->();
      ++num_dso_module;
      modules.emplace_back(dso_module);
      ICHECK_EQ(num_dso_module, 1U) << "Multiple dso module detected, please upgrade tvm "
                                    << " to the latest before exporting the module";
    } else if (tkey == "_import_tree") {
      ICHECK(stream->Read(&import_tree_row_ptr));
      ICHECK(stream->Read(&import_tree_child_indices));
    } else {
      auto m = LoadModuleFromBinary(tkey, stream);
      modules.emplace_back(m);
    }
  }

  // if we are using old dll, we don't have import tree
  // so that we can't reconstruct module relationship using import tree
  if (import_tree_row_ptr.empty()) {
    auto n = make_object<LibraryModuleNode>(lib, packed_func_wrapper);
    auto module_import_addr = ModuleInternal::GetImportsAddr(n.operator->());
    for (const auto& m : modules) {
      module_import_addr->emplace_back(m);
    }
    *dso_ctx_addr = n.get();
    *root_module = Module(n);
  } else {
    for (size_t i = 0; i < modules.size(); ++i) {
      for (size_t j = import_tree_row_ptr[i]; j < import_tree_row_ptr[i + 1]; ++j) {
        auto module_import_addr = ModuleInternal::GetImportsAddr(modules[i].operator->());
        auto child_index = import_tree_child_indices[j];
        ICHECK(child_index < modules.size());
        module_import_addr->emplace_back(modules[child_index]);
      }
    }

    ICHECK(!modules.empty()) << "modules cannot be empty when import tree is present";
    // invariance: root module is always at location 0.
    // The module order is collected via DFS
    *root_module = modules[0];
  }
}

Module CreateModuleFromLibrary(ObjectPtr<Library> lib, PackedFuncWrapper packed_func_wrapper) {
  InitContextFunctions([lib](const char* fname) { return lib->GetSymbol(fname); });
  auto n = make_object<LibraryModuleNode>(lib, packed_func_wrapper);
  // Load the imported modules
  const char* dev_mblob =
      reinterpret_cast<const char*>(lib->GetSymbol(runtime::symbol::tvm_dev_mblob));

  Module root_mod;
  runtime::ModuleNode* dso_ctx_addr = nullptr;
  if (dev_mblob != nullptr) {
    ProcessModuleBlob(dev_mblob, lib, packed_func_wrapper, &root_mod, &dso_ctx_addr);
  } else {
    // Only have one single DSO Module
    root_mod = Module(n);
    dso_ctx_addr = root_mod.operator->();
  }

  // allow lookup of symbol from root (so all symbols are visible).
  if (auto* ctx_addr = reinterpret_cast<void**>(lib->GetSymbol(runtime::symbol::tvm_module_ctx))) {
    *ctx_addr = dso_ctx_addr;
  }

  return root_mod;
}
}  // namespace runtime
}  // namespace tvm
