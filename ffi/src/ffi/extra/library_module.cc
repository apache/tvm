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
 * \file src/ffi/extra/library_module.cc
 *
 * \brief Library module implementation.
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/module.h>

#include "buffer_stream.h"
#include "module_internal.h"

namespace tvm {
namespace ffi {

class LibraryModuleObj final : public ModuleObj {
 public:
  explicit LibraryModuleObj(ObjectPtr<Library> lib) : lib_(lib) {}

  const char* kind() const final { return "library"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return Module::kBinarySerializable | Module::kRunnable; };

  Optional<ffi::Function> GetFunction(const String& name) final {
    TVMFFISafeCallType faddr;
    faddr = reinterpret_cast<TVMFFISafeCallType>(lib_->GetSymbol(name.c_str()));
    // ensure the function keeps the Library Module alive
    Module self_strong_ref = GetRef<Module>(this);
    if (faddr != nullptr) {
      return ffi::Function::FromPacked([faddr, self_strong_ref](ffi::PackedArgs args,
                                                                ffi::Any* rv) {
        TVM_FFI_ICHECK_LT(rv->type_index(), ffi::TypeIndex::kTVMFFIStaticObjectBegin);
        TVM_FFI_CHECK_SAFE_CALL((*faddr)(nullptr, reinterpret_cast<const TVMFFIAny*>(args.data()),
                                         args.size(), reinterpret_cast<TVMFFIAny*>(rv)));
      });
    }
    return std::nullopt;
  }

 private:
  ObjectPtr<Library> lib_;
};

Module LoadModuleFromBytes(const std::string& kind, const Bytes& bytes) {
  std::string loader_key = "ffi.Module.load_from_bytes." + kind;
  const auto floader = tvm::ffi::Function::GetGlobal(loader_key);
  if (!floader.has_value()) {
    TVM_FFI_THROW(RuntimeError) << "Library binary was created using {" << kind
                                << "} but a loader of that name is not registered. "
                                << "Make sure to have runtime that registers " << loader_key;
  }
  return (*floader)(bytes).cast<Module>();
}

/*!
 * \brief Process libary binary to recover binary-serialized modules
 * \param library_bin The binary embedded in the library.
 * \param opt_lib The library, can be nullptr in which case we expect to deserialize
 *            all binary-serialized modules
 * \param library_ctx_addr the pointer to library module as ctx addr
 * \return the root module
 *
 */
Module ProcessLibraryBin(const char* library_bin, ObjectPtr<Library> opt_lib,
                         void** library_ctx_addr = nullptr) {
  // Layout of the library binary:
  // <nbytes : u64> <import_tree> <key0: str> <val0: bytes> <key1: str> <val1: bytes> ...
  // key can be: "_lib", or a module kind
  //   - "_lib" indicate this location places the library module
  //   - other keys are module kinds
  // Import tree structure (CSR structure of child indices):
  // <import_tree> = <indptr: vec<u64>> <child_indices: vec<u64>>
  TVM_FFI_ICHECK(library_bin != nullptr);
  uint64_t nbytes = 0;
  for (size_t i = 0; i < sizeof(nbytes); ++i) {
    uint64_t c = library_bin[i];
    nbytes |= (c & 0xffUL) << (i * 8);
  }

  BufferInStream stream(library_bin + sizeof(nbytes), static_cast<size_t>(nbytes));
  std::vector<uint64_t> import_tree_indptr;
  std::vector<uint64_t> import_tree_child_indices;
  TVM_FFI_ICHECK(stream.Read(&import_tree_indptr));
  TVM_FFI_ICHECK(stream.Read(&import_tree_child_indices));
  size_t num_modules = import_tree_indptr.size() - 1;
  std::vector<Module> modules;
  modules.reserve(num_modules);

  for (uint64_t i = 0; i < num_modules; ++i) {
    std::string kind;
    TVM_FFI_ICHECK(stream.Read(&kind));
    // "_lib" serves as a placeholder in the module import tree to indicate where
    // to place the DSOModule
    if (kind == "_lib") {
      TVM_FFI_ICHECK(opt_lib != nullptr) << "_lib is not allowed during module serialization";
      auto lib_mod_ptr = make_object<LibraryModuleObj>(opt_lib);
      if (library_ctx_addr) {
        *library_ctx_addr = lib_mod_ptr.get();
      }
      modules.emplace_back(Module(lib_mod_ptr));
    } else {
      std::string module_bytes;
      TVM_FFI_ICHECK(stream.Read(&module_bytes));
      Module m = LoadModuleFromBytes(kind, Bytes(module_bytes));
      modules.emplace_back(m);
    }
  }
  for (size_t i = 0; i < modules.size(); ++i) {
    for (size_t j = import_tree_indptr[i]; j < import_tree_indptr[i + 1]; ++j) {
      Array<Any>* module_imports = ModuleObj::InternalUnsafe::GetImports(modules[i].operator->());
      auto child_index = import_tree_child_indices[j];
      TVM_FFI_ICHECK(child_index < modules.size());
      module_imports->emplace_back(modules[child_index]);
    }
  }
  return modules[0];
}

// registry to store context symbols
class ContextSymbolRegistry {
 public:
  void InitContextSymbols(ObjectPtr<Library> lib) {
    for (const auto& [name, symbol] : context_symbols_) {
      if (void** symbol_addr = reinterpret_cast<void**>(lib->GetSymbol(name.c_str()))) {
        *symbol_addr = symbol;
      }
    }
  }

  void VisitContextSymbols(const ffi::TypedFunction<void(String, void*)>& callback) {
    for (const auto& [name, symbol] : context_symbols_) {
      callback(name, symbol);
    }
  }

  void Register(String name, void* symbol) { context_symbols_.emplace_back(name, symbol); }

  static ContextSymbolRegistry* Global() {
    static ContextSymbolRegistry* inst = new ContextSymbolRegistry();
    return inst;
  }

 private:
  std::vector<std::pair<String, void*>> context_symbols_;
};

void Module::VisitContextSymbols(const ffi::TypedFunction<void(String, void*)>& callback) {
  ContextSymbolRegistry::Global()->VisitContextSymbols(callback);
}

Module CreateLibraryModule(ObjectPtr<Library> lib) {
  const char* library_bin =
      reinterpret_cast<const char*>(lib->GetSymbol(ffi::symbol::tvm_ffi_library_bin));
  void** library_ctx_addr =
      reinterpret_cast<void**>(lib->GetSymbol(ffi::symbol::tvm_ffi_library_ctx));

  ContextSymbolRegistry::Global()->InitContextSymbols(lib);
  if (library_bin != nullptr) {
    // we have embedded binaries that needs to be deserialized
    return ProcessLibraryBin(library_bin, lib, library_ctx_addr);
  } else {
    // Only have one single DSO Module
    auto lib_mod_ptr = make_object<LibraryModuleObj>(lib);
    Module root_mod = Module(lib_mod_ptr);
    if (library_ctx_addr) {
      *library_ctx_addr = root_mod.operator->();
    }
    return root_mod;
  }
}

}  // namespace ffi
}  // namespace tvm

int TVMFFIEnvModRegisterContextSymbol(const char* name, void* symbol) {
  TVM_FFI_SAFE_CALL_BEGIN();
  tvm::ffi::String s_name(name);
  tvm::ffi::ContextSymbolRegistry::Global()->Register(s_name, symbol);
  TVM_FFI_SAFE_CALL_END();
}
