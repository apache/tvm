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
 * \file dso_libary.cc
 * \brief Create library module to load from dynamic shared library.
 */
#include <tvm/runtime/memory.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "library_module.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#endif

#if defined(__hexagon__)
extern "C" {
#include <HAP_farf.h>
}
#endif

namespace tvm {
namespace runtime {

/*!
 * \brief Dynamic shared library object used to load
 * and retrieve symbols by name. This is the default
 * module TVM uses for host-side AOT compilation.
 */
class DSOLibrary final : public Library {
 public:
  ~DSOLibrary();

  /*!
   * \brief Initialize by loading and storing
   * a handle to the underlying shared library.
   * \param name The string name/path to the
   * shared library over which to initialize.
   */
  void Init(const std::string& name);

  /*!
   * \brief Returns the symbol address within
   * the shared library for a given symbol name.
   * \param name The name of the symbol.
   * \return The symbol.
   */
  void* GetSymbol(const char* name) final;

  /*! \brief List symbols available within the module
   *
   * \param callback The callback to be executed for each symbol in
   *    the library.
   */
  void ListSymbols(std::function<void(const char*)> callback) final;

 private:
  /*! \brief Private implementation of symbol lookup.
   *
   * Implementation is operating system dependent.
   *
   * \param name The name of the symbol.
   *
   * \return The symbol.
   */
  void* GetSymbol_(const char* name);

  /*! \brief Private implementation of symbol lookup.
   *
   * Implementation is operating system dependent.
   *
   * \param callback The callback for each symbol located.
   *
   * \return The symbol.
   */
  void ListSymbols_(std::function<void(const char*)> callback);

  /*! \brief Implementation of shared library load.
   *  Implementation is operating system dependent.
   *  \param The name/path of the shared library.
   */
  void Load(const std::string& name);

  /*! \brief Implementation of shared library unload.
   *  Implementation is operating system dependent.
   */
  void Unload();

#if defined(_WIN32)
  //! \brief Windows library handle
  HMODULE lib_handle_{nullptr};
#else
  // \brief Linux library handle
  void* lib_handle_{nullptr};
#endif
};

DSOLibrary::~DSOLibrary() {
  if (lib_handle_) Unload();
}

void DSOLibrary::Init(const std::string& name) { Load(name); }

void* DSOLibrary::GetSymbol(const char* name) { return GetSymbol_(name); }

void DSOLibrary::ListSymbols(std::function<void(const char*)> callback) { ListSymbols_(callback); }

#if defined(_WIN32)

void* DSOLibrary::GetSymbol_(const char* name) {
  return reinterpret_cast<void*>(GetProcAddress(lib_handle_, (LPCSTR)name));  // NOLINT(*)
}

void DSOLibrary::ListSymbols_(std::function<void(const char*)> callback) {}

void DSOLibrary::Load(const std::string& name) {
  // use wstring version that is needed by LLVM.
  std::wstring wname(name.begin(), name.end());
  lib_handle_ = LoadLibraryW(wname.c_str());
  ICHECK(lib_handle_ != nullptr) << "Failed to load dynamic shared library " << name;
}

void DSOLibrary::Unload() {
  FreeLibrary(lib_handle_);
  lib_handle_ = nullptr;
}

#else

void DSOLibrary::Load(const std::string& name) {
  lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  ICHECK(lib_handle_ != nullptr) << "Failed to load dynamic shared library " << name << " "
                                 << dlerror();
#if defined(__hexagon__)
  int p;
  int rc = dlinfo(lib_handle_, RTLD_DI_LOAD_ADDR, &p);
  if (rc)
    FARF(ERROR, "error getting model .so start address : %u", rc);
  else
    FARF(ALWAYS, "Model .so Start Address : %x", p);
#endif
}

void* DSOLibrary::GetSymbol_(const char* name) { return dlsym(lib_handle_, name); }

void DSOLibrary::ListSymbols_(std::function<void(const char*)> callback) {
  // Adapted from https://stackoverflow.com/a/62205128

  struct link_map* map = nullptr;
  dlinfo(lib_handle_, RTLD_DI_LINKMAP, &map);

  Elf64_Sym* symbol_table = nullptr;
  char* string_table = nullptr;
  int entry_size = 0;
  for (auto section = map->l_ld; section->d_tag != DT_NULL; ++section) {
    if (section->d_tag == DT_SYMTAB) {
      symbol_table = (Elf64_Sym*)section->d_un.d_ptr;
    } else if (section->d_tag == DT_STRTAB) {
      string_table = (char*)section->d_un.d_ptr;
    } else if (section->d_tag == DT_SYMENT) {
      entry_size = section->d_un.d_val;
    }
  }

  CHECK(symbol_table) << "RuntimeError: "
                      << "Malformed ELF binary '" << map->l_name
                      << "', no symbol table (DT_SYMTAB) found";
  CHECK(string_table) << "RuntimeError: "
                      << "Malformed ELF binary '" << map->l_name
                      << "', no string table (DT_STRTAB) found";

  int symbol_table_size = string_table - (char*)symbol_table;
  int num_symbols = symbol_table_size / entry_size;
  for (int i = 0; i < num_symbols; i++) {
    Elf64_Sym* symbol = &symbol_table[i];

    if (
        // If the symbol is a function
        ELF64_ST_TYPE(symbol->st_info) == STT_FUNC &&
        // defined with global linkage
        ELF64_ST_BIND(symbol->st_info) == STB_GLOBAL &&
        // and visible to external modules.
        ELF64_ST_VISIBILITY(symbol->st_other) == STV_DEFAULT) {
      const char* symbol_name = &string_table[symbol->st_name];
      callback(symbol_name);
    }
  }
}

void DSOLibrary::Unload() {
  dlclose(lib_handle_);
  lib_handle_ = nullptr;
}

#endif

ObjectPtr<Library> CreateDSOLibraryObject(std::string library_path) {
  auto n = make_object<DSOLibrary>();
  n->Init(library_path);
  return n;
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_so").set_body([](TVMArgs args, TVMRetValue* rv) {
  ObjectPtr<Library> n = CreateDSOLibraryObject(args[0]);
  *rv = CreateModuleFromLibrary(n);
});
}  // namespace runtime
}  // namespace tvm
