/*!
 *  Copyright (c) 2017 by Contributors
 * \file dso_module.cc
 * \brief Module to load from dynamic shared library.
 */
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include "./meta_data.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace tvm {
namespace runtime {

// Module to load from dynamic shared libary.
// This is the default module TVM used for hostside AOT
class DSOModuleNode : public ModuleNode {
 public:
  ~DSOModuleNode() {
    if (lib_handle_) Unload();
  }

  const char* type_key() const {
    return "dso";
  }

  void PreCompile(const std::string& name, TVMContext ctx) final {
    GetFuncPtr(name);
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    BackendPackedCFunc faddr = GetFuncPtr(name);
    if (faddr == nullptr) return PackedFunc();
    return PackedFunc([faddr, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        int ret = (*faddr)(
            (void*)args.values, // NOLINT(*)
            (int*)args.type_codes, // NOLINT(*)
            args.num_args);
        CHECK_EQ(ret, 0) << TVMGetLastError();
      });
  }

  void SaveToFile(const std::string& file_name,
                  const std::string& format) final {
    LOG(FATAL) << "Cannot save dso to another file";
  }

  std::string GetSource(const std::string& format) final {
    return "";
  }

  void Init(const std::string& name) {
    Load(name);
    CHECK(lib_handle_ != nullptr)
        << "Failed to load dynamic shared library " << name;
    void** ctx_addr =
        reinterpret_cast<void**>(
            GetGlobalVPtr(runtime::symbol::tvm_module_ctx));
    if (ctx_addr != nullptr) {
      *ctx_addr = this;
    }
  }

 private:
  // Platform dependent handling.
#if defined(_WIN32)
  // library handle
  HMODULE lib_handle_{nullptr};
  // Load the library
  void Load(const std::string& name) {
    // use wstring version that is needed by LLVM.
    std::wstring wname(name.begin(), name.end());
    lib_handle_ = LoadLibraryW(wname.c_str());
  }
  BackendPackedCFunc GetFuncPtr(const std::string& name) {
    return reinterpret_cast<BackendPackedCFunc>(
        GetProcAddress(lib_handle_, (LPCSTR)name.c_str()));  // NOLINT(*)
  }
  void* GetGlobalVPtr(const std::string& name) {
    return reinterpret_cast<void*>(
        GetProcAddress(lib_handle_, name.c_str())); // NOLINT(*)
  }
  void Unload() {
    FreeLibrary(lib_handle_);
  }
#else
  // Library handle
  void* lib_handle_{nullptr};
  // load the library
  void Load(const std::string& name) {
    lib_handle_ = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  }
  BackendPackedCFunc GetFuncPtr(const std::string& name) {
    return reinterpret_cast<BackendPackedCFunc>(
        dlsym(lib_handle_, name.c_str()));
  }
  void* GetGlobalVPtr(const std::string& name) {
    return dlsym(lib_handle_, name.c_str());
  }
  void Unload() {
    dlclose(lib_handle_);
  }
#endif
};

TVM_REGISTER_GLOBAL(_module_loadfile_so)
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<DSOModuleNode> n = std::make_shared<DSOModuleNode>();
    n->Init(args[0]);
    *rv = runtime::Module(n);
  });
}  // namespace runtime
}  // namespace tvm
