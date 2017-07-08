/*!
 *  Copyright (c) 2017 by Contributors
 * \file system_lib_module.cc
 * \brief SystemLib module.
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_backend_api.h>
#include <mutex>
#include "./module_util.h"

namespace tvm {
namespace runtime {

class SystemLibModuleNode : public ModuleNode {
 public:
  SystemLibModuleNode() {
  }
  const char* type_key() const final {
    return "system_lib";
  }

  PackedFunc GetFunction(
      const std::string& name,
      const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tbl_.find(name);
    if (it != tbl_.end()) {
      return WrapPackedFunc(
          reinterpret_cast<BackendPackedCFunc>(it->second), sptr_to_self);
    } else {
      return PackedFunc();
    }
  }

  void RegisterSymbol(const std::string& name, void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (name == symbol::tvm_module_ctx) {
      void** ctx_addr = reinterpret_cast<void**>(ptr);
      *ctx_addr = this;
    } else if (name == symbol::tvm_dev_mblob) {
      ImportModuleBlob(reinterpret_cast<const char*>(ptr), &imports_);
    } else {
      auto it = tbl_.find(name);
      if (it != tbl_.end()) {
        if (ptr != it->second) {
          LOG(WARNING) << "SystemLib symbol " << name
                       << " get overriden to a different address "
                   << ptr << "->" << it->second;
          tbl_[name] = ptr;
        }
      } else {
        tbl_[name] = ptr;
      }
    }
  }

  static const std::shared_ptr<SystemLibModuleNode>& Global() {
    static std::shared_ptr<SystemLibModuleNode> inst =
        std::make_shared<SystemLibModuleNode>();
    return inst;
  }

 private:
  // Internal mutex
  std::mutex mutex_;
  // Internal symbol table
  std::unordered_map<std::string, void*> tbl_;
};

TVM_REGISTER_GLOBAL("module._GetSystemLib")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    *rv = runtime::Module(SystemLibModuleNode::Global());
  });
}  // namespace runtime
}  // namespace tvm

int TVMBackendRegisterSystemLibSymbol(const char* name, void* ptr) {
  tvm::runtime::SystemLibModuleNode::Global()->RegisterSymbol(name, ptr);
  return 0;
}
