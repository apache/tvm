/*!
 *  Copyright (c) 2017 by Contributors
 * \file packed_func_registry.cc
 * \brief The global registry of packed function.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/packed_func.h>
#include <unordered_map>
#include <memory>
#include "./runtime_base.h"

namespace tvm {
namespace runtime {

struct PackedFuncRegistry {
  // map storing the functions.
  // We delibrately used raw pointer
  // This is because PackedFunc can contain callbacks into the host languge(python)
  // and the resource can become invalid because of indeterminstic order of destruction.
  // The resources will only be recycled during program exit.
  std::unordered_map<std::string, PackedFunc*> fmap;

  static PackedFuncRegistry* Global() {
    static PackedFuncRegistry inst;
    return &inst;
  }
};

const PackedFunc& PackedFunc::RegisterGlobal(
    const std::string& name, PackedFunc f) {
  PackedFuncRegistry* r = PackedFuncRegistry::Global();
  auto it = r->fmap.find(name);
  CHECK(it == r->fmap.end())
      << "Global PackedFunc " << name << " is already registered";
  PackedFunc* fp = new PackedFunc(f);
  r->fmap[name] = fp;
  return *fp;
}

const PackedFunc& PackedFunc::GetGlobal(const std::string& name) {
  PackedFuncRegistry* r = PackedFuncRegistry::Global();
  auto it = r->fmap.find(name);
  CHECK(it != r->fmap.end())
      << "Global PackedFunc " << name << " is not registered";
  return *(it->second);
}

std::vector<std::string> PackedFunc::ListGlobalNames() {
  PackedFuncRegistry* r = PackedFuncRegistry::Global();
  std::vector<std::string> keys;
  keys.reserve(r->fmap.size());
  for (const auto &kv : r->fmap) {
    keys.push_back(kv.first);
  }
  return keys;
}

}  // namespace runtime
}  // namespace tvm

int TVMFuncRegisterGlobal(const char* name, TVMFunctionHandle f) {
  using tvm::runtime::PackedFunc;
  API_BEGIN();
  PackedFunc::RegisterGlobal(name, *static_cast<PackedFunc*>(f));
  API_END();
}

int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {
  using tvm::runtime::PackedFunc;
  API_BEGIN();
  *out = new PackedFunc(PackedFunc::GetGlobal(name));
  API_END();
}
