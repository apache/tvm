/*!
 *  Copyright (c) 2017 by Contributors
 * \file api_registry.cc
 */
#include <tvm/expr.h>
#include <tvm/tensor.h>
#include <tvm/api_registry.h>
#include <memory>

namespace tvm {

struct APIManager {
  std::unordered_map<std::string, std::unique_ptr<APIRegistry> > fmap;

  static APIManager* Global() {
    static APIManager inst;
    return &inst;
  }
};

APIRegistry& APIRegistry::__REGISTER__(const std::string& name) {  // NOLINT(*)
  APIManager* m = APIManager::Global();
  CHECK(!m->fmap.count(name))
      << "API function " << name << " has already been registered";
  std::unique_ptr<APIRegistry> p(new APIRegistry());
  p->name_ = name;
  m->fmap[name] = std::move(p);
  return *(m->fmap[name]);
}

APIRegistry& APIRegistry::set_body(PackedFunc f) {  // NOLINT(*)
  PackedFunc::RegisterGlobal(name_, f);
  return *this;
}
}  // namespace tvm
