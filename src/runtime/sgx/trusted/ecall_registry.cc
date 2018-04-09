/*!
 *  Copyright (c) 2018 by Contributors
 * \file ecall_registry.cc
 * \brief The global registry of packed functions available via ecall_packed_func.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {
namespace sgx {

class ECallRegistry: public Registry {
 public:
  struct Manager {
    std::vector<ECallRegistry> exports;
    std::mutex mutex;

    static Manager* Global() {
      static Manager inst;
      if (!inst.exports.size()) {
        ECallRegistry* init = new ECallRegistry("__init__");
        init->set_body([&](TVMArgs args, TVMRetValue* rv) {
          std::lock_guard<std::mutex> lock(inst.mutex);
          std::string exports = "";
          for (const auto& r : inst.exports) {
            exports += r.name_ + " ";
          }
          *rv = exports;
        });
        inst.exports.push_back(*init);
      }
      return &inst;
    }
  };

  explicit ECallRegistry(std::string name) {
    name_ = name;
  }

  Registry& set_body(PackedFunc f) {
     func_ = f;
     return *this;
  }

  Registry& set_body(PackedFunc::FType f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }

  static Registry& Register(const std::string& name, bool override = false) {
    Manager* m = Manager::Global();
    std::lock_guard<std::mutex> lock(m->mutex);
    for (auto& r : m->exports) {
      if (r.name_ == name) {
        CHECK(override)
          << "ecall " << name << " is already registered";
        return r;
      }
    }
    m->exports.emplace_back(name);
    return m->exports.back();
  }

  static bool Remove(const std::string& name) {
    LOG(FATAL) << "Removing enclave exports is not supported.";
  }

  static const PackedFunc* Get(const std::string& name) {
    ECallRegistry::Manager* m = ECallRegistry::Manager::Global();
    std::lock_guard<std::mutex> lock(m->mutex);
    for (const auto& r : m->exports) {
      if (r.name_ == name) return &r.func_;
    }
    return nullptr;
  }

  static const PackedFunc* Get(int func_id) {
    ECallRegistry::Manager* m = ECallRegistry::Manager::Global();
    std::lock_guard<std::mutex> lock(m->mutex);
    if (func_id >= m->exports.size()) return nullptr;
    return &m->exports[func_id].func_;
  }

  static std::vector<std::string> ListNames() {
    Manager* m = Manager::Global();
    std::lock_guard<std::mutex> lock(m->mutex);
    std::vector<std::string> names;
    names.resize(m->exports.size());
    std::transform(m->exports.begin(), m->exports.end(), names.begin(),
        [](ECallRegistry r) { return r.name_; });
    return names;
  }
};

/*!
 * \brief Register a function callable via ecall_packed_func
 * \code
 *   TVM_REGISTER_ENCLAVE_FUNC("DoThing")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_ENCLAVE_FUNC(OpName)                              \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) =                  \
      ::tvm::runtime::sgx::ECallRegistry::Register(OpName)

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm
