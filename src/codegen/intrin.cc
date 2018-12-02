/*!
 *  Copyright (c) 2018 by Contributors
 * \file intrin.cc
 * \brief Support for intrinsic registry.
 */
#include <tvm/base.h>
#include <tvm/intrin.h>

#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_set>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(tvm::Intrin);
}  // namespace dmlc

namespace tvm {

// single manager of intrinsic information.
struct IntrinManager {
  // mutex to avoid registration from multiple threads.
  // recursive is needed for trigger(which calls UpdateAttrMap)
  std::recursive_mutex mutex;
  // global intrinsic counter
  std::atomic<int> op_counter{0};
  // storage of additional attribute table.
  std::unordered_map<std::string, std::unique_ptr<dmlc::any> > attr;
  // get singleton of the
  static IntrinManager* Global() {
    static IntrinManager inst;
    return &inst;
  }
};

// constructor
Intrin::Intrin() {
  IntrinManager* mgr = IntrinManager::Global();
  index_ = mgr->op_counter++;
  inplace = false;
  inplace_map.clear();
}

// find intrinsic by name
const Intrin* Intrin::Get(const std::string& name) {
  const Intrin* op = dmlc::Registry<Intrin>::Find(name);
  CHECK(op != nullptr)
      << "Intrinerator " << name << " is not registered";
  return op;
}

// Get attribute map by key
const dmlc::any* Intrin::GetAttrMap(const std::string& key) {
  auto& dict =  IntrinManager::Global()->attr;
  auto it = dict.find(key);
  if (it != dict.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

// update attribute map
void Intrin::UpdateAttrMap(const std::string& key,
                       std::function<void(dmlc::any*)> updater) {
  IntrinManager* mgr = IntrinManager::Global();
  std::lock_guard<std::recursive_mutex>(mgr->mutex);
  std::unique_ptr<dmlc::any>& value = mgr->attr[key];
  if (value.get() == nullptr) value.reset(new dmlc::any());
  if (updater != nullptr) updater(value.get());
}

}  // namespace tvm
