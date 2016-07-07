/*!
 *  Copyright (c) 2016 by Contributors
 * \file op.cc
 * \brief Support for operator registry.
 */
#include <nnvm/base.h>
#include <nnvm/op.h>

#include <atomic>
#include <mutex>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(nnvm::Op);
}  // namespace dmlc

namespace nnvm {

// single manager of operator information.
struct OpManager {
  // mutex to avoid registration from multiple threads.
  std::mutex mutex;
  // global operator counter
  std::atomic<int> op_counter{0};
  // storage of additional attribute table.
  std::unordered_map<std::string, any> attr;
  // get singleton of the
  static OpManager* Global() {
    static OpManager inst;
    return &inst;
  }
};

// constructor
Op::Op() {
  OpManager* mgr = OpManager::Global();
  index_ = mgr->op_counter++;
}

// find operator by name
const Op* Op::Get(const std::string& name) {
  const Op* op = dmlc::Registry<Op>::Find(name);
  CHECK(op != nullptr)
      << "Operator " << name << " is not registered";
  return op;
}

// Get attribute map by key
const any& Op::GetAttrMap(const std::string& key) {
  // assume no operator registration during
  // the execution phase.
  const auto& dict = OpManager::Global()->attr;
  auto it = dict.find(key);
  CHECK(it != dict.end() && it->first == key)
      << "Cannot find Operator attribute " << key
      << " for any operator";
  return it->second;
}

// update attribute map by updater function.
void Op::UpdateAttrMap(const std::string& key,
                       std::function<void(any*)> updater) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex>(mgr->mutex);
  any& value = mgr->attr[key];
  updater(&value);
}

}  // namespace nnvm
