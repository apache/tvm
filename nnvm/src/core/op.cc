/*!
 *  Copyright (c) 2016 by Contributors
 * \file op.cc
 * \brief Support for operator registry.
 */
#include <nnvm/base.h>
#include <nnvm/op.h>

#include <memory>
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
  std::unordered_map<std::string, std::unique_ptr<any> > attr;
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
const any* Op::GetAttrMap(const std::string& key) {
  auto& dict =  OpManager::Global()->attr;
  auto it = dict.find(key);
  if (it != dict.end()) {
    return it->second.get();
  } else {
    return nullptr;
  }
}

// update attribute map
void Op::UpdateAttrMap(const std::string& key,
                       std::function<void(any*)> updater) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::mutex>(mgr->mutex);
  std::unique_ptr<any>& value = mgr->attr[key];
  if (value.get() == nullptr) value.reset(new any());
  if (updater != nullptr) updater(value.get());
}

}  // namespace nnvm
