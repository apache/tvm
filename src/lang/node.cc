/*!
 *  Copyright (c) 2018 by Contributors
 *  Implementation of IR Node API
 * \file node.cc
 */
#include <tvm/node/node.h>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace tvm {

namespace {
// single manager of operator information.
struct TypeManager {
  // mutex to avoid registration from multiple threads.
  // recursive is needed for trigger(which calls UpdateAttrMap)
  std::mutex mutex;
  std::atomic<uint32_t> type_counter{0};
  std::unordered_map<std::string, uint32_t> key2index;
  std::vector<std::string> index2key;
  // get singleton of the
  static TypeManager* Global() {
    static TypeManager inst;
    return &inst;
  }
};
}  // namespace

const bool Node::_DerivedFrom(uint32_t tid) const {
  static uint32_t tindex = TypeKey2Index(Node::_type_key);
  return tid == tindex;
}

// this is slow, usually caller always hold the result in a static variable.
uint32_t Node::TypeKey2Index(const char* key) {
  TypeManager *t = TypeManager::Global();
  std::lock_guard<std::mutex>(t->mutex);
  std::string skey = key;
  auto it = t->key2index.find(skey);
  if (it != t->key2index.end()) {
    return it->second;
  }
  uint32_t tid = ++(t->type_counter);
  t->key2index[skey] = tid;
  t->index2key.push_back(skey);
  return tid;
}

const char* Node::TypeIndex2Key(uint32_t index) {
  TypeManager *t = TypeManager::Global();
  std::lock_guard<std::mutex>(t->mutex);
  internal_assert(index != 0);
  return t->index2key.at(index - 1).c_str();
}

}  // namespace tvm
