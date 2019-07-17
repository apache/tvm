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
 *  Implementation of Node API
 * \file node.cc
 */
#include <tvm/node/node.h>
#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_map>

// TODO(tqchen):
// Think of re-organize and consolidate with object.
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

TVM_DLL const bool Node::_DerivedFrom(uint32_t tid) const {
  static uint32_t tindex = TypeKey2Index(Node::_type_key);
  return tid == tindex;
}

// this is slow, usually caller always hold the result in a static variable.
TVM_DLL uint32_t Node::TypeKey2Index(const char* key) {
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

TVM_DLL const char* Node::TypeIndex2Key(uint32_t index) {
  TypeManager *t = TypeManager::Global();
  std::lock_guard<std::mutex>(t->mutex);
  CHECK_NE(index, 0);
  return t->index2key.at(index - 1).c_str();
}
}  // namespace tvm
