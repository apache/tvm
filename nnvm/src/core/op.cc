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
 *  Copyright (c) 2016 by Contributors
 * \file op.cc
 * \brief Support for operator registry.
 */
#include <nnvm/base.h>
#include <nnvm/op.h>

#include <memory>
#include <atomic>
#include <mutex>
#include <unordered_set>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(nnvm::Op);
}  // namespace dmlc

namespace nnvm {

// single manager of operator information.
struct OpManager {
  // mutex to avoid registration from multiple threads.
  // recursive is needed for trigger(which calls UpdateAttrMap)
  std::recursive_mutex mutex;
  // global operator counter
  std::atomic<int> op_counter{0};
  // storage of additional attribute table.
  std::unordered_map<std::string, std::unique_ptr<any> > attr;
  // storage of existing triggers
  std::unordered_map<std::string, std::vector<std::function<void(Op*)>  > > tmap;
  // group of each operator.
  std::vector<std::unordered_set<std::string> > op_group;
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

Op& Op::add_alias(const std::string& alias) {  // NOLINT(*)
  dmlc::Registry<Op>::Get()->AddAlias(this->name, alias);
  return *this;
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
  std::lock_guard<std::recursive_mutex>(mgr->mutex);
  std::unique_ptr<any>& value = mgr->attr[key];
  if (value.get() == nullptr) value.reset(new any());
  if (updater != nullptr) updater(value.get());
}

void Op::AddGroupTrigger(const std::string& group_name,
                         std::function<void(Op*)> trigger) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::recursive_mutex>(mgr->mutex);
  auto& tvec = mgr->tmap[group_name];
  tvec.push_back(trigger);
  auto& op_group = mgr->op_group;
  for (const Op* op : dmlc::Registry<Op>::List()) {
    if (op->index_ < op_group.size() &&
        op_group[op->index_].count(group_name) != 0) {
      trigger((Op*)op);  // NOLINT(*)
    }
  }
}

Op& Op::include(const std::string& group_name) {
  OpManager* mgr = OpManager::Global();
  std::lock_guard<std::recursive_mutex>(mgr->mutex);
  auto it = mgr->tmap.find(group_name);
  if (it != mgr->tmap.end()) {
    for (auto& trigger : it->second) {
      trigger(this);
    }
  }
  auto& op_group = mgr->op_group;
  if (index_ >= op_group.size()) {
    op_group.resize(index_ + 1);
  }
  op_group[index_].insert(group_name);
  return *this;
}

}  // namespace nnvm
