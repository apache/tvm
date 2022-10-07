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
#include "module_equality.h"

#include <tvm/ir/module.h>

#include <memory>

namespace tvm {
namespace meta_schedule {

class ModuleEqualityManager {
 public:
  ModuleEqualityManager();

  static ModuleEqualityManager* Get();

  void Set(std::unique_ptr<ModuleEquality> eq) { cur_ = std::move(eq); }

  size_t Hash(IRModule mod) { return cur_->Hash(mod); }

  bool Equal(IRModule lhs, IRModule rhs) { return cur_->Equal(lhs, rhs); }

 private:
  std::unique_ptr<ModuleEquality> cur_;
};

ModuleEqualityManager::ModuleEqualityManager() {
  cur_ = std::make_unique<ModuleEqualityStructural>();
}

ModuleEqualityManager* ModuleEqualityManager::Get() {
  static ModuleEqualityManager manager;
  return &manager;
}

size_t ModuleHash::operator()(const IRModule& mod) const {
  return ModuleEqualityManager::Get()->Hash(mod);
}

bool ModuleEqual::operator()(const IRModule& lhs, const IRModule& rhs) const {
  return ModuleEqualityManager::Get()->Equal(lhs, rhs);
}

void SetModuleEquality(std::unique_ptr<ModuleEquality> eq) {
  ModuleEqualityManager::Get()->Set(std::move(eq));
}

}  // namespace meta_schedule
}  // namespace tvm
