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
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>

#include <memory>

namespace tvm {
namespace meta_schedule {

class ModuleEqualityStructural : public ModuleEquality {
 public:
  size_t Hash(IRModule mod) const { return tvm::StructuralHash()(mod); }
  bool Equal(IRModule lhs, IRModule rhs) const { return tvm::StructuralEqual()(lhs, rhs); }
};

std::unique_ptr<ModuleEquality> ModuleEquality::Create(const std::string& mod_eq_name) {
  if (mod_eq_name == "structural") {
    return std::make_unique<ModuleEqualityStructural>();
  }
  LOG(FATAL) << "Unknown module equality " << mod_eq_name;
  return nullptr;
}

}  // namespace meta_schedule
}  // namespace tvm
