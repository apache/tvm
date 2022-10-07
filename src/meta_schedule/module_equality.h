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
#ifndef TVM_META_SCHEDULE_MODULE_EQUAILITY_H_
#define TVM_META_SCHEDULE_MODULE_EQUAILITY_H_

#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>

#include <memory>

namespace tvm {
namespace meta_schedule {

class ModuleEquality {
 public:
  virtual ~ModuleEquality() = default;

  virtual size_t Hash(IRModule mod) = 0;
  virtual bool Equal(IRModule lhs, IRModule rhs) = 0;
};

class ModuleEqualityStructural : public ModuleEquality {
 public:
  size_t Hash(IRModule mod) { return tvm::StructuralHash()(mod); }
  bool Equal(IRModule lhs, IRModule rhs) { return tvm::StructuralEqual()(lhs, rhs); }
};

class ModuleHash {
 public:
  size_t operator()(const IRModule& mod) const;
};

class ModuleEqual {
 public:
  bool operator()(const IRModule& lhs, const IRModule& rhs) const;
};

void SetModuleEquality(std::unique_ptr<ModuleEquality> eq);

}  // namespace meta_schedule
}  // namespace tvm

#endif
