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
#ifndef TVM_META_SCHEDULE_MODULE_EQUALITY_H_
#define TVM_META_SCHEDULE_MODULE_EQUALITY_H_

#include <tvm/ir/module.h>
#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>

#include <memory>
#include <string>

namespace tvm {
namespace meta_schedule {

class ModuleEquality {
 public:
  virtual ~ModuleEquality() = default;

  virtual size_t Hash(IRModule mod) const = 0;
  virtual bool Equal(IRModule lhs, IRModule rhs) const = 0;

  static std::unique_ptr<ModuleEquality> Create(const std::string& mod_eq_name);
};

class ModuleHash {
 public:
  explicit ModuleHash(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}
  size_t operator()(const IRModule& mod) const { return mod_eq_.Hash(mod); }

 private:
  const ModuleEquality& mod_eq_;
};

class ModuleEqual {
 public:
  explicit ModuleEqual(const ModuleEquality& mod_eq) : mod_eq_(mod_eq) {}
  bool operator()(const IRModule& lhs, const IRModule& rhs) const {
    return mod_eq_.Equal(lhs, rhs);
  }

 private:
  const ModuleEquality& mod_eq_;
};

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_MODULE_EQUALITY_H_
