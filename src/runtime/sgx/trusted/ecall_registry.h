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
 *  Copyright (c) 2018 by Contributors
 * \file ecall_registry.h
 * \brief The global registry of packed functions available via ecall_packed_func.
 */
#ifndef TVM_RUNTIME_SGX_TRUSTED_ECALL_REGISTRY_H_
#define TVM_RUNTIME_SGX_TRUSTED_ECALL_REGISTRY_H_

#include <dmlc/logging.h>
#include <tvm/runtime/registry.h>
#include <string>
#include <algorithm>
#include <vector>

namespace tvm {
namespace runtime {
namespace sgx {

class ECallRegistry: public Registry {
 public:
  explicit ECallRegistry(std::string name) {
    name_ = name;
  }

  Registry& set_body(PackedFunc f) {
     func_ = f;
     return *this;
  }

  Registry& set_body(PackedFunc::FType f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }

  static Registry& Register(const std::string& name, bool override = false) {
    for (auto& r : exports_) {
      if (r.name_ == name) {
        CHECK(override) << "ecall " << name << " is already registered";
        return r;
      }
    }
    TVM_SGX_CHECKED_CALL(
        tvm_ocall_register_export(name.c_str(), exports_.size()));
    exports_.emplace_back(name);
    return exports_.back();
  }

  static bool Remove(const std::string& name) {
    LOG(FATAL) << "Removing enclave exports is not supported.";
  }

  static const PackedFunc* Get(const std::string& name) {
    for (const auto& r : exports_) {
      if (r.name_ == name) return &r.func_;
    }
    return nullptr;
  }

  static const PackedFunc* Get(unsigned func_id) {
    return func_id >= exports_.size() ? nullptr : &exports_[func_id].func_;
  }

  static std::vector<std::string> ListNames() {
    std::vector<std::string> names;
    names.resize(exports_.size());
    std::transform(exports_.begin(), exports_.end(), names.begin(),
                   [](ECallRegistry r) { return r.name_; });
    return names;
  }

  static std::vector<ECallRegistry> exports_;
};

std::vector<ECallRegistry> ECallRegistry::exports_;

/*!
 * \brief Register a function callable via ecall_packed_func
 * \code
 *   TVM_REGISTER_ENCLAVE_FUNC("DoThing")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_ENCLAVE_FUNC(OpName)                              \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) =                  \
      ::tvm::runtime::sgx::ECallRegistry::Register(OpName, true)

}  // namespace sgx
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_SGX_TRUSTED_ECALL_REGISTRY_H_
