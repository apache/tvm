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
 * \file tirx/analysis/verify_well_formed.cc
 * \brief Check if schedulable tirx is well-formed.
 */

#include "verify_well_formed.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <exception>
#include <optional>
#include <tuple>
#include <variant>

#include "../ir/tir_visitor_with_path.h"
#include "tvm/ir/module.h"

namespace tvm {
namespace tirx {

using AccessPath = ffi::reflection::AccessPath;

/* \brief Verify unique tirx::Var for each environment thread
 *
 * Environment threads, such as CUDA's `threadIdx.x`, are defined in
 * TIR using an `AttrStmt` with the key `attr::thread_extent`.  A
 * `PrimFunc` may contain multiple such attributes for the same
 * environment thread.  However, all such attributes must use the same
 * `tirx::Var` for a given thread.
 */
class SingleEnvThreadVerifier : public Verifier<SingleEnvThreadVerifier> {
 public:
  using Verifier::Verifier;

 private:
  void Visit(const PrimFunc& prim_func, AccessPath path) override {
    Verifier::Visit(prim_func, path);
    env_thread_vars_.clear();
  }

  void EnterDef(const IterVar& iter_var, AccessPath path) override {
    if (iter_var->iter_type == IterVarType::kThreadIndex) {
      if (auto it = env_thread_vars_.find(iter_var->thread_tag); it != env_thread_vars_.end()) {
        const auto& [prev_var, prev_path] = it->second;
        Verify(prev_var.same_as(iter_var->var))
            << "ValueError: "
            << "PrimFunc uses multiple distinct TIR variables "
            << " for the environment thread \"" << iter_var->thread_tag << "\".  "
            << "While multiple tirx::AttrStmt may define the same environment thread, "
            << "all definitions within a single PrimFunc must share the same tirx::Var.  "
            << "Binding of environment thread \"" << iter_var->thread_tag
            << "\" to the TIR variable " << iter_var->var->name << " at " << path
            << " conflicts with the previous binding to the TIR variable " << prev_var->name
            << " at " << path;
      } else {
        env_thread_vars_.insert({iter_var->thread_tag, {iter_var->var, path}});
      }
    }
  }

  std::unordered_map<ffi::String, std::tuple<Var, AccessPath>> env_thread_vars_;
};

bool VerifyWellFormed(const PrimFunc& func, bool assert_mode) {
  return VerifyWellFormedCommon<TIRVisitorWithPath>(func, assert_mode);
}

bool VerifyWellFormed(const IRModule& mod, bool assert_mode) {
  return VerifyWellFormedCommon<TIRVisitorWithPath>(mod, assert_mode);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.analysis.VerifyWellFormed", [](const ffi::ObjectRef& obj, bool assert_mode) {
        if (auto opt = obj.as<PrimFunc>()) {
          return VerifyWellFormed(opt.value(), assert_mode);
        } else if (auto opt = obj.as<IRModule>()) {
          return VerifyWellFormed(opt.value(), assert_mode);
        } else {
          TVM_FFI_THROW(InternalError)
              << "Expected VerifyWellFormed argument to be a PrimFunc or IRModule, but found "
              << obj->GetTypeKey();
        }
      });
}

}  // namespace tirx
}  // namespace tvm
