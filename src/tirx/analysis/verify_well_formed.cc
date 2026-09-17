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

namespace {
std::vector<bool (*)(const PrimFunc&, bool)>& WellFormedExtensions() {
  static std::vector<bool (*)(const PrimFunc&, bool)> extensions;
  return extensions;
}
}  // namespace
void RegisterWellFormedExtension(bool (*verify)(const PrimFunc&, bool)) {
  WellFormedExtensions().push_back(verify);
}

class UndefinedVarVerifier : public Verifier<UndefinedVarVerifier> {
 public:
  // Until templated-this arrives in C++23, the CRTP can't inject a
  // constructor into the child class.  Therefore, must explicitly add
  // the constructor.
  using Verifier::Verifier;

 private:
  using Verifier::Visit;
  void Visit(const PrimFunc& prim_func, AccessPath path) override {
    Verifier::Visit(prim_func, path);
    redefine_allowed_within_function_.clear();
  }

  void EnterDef(const IterVar& iter_var, AccessPath path) override {
    Verifier::EnterDef(iter_var, path);
    if (iter_var->iter_type == IterVarType::kThreadIndex) {
      redefine_allowed_within_function_.insert(iter_var->var);
    }
  }

  void EnterDef(const BufferVar& buffer, AccessPath path) override {
    Verifier::EnterDef(buffer, path);
  }

  void EnterDef(const Var& var, AccessPath path) override {
    bool redefine_is_allowed = redefine_allowed_within_function_.count(var);
    {
      auto it = currently_defined_.find(var);
      auto verify = Verify(it == currently_defined_.end() || redefine_is_allowed);
      verify << "ValueError: "
             << "TIR is ill-formed, "
             << "due to multiple nested definitions of variable " << var->name << ".";
      if (it != currently_defined_.end()) {
        verify << " It was first defined at " << it->second << ", and was re-defined at " << path;
      }
    }

    {
      auto it = previously_defined_.find(var);
      auto verify = Verify(it == previously_defined_.end() || redefine_is_allowed);
      verify << "ValueError: "
             << "TIR is ill-formed, "
             << "due to multiple definitions of variable " << var->name << ".";
      if (it != previously_defined_.end()) {
        verify << " It was first defined at " << it->second << ", and was later re-defined at "
               << path;
      }
    }

    currently_defined_.insert({var, path});
  }

  void ExitDef(const Var& var, AccessPath path) override {
    auto active_def = currently_defined_.find(var);

    currently_defined_.erase(active_def);
    previously_defined_.insert({var, path});
  }

  void Dispatch_(const VarNode* op, AccessPath path) override {
    auto var = ffi::GetRef<Var>(op);

    auto active_def = currently_defined_.find(var);
    auto verify = Verify(active_def != currently_defined_.end());
    verify << "ValueError: "
           << "Invalid use of undefined variable " << var->name << " at " << path << ".";

    // Check if there was a previous definition, and append the
    // location to the error message if there was.  This is to aid in
    // debugging, by distinguishing between a variable that is
    // currently out-of-scope, and a variable that never had a
    // definition in the first place.
    if (auto prev_def = previously_defined_.find(var); prev_def != previously_defined_.end()) {
      verify << ".  While this variable was previously defined at " << prev_def->second
             << ", this definition is no longer in-scope.";
    }
  }

  // Variables that are defined in the currently-visited scope.
  std::unordered_map<Var, AccessPath> currently_defined_;

  // Variables that were previously defined, and are now out of scope.
  std::unordered_map<Var, AccessPath> previously_defined_;

  // Special variables that are allowed to be re-defined, so long as
  // that re-definition occurs within the same PrimFunc.  For example
  std::unordered_set<Var> redefine_allowed_within_function_;
};

/*! \brief Verify that buffers with a declaration are not used outside their declared scope.
 *
 * When a buffer is declared via one of the following sites:
 *   - BufferType-annotated PrimFunc parameters
 *   - DeclBuffer statement
 *   - SBlock::alloc_buffers
 *   - SBlock::match_buffers
 *
 * it must not appear in a BufferLoad, BufferStore, or TensorRegion outside that declaration's
 * scope.
 *
 * All buffers that appear in TensorLoad or BufferStore must have a prior declaration.
 */
class UndefinedBufferVerifier : public Verifier<UndefinedBufferVerifier> {
 public:
  using Verifier::Verifier;

 private:
  using Verifier::Visit;

  void Visit(const PrimFunc& prim_func, AccessPath path) override {
    Verifier::Visit(prim_func, path);
    // Clear per-function state (buffers should not cross function boundaries).
    currently_defined_.clear();
    previously_defined_.clear();
  }

  void EnterDef(const BufferVar& buffer, AccessPath path) override {
    // Call the base class to visit buffer's internal vars (shape, strides, etc.)
    Verifier::EnterDef(buffer, path);
    currently_defined_.insert({buffer, path});
  }

  void ExitDef(const BufferVar& buffer, AccessPath path) override {
    auto active_def = currently_defined_.find(buffer);
    if (active_def != currently_defined_.end()) {
      currently_defined_.erase(active_def);
    }
    previously_defined_.insert({buffer, path});
  }

  void VisitBufferUse(const BufferVar& buffer, AccessPath path) override {
    bool is_declared = currently_defined_.count(buffer);
    bool was_declared = previously_defined_.count(buffer);

    if (was_declared && !is_declared) {
      // BufferVar was previously declared but is now out of scope — always an error.
      auto prev_def = previously_defined_.find(buffer);
      Verify(false) << "TIR is ill-formed: buffer " << buffer.name() << " is used at " << path
                    << " but its declaration is no longer in-scope. "
                    << "It was declared at " << prev_def->second << ".";
    } else if (!is_declared && !was_declared) {
      // BufferVar was never declared — error.
      Verify(false) << "TIR is ill-formed: buffer " << buffer.name() << " is used at " << path
                    << " without a prior DeclBuffer or other declaration.";
    }
    // BufferVar fields are visited at definition site (EnterDef), not here.
    Verifier::VisitBufferUse(buffer, path);
  }

  // Buffers defined in the currently-visited scope.
  std::unordered_map<BufferVar, AccessPath, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      currently_defined_;
  // Buffers that were previously defined and are now out of scope.
  std::unordered_map<BufferVar, AccessPath, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      previously_defined_;
};

/*! \brief Verify the asserted type of each tirx buffer load. */
class TensorLoadTypeVerifier : public Verifier<TensorLoadTypeVerifier> {
 public:
  using Verifier::Verifier;

 private:
  void Dispatch_(const TensorLoadNode* op, AccessPath path) override {
    auto buffer = op->source.as<BufferVar>();
    auto valid_source = Verify(buffer.has_value());
    valid_source << "TypeError: TIR TensorLoad source at " << path->Attr("source")
                 << " must be a BufferVar.";
    if (!buffer.has_value()) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    bool valid_indices = buffer.value()->shape.size() == op->indices.size();
    auto valid_rank = Verify(valid_indices);
    valid_rank << "ValueError: TIR TensorLoad at " << path << " indexes "
               << buffer.value()->shape.size() << "-dimensional buffer " << buffer.value().name()
               << " with " << op->indices.size() << " indices.";
    if (!valid_indices) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    for (size_t i = 0; i + 1 < op->indices.size(); ++i) {
      bool is_scalar = op->indices[i].ty().IsScalar();
      auto valid_index = Verify(is_scalar);
      valid_index << "TypeError: TIR TensorLoad index " << i << " at "
                  << path->Attr("indices")->ArrayItem(i)
                  << " must be scalar because only the final index may be vector-valued.";
      valid_indices = valid_indices && is_scalar;
    }
    if (!valid_indices) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    ffi::Optional<PrimType> index_ty = op->indices.empty()
                                           ? ffi::Optional<PrimType>(buffer.value()->dtype)
                                           : op->indices.back().ty().as<PrimType>();
    AccessPath final_index_path = op->indices.empty()
                                      ? path->Attr("indices")
                                      : path->Attr("indices")->ArrayItem(op->indices.size() - 1);
    auto valid_index_type = Verify(index_ty.has_value());
    valid_index_type << "TypeError: TIR TensorLoad final index at " << final_index_path
                     << " must have a primitive type.";
    if (!index_ty.has_value()) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    bool scalable_compatible = op->indices.empty() || !(buffer.value()->dtype.IsScalableVector() &&
                                                        index_ty.value().IsScalableVector());
    auto valid_scalability = Verify(scalable_compatible);
    valid_scalability << "TypeError: TIR TensorLoad at " << path
                      << " cannot combine a scalable buffer dtype with a scalable index.";
    if (!scalable_compatible) {
      Visit(op->indices, path->Attr("indices"));
      return;
    }

    TensorLoad expected = BufferLoad(buffer.value(), op->indices, op->span);
    ffi::Optional<PrimType> asserted_ty = op->ty.as<PrimType>();
    ffi::Optional<PrimType> expected_ty = expected->ty.as<PrimType>();
    auto valid_type = Verify(asserted_ty.has_value() && expected_ty.has_value() &&
                             asserted_ty.value() == expected_ty.value());
    valid_type << "TypeError: TIR TensorLoad at " << path << " asserts result type " << op->ty
               << ", but its source and indices imply " << expected->ty << ".";
    TIRVisitorWithPath::Dispatch_(op, path);
  }
};

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
  for (auto verify : WellFormedExtensions()) {
    if (!verify(func, assert_mode)) return false;
  }

  if (!UndefinedVarVerifier::Verify(func, assert_mode)) return false;

  if (!UndefinedBufferVerifier::Verify(func, assert_mode)) return false;

  if (!TensorLoadTypeVerifier::Verify(func, assert_mode)) return false;

  // TODO(Siyuan): add more checks here.
  return true;
}

bool VerifyWellFormed(const IRModule& mod, bool assert_mode) {
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto prim_func = base_func.as<PrimFunc>()) {
      bool res = VerifyWellFormed(prim_func.value(), assert_mode);
      if (!res) {
        return false;
      }
    }
  }

  if (!UndefinedVarVerifier::Verify(mod, assert_mode)) return false;

  if (!UndefinedBufferVerifier::Verify(mod, assert_mode)) return false;

  if (!TensorLoadTypeVerifier::Verify(mod, assert_mode)) return false;

  return true;
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
