/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file lower_tirx_opaque.cc
 * \brief Lower opaque constructs in TIRX programs. This is the tirx-specific
 *        counterpart of s_tirx::LowerOpaqueBlock, handling only the non-SBlock
 *        parts: AllocTensor lowering, For(thread_binding) → RegionStmt(launch_thread),
 *        unit loop elimination, and loop policy inheritance.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>

#include "ir_utils.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

/*!
 * \brief Lower opaque constructs for TIRX: AllocTensor, thread bindings, unit loops.
 *
 * Unlike s_tirx::LowerOpaqueBlock, this pass does NOT handle SBlock/SBlockRealize,
 * since TIRX programs do not contain SBlock nodes.
 */
class TIRxOpaqueLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  static Stmt Rewrite(Stmt body) {
    return ffi::make_object<TIRxOpaqueLower>()
        ->Mutate(body, InplaceMode::kAllow)
        .ValueOrUnchanged(body);
  }

 private:
  struct UnrollPolicy {
    std::optional<ffi::Any> auto_unroll_max_step;
    std::optional<ffi::Any> unroll_explicit;
  };

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->Mutate(op->min, inplace_mode).ValueOrUnchanged(op->min);
    PrimExpr extent = this->Mutate(op->extent, inplace_mode).ValueOrUnchanged(op->extent);
    auto step = this->Mutate(op->step, inplace_mode)
                    .as_or_throw<UnchangedOr<ffi::Optional<PrimExpr>>>()
                    .ValueOrUnchanged(op->step);
    ffi::Any previous_remap = VarRemapGet(op->loop_var);
    PrimVar launch_var(ffi::UnsafeInit{});
    if (tvm::tirx::GetThreadBinding(op).has_value()) {
      TVM_FFI_ICHECK(IsZero(min)) << "Thread binding must have zero minimum";
      launch_var = PrimVar(op->loop_var->name, extent.ty());
      VarRemapSet(op->loop_var, prim::cast(op->loop_var.ty(), launch_var));
    } else if (IsOne(extent) && op->annotations.empty()) {
      VarRemapSet(op->loop_var, prim::cast(op->loop_var.ty(), min));
    }

    // Keep policy on surviving descendants when this owner is lowered away.
    auto parent_policy = unroll_policy_.Current();
    auto annotations = op->annotations;
    Stmt body = unroll_policy_.WithNewScope([&]() {
      unroll_policy_.Current() = parent_policy;
      UpdateUnrollPolicy(op->annotations);
      const auto& policy = unroll_policy_.Current();
      if (policy.auto_unroll_max_step.has_value()) {
        annotations.Set(tirx::attr::auto_unroll_max_step, policy.auto_unroll_max_step.value());
      }
      if (policy.unroll_explicit.has_value()) {
        annotations.Set(tirx::attr::unroll_explicit, policy.unroll_explicit.value());
      }
      // Rewrite annotations before visiting body-local definitions.
      annotations = this->Mutate(annotations, inplace_mode)
                        .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>()
                        .ValueOrUnchanged(annotations);
      return this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    });
    VarRemapSet(op->loop_var, previous_remap);

    // Step 2. Create the lowered loop or launch region.
    if (tvm::tirx::GetThreadBinding(op).has_value()) {
      // Case 1. Thread binding → RegionStmt(launch_thread)
      TVM_FFI_ICHECK(!op->annotations.count("loop_partition_hint") ||
                     op->annotations.at("loop_partition_hint") == nullptr)
          << "Run LoopPartition before opaque lowering of a thread-binding loop with "
             "loop_partition_hint";
      TVM_FFI_ICHECK(tvm::tirx::GetThreadBinding(op).has_value());
    } else if (IsOne(extent) && op->annotations.empty()) {
      // Case 2. Unit loop elimination
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body),
                 FilterAnnotations(annotations), step);
    }
    if (tvm::tirx::GetThreadBinding(op).has_value()) {
      return RegionStmt(tirx::launch_thread_op(),
                        {StringImm(tvm::tirx::GetThreadBinding(op).value()), extent}, {launch_var},
                        DictAttrs(), body, {}, op->span);
    }
    return body;
  }

  void UpdateUnrollPolicy(const ffi::Map<ffi::String, ffi::Any>& annotations) {
    auto& policy = unroll_policy_.Current();
    if (auto value = annotations.Get(tirx::attr::auto_unroll_max_step);
        value.has_value() && value.value() != nullptr) {
      policy.auto_unroll_max_step = value.value();
    }
    if (auto value = annotations.Get(tirx::attr::unroll_explicit);
        value.has_value() && value.value() != nullptr) {
      policy.unroll_explicit = value.value();
    }
  }

  // Effective policy is materialized on each surviving loop, preserving nested overrides.
  ScopeStack<UnrollPolicy> unroll_policy_;

  // Null optional policies do not override inherited values or reach code generation.
  ffi::Map<ffi::String, ffi::Any> FilterAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations) {
    ffi::Map<ffi::String, ffi::Any> preserved;
    for (const auto& [key, value] : annotations) {
      if ((key == tirx::attr::auto_unroll_max_step || key == tirx::attr::unroll_explicit ||
           key == "pragma_unroll") &&
          value == nullptr) {
        continue;
      }
      preserved.Set(key, value);
    }
    return preserved;
  }
};

namespace transform {

Pass LowerTIRxOpaque() {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto fptr = f.CopyOnWrite();
    fptr->body = TIRxOpaqueLower::Rewrite(std::move(fptr->body).value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.LowerTIRxOpaque");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerTIRxOpaque", LowerTIRxOpaque);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
