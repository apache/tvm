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
 * \file lambdaexpr.cc
 * \brief Implementation of LambdaExpr, a reified lambda used by tile primitive ops.
 */

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/tile_primitive.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

namespace {

ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> LambdaExprVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LambdaExprNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return visitor->VisitExpected(self->vars); }));
  return visitor->VisitExpected(self->pred);
}

ffi::Expected<ffi::UnchangedOr<ffi::Any>> LambdaExprMutate(ffi::StructuralMutatorObj* mutator,
                                                           ffi::AnyView value) noexcept {
  const auto* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LambdaExprNode>(value);
  // Parameters shadow substitutions from the surrounding expression. Save and restore
  // their remaps so a binder rewrite stays local to this lambda's body.
  std::vector<ffi::Any> saved;
  for (const Var& var : self->vars) {
    auto previous = mutator->VarRemapGetExpected(var);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(previous);
    saved.push_back(std::move(previous).value());
  }
  auto result = [&]() -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    for (const Var& var : self->vars) {
      auto cleared = mutator->VarRemapSetExpected(var, nullptr);
      TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(cleared);
    }
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ty,
                                      mutator->MutateExpected(self->ty));
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_vars,
                                      mutator->WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&]() {
                                        return mutator->MutateExpected(self->vars);
                                      }));
    auto vars = std::move(mapped_vars).ValueOrUnchanged(self->vars);
    for (size_t i = 0; i < vars.size(); ++i) {
      auto remap = mutator->VarRemapSetExpected(self->vars[i], vars[i]);
      TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap);
    }
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<PrimExpr>, mapped_pred,
                                      mutator->MutateExpected(self->pred));
    if (vars.same_as(self->vars) && mapped_pred.UnchangedOrSameAs(self->pred) &&
        mapped_ty.UnchangedOrSameAs(self->ty)) {
      return ffi::Unchanged();
    }
    auto copy = ffi::make_object<LambdaExprNode>(*self);
    copy->vars = std::move(vars);
    copy->pred = std::move(mapped_pred).ValueOrUnchanged(self->pred);
    copy->ty = std::move(mapped_ty).ValueOrUnchanged(self->ty);
    return ffi::Any(std::move(copy));
  }();
  for (size_t i = 0; i < self->vars.size(); ++i) {
    auto restored = mutator->VarRemapSetExpected(self->vars[i], saved[i]);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(restored);
  }
  return result;
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  LambdaExprNode::RegisterReflection();
  refl::TypeAttrDef<LambdaExprNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&LambdaExprVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&LambdaExprMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&LambdaExprMutate>());
}

PrimExpr LambdaExprNode::Apply(const ffi::Array<PrimExpr>& indices) const {
  TVM_FFI_ICHECK_EQ(indices.size(), vars.size());

  ffi::Map<Var, PrimExpr> vmap;

  for (size_t i = 0; i < vars.size(); i++) {
    vmap.Set(vars[i], indices[i]);
  }
  auto f_substitute = [&vmap](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = vmap.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(std::move(pred), f_substitute)
      .as_or_throw<PrimExpr>();
}

LambdaExpr::LambdaExpr(ffi::Array<Var> vars, PrimExpr pred) : Expr(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<LambdaExprNode>(std::move(pred));
  n->vars = std::move(vars);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.LambdaExpr",
                        [](ffi::Array<Var> vars, PrimExpr pred) { return LambdaExpr(vars, pred); });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.LambdaExprApply", [](LambdaExpr pred, ffi::Array<PrimExpr> indices) {
    return pred->Apply(indices);
  });
}

}  // namespace tirx
}  // namespace tvm
