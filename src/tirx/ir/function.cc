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
 * \file src/tirx/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/function.h>

namespace tvm {
namespace tirx {
namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> FunctionVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), ty (derived from the function signature)
  const FunctionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(self->params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret_type));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FunctionMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), ty (derived from the function signature)
  const FunctionNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->params);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_type,
                                    mutator->MutateExpected(self->ret_type));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_params.UnchangedOrSameAs(self->params) &&
      mapped_ret_type.UnchangedOrSameAs(self->ret_type) &&
      (mapped_body.IsUnchanged() || ffi::AnyView(mapped_body).same_as(self->body))) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<FunctionNode> copy = ffi::make_object<FunctionNode>(*self);
  copy->params = std::move(mapped_params).ValueOrUnchanged(std::move(copy->params));
  copy->ret_type = std::move(mapped_ret_type).ValueOrUnchanged(std::move(copy->ret_type));
  if (!mapped_body.IsUnchanged()) {
    auto replacement = std::move(mapped_body).ValueUnchecked();
    copy->body = replacement.has_value()
                     ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                     : std::nullopt;
  }
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FunctionMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), ty (derived from the function signature)
  FunctionNode* self = const_cast<FunctionNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FunctionNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->params,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Type>, mapped_ret_type,
      mutator->MutateExpected(self->ret_type, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Stmt>>, mapped_body,
                                    mutator->MutateExpected(self->body, ffi::InplaceMode::kAllow));
  if (!mapped_params.UnchangedOrSameAs(self->params)) {
    self->params = std::move(mapped_params).ValueUnchecked();
  }
  if (!mapped_ret_type.UnchangedOrSameAs(self->ret_type)) {
    self->ret_type = std::move(mapped_ret_type).ValueUnchecked();
  }
  if (!(mapped_body.IsUnchanged() || ffi::AnyView(mapped_body).same_as(self->body))) {
    auto replacement = std::move(mapped_body).ValueUnchecked();
    self->body = replacement.has_value()
                     ? ffi::Optional<SeqStmt>(SeqStmt(std::move(replacement).value()))
                     : std::nullopt;
  }
  return ffi::Unchanged();
}

}  // namespace

// Get the function type of a Function
Function::Function(ffi::Array<tvm::Var> params, ffi::Optional<SeqStmt> body, Type ret_type,
                   DictAttrs attrs, ffi::Optional<Location> loc)
    : BaseFunc(ffi::UnsafeInit{}) {
  if (ret_type.as<MissingType>().has_value()) {
    ret_type = VoidType();
  }

  auto n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->attrs = std::move(attrs);
  n->loc = loc.value_or(Location());
  n->ty = n->func_type_annotation();
  data_ = std::move(n);
}

Function RenewDef(Function func) {
  // Seed original definitions so identities created by structural hooks are preserved.
  ffi::Map<Var, Var> definition_remap;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(func,
                                                  [&](const Var& var, TVMFFIDefRegionKind kind) {
                                                    if (kind != kTVMFFIDefRegionKindNone)
                                                      definition_remap.Set(var, var);
                                                    return ffi::WalkResult::Advance();
                                                  });
  return ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
             func,
             [remap = std::move(definition_remap)](const Var& var,
                                                   TVMFFIDefRegionKind kind) mutable {
               auto mapped = remap.Get(var);
               if (!mapped.has_value()) return var;
               if (!mapped.value().same_as(var)) return mapped.value();
               if (kind == kTVMFFIDefRegionKindNone) return var;
               Var fresh(var->name, var->ty, var->loc);
               remap.Set(var, fresh);
               return fresh;
             },
             [](const Function& mapped) {
               return Function(mapped->params, mapped->body, mapped->ret_type, mapped->attrs,
                               mapped->loc);
             })
      .as_or_throw<Function>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FunctionNode::RegisterReflection();
  refl::TypeAttrDef<FunctionNode>()
      .attr(refl::type_attr::kStructuralVisit, ffi::FStructuralVisit::FromNative<&FunctionVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&FunctionMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&FunctionMaybeInplaceMutate>());

  refl::GlobalDef().def(
      "tirx.Function",
      [](ffi::Array<tvm::Var> params, ffi::Optional<SeqStmt> body, Type ret_type, DictAttrs attrs,
         ffi::Optional<Location> loc) { return Function(params, body, ret_type, attrs, loc); });
  refl::GlobalDef().def("tirx.RenewDef", RenewDef);
}

FuncType FunctionNode::func_type_annotation() const {
  ffi::Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(param->ty);
  }
  return FuncType(param_types, ret_type);
}

}  // namespace tirx
}  // namespace tvm
