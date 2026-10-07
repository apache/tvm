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
  // skips: attrs (metadata), ty (derived by RefreshType)
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
  // skips: attrs (metadata), ty (derived by RefreshType)
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
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<FunctionNode> copy = ffi::make_object<FunctionNode>(*self);
  copy->params = std::move(mapped_params).ValueOrUnchanged(std::move(copy->params));
  copy->ret_type = std::move(mapped_ret_type).ValueOrUnchanged(std::move(copy->ret_type));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  copy->RefreshType();
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> FunctionMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), ty (derived by RefreshType)
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
  if (!mapped_body.UnchangedOrSameAs(self->body)) {
    self->body = std::move(mapped_body).ValueUnchecked();
  }
  // A parameter's type may change in place while the parameter and array keep their identities.
  self->RefreshType();
  return ffi::Unchanged();
}

}  // namespace

// Get the function type of a Function
Function::Function(ffi::Array<tirx::Var> params, ffi::Optional<Stmt> body, Type ret_type,
                   DictAttrs attrs, Span span)
    : BaseFunc(ffi::UnsafeInit{}) {
  if (ret_type.as<MissingType>().has_value()) {
    ret_type = VoidType();
  }

  auto n = ffi::make_object<FunctionNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->attrs = std::move(attrs);
  n->span = std::move(span);
  n->RefreshType();
  data_ = std::move(n);
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

  refl::GlobalDef().def("tirx.Function", [](ffi::Array<tirx::Var> params, ffi::Optional<Stmt> body,
                                            Type ret_type, DictAttrs attrs, Span span) {
    return Function(params, body, ret_type, attrs, span);
  });
}

FuncType FunctionNode::func_type_annotation() const {
  ffi::Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(param->ty);
  }
  return FuncType(param_types, ret_type);
}

void FunctionNode::RefreshType() const {
  const auto* signature = ty.as<FuncTypeNode>();
  bool current = signature && signature->ret_type.same_as(ret_type) &&
                 signature->arg_types.size() == params.size();
  for (size_t i = 0; current && i < params.size(); ++i) {
    current = signature->arg_types[i].same_as(params[i]->ty);
  }
  if (!current) {
    ty = func_type_annotation();
  }
}

}  // namespace tirx
}  // namespace tvm
