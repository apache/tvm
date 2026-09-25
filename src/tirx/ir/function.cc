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
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

tvm::Type InferType(const PrimFunc& prim_func) {
  ffi::Array<tvm::Type> params;
  for (const auto& param : prim_func->params) {
    tvm::Type param_ty = [&]() -> tvm::Type {
      if (param->ty.as<BufferTypeNode>()) {
        BufferVar buf(param);
        With<prim::OpConstFoldScope> normalize_shape(true);
        relax::ShapeExpr shape(
            buf->shape.Map([](PrimExpr dim) { return cast(PrimType::Int(64), dim); }));
        return relax::TensorType(shape, buf->dtype);
      }

      // A pointer parameter without a buffer annotation is an opaque runtime
      // object from Relax's perspective (for example, a DLTensor*).  Keep the
      // same Relax-facing wildcard semantics that opaque handle parameters had
      // before pointers became exact IR types.
      if (param->ty.as<PointerTypeNode>()) {
        return relax::AnyType();
      }

      return param->ty;
    }();
    params.push_back(param_ty);
  }

  tvm::Type ret = [&]() -> tvm::Type {
    if (const auto* prim = prim_func->ret_type.as<PrimTypeNode>()) {
      return tvm::PrimType(prim->dtype);
    } else if (IsVoidType(prim_func->ret_type)) {
      return relax::TupleType(ffi::Array<tvm::Type>{});
    } else {
      return relax::AnyType();
    }
  }();

  bool purity = prim_func->body.defined() ? s_tir::IsPureFunction(prim_func) : false;

  return relax::FuncType(params, ret, purity);
}

TVMFFIAny PrimFuncVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), ty (derived by InferType)
  const PrimFuncNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PrimFuncNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(self->params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret_type));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->body));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny PrimFuncMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: attrs (metadata), ty (derived by InferType)
  const PrimFuncNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PrimFuncNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->params);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_type,
                                    mutator->MutateExpected(self->ret_type));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
                                    mutator->MutateExpected(self->body));
  if (mapped_params.UnchangedOrSameAs(self->params) &&
      mapped_ret_type.UnchangedOrSameAs(self->ret_type) &&
      mapped_body.UnchangedOrSameAs(self->body)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<PrimFuncNode> copy = ffi::make_object<PrimFuncNode>(*self);
  copy->params = std::move(mapped_params).ValueOrUnchanged(std::move(copy->params));
  copy->ret_type = std::move(mapped_ret_type).ValueOrUnchanged(std::move(copy->ret_type));
  copy->body = std::move(mapped_body).ValueOrUnchanged(std::move(copy->body));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny PrimFuncMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  // skips: attrs (metadata), ty (derived by InferType)
  PrimFuncNode* self = const_cast<PrimFuncNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PrimFuncNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Var>>, mapped_params,
                                    mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
                                      return mutator->MutateExpected(self->params,
                                                                     ffi::InplaceMode::kAllow);
                                    }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Type>, mapped_ret_type,
      mutator->MutateExpected(self->ret_type, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Stmt>, mapped_body,
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
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(ffi::Array<tirx::Var> params, Stmt body, Type ret_type, DictAttrs attrs,
                   Span span) {
  if (ret_type.IsMissing()) {
    ret_type = VoidType();
  }

  auto n = ffi::make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->attrs = std::move(attrs);
  n->ty = relax::FuncType::OpaqueFunc();
  n->span = std::move(span);
  data_ = std::move(n);

  (*this)->ty = InferType(*this);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PrimFuncNode::RegisterReflection();
  refl::TypeAttrDef<PrimFuncNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&PrimFuncVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&PrimFuncMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&PrimFuncMaybeInplaceMutate));

  refl::GlobalDef().def("tirx.PrimFunc",
                        [](ffi::Array<tirx::Var> params, Stmt body, Type ret_type, DictAttrs attrs,
                           Span span) { return PrimFunc(params, body, ret_type, attrs, span); });
}

FuncType PrimFuncNode::func_type_annotation() const {
  ffi::Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(param->ty);
  }
  return FuncType(param_types, ret_type);
}

}  // namespace tirx
}  // namespace tvm
