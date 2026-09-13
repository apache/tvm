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
 * \file src/ir/type.cc
 * \brief Common type system AST nodes throughout the IR.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/type.h>

#include <cstdint>
#include <unordered_map>

namespace tvm {

namespace {

DLDataType ScalableVectorDType(DLDataTypeCode code, int bits, int lanes) {
  TVM_FFI_ICHECK_GT(lanes, 1) << "Invalid value for vscale factor " << lanes;
  TVM_FFI_ICHECK_LT(lanes, 32768);
  return DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                    static_cast<uint16_t>(-lanes)};
}

uint32_t PackDataTypeKey(DLDataType dtype) {
  return (static_cast<uint32_t>(dtype.code) << 24) | (static_cast<uint32_t>(dtype.bits) << 16) |
         static_cast<uint32_t>(dtype.lanes);
}

ffi::ObjectPtr<PrimTypeNode> GetCachedPrimTypeNode(DLDataType dtype) {
  thread_local std::unordered_map<uint32_t, ffi::ObjectPtr<PrimTypeNode>> cache;
  uint32_t key = PackDataTypeKey(dtype);
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }

  ffi::ObjectPtr<PrimTypeNode> node = ffi::make_object<PrimTypeNode>();
  node->dtype = dtype;
  return cache.emplace(key, std::move(node)).first->second;
}

// Structural traversal hooks

TVMFFIAny TypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // Type::Missing() is the only concrete TypeNode value; span is ignored debug metadata.
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny OpaqueTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // OpaqueType is a field-less construction-time marker; span is ignored debug metadata.
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny OpaqueTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny OpaqueTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

int64_t PrimTypeAnyHash(const ffi::Any& src) {
  return static_cast<int64_t>(PackDataTypeKey(src.cast<PrimType>()->dtype));
}

bool PrimTypeAnyEqual(const ffi::Any& lhs, const ffi::Any& rhs) {
  return lhs.cast<PrimType>()->dtype == rhs.cast<PrimType>()->dtype;
}

TVMFFIAny PrimTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  // dtype is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny PrimTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // dtype is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny PrimTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  // dtype is a constant: reflected for StructuralEqual/Hash,
  // not traversed by the visitor/mutator contract.
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny PointerTypeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: storage_scope (scalar)
  const PointerTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PointerTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->element_type));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny PointerTypeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: storage_scope (scalar)
  const PointerTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PointerTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_element_type,
                                    mutator->MutateExpected(self->element_type));
  if (mapped_element_type.UnchangedOrSameAs(self->element_type)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<PointerTypeNode> copy = ffi::make_object<PointerTypeNode>(*self);
  copy->element_type =
      std::move(mapped_element_type).ValueOrUnchanged(std::move(copy->element_type));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny PointerTypeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                        ffi::AnyView value) noexcept {
  // skips: storage_scope (scalar)
  PointerTypeNode* self = const_cast<PointerTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const PointerTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<Type>, mapped_element_type,
      mutator->MaybeInplaceMutateIfUniqueExpected(self->element_type));
  if (!mapped_element_type.UnchangedOrSameAs(self->element_type)) {
    self->element_type = std::move(mapped_element_type).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny FuncTypeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const FuncTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->arg_types));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret_type));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny FuncTypeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const FuncTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, mapped_arg_types,
                                    mutator->MutateExpected(self->arg_types));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_type,
                                    mutator->MutateExpected(self->ret_type));
  if (mapped_arg_types.UnchangedOrSameAs(self->arg_types) &&
      mapped_ret_type.UnchangedOrSameAs(self->ret_type)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<FuncTypeNode> copy = ffi::make_object<FuncTypeNode>(*self);
  copy->arg_types = std::move(mapped_arg_types).ValueOrUnchanged(std::move(copy->arg_types));
  copy->ret_type = std::move(mapped_ret_type).ValueOrUnchanged(std::move(copy->ret_type));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny FuncTypeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  FuncTypeNode* self = const_cast<FuncTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, mapped_arg_types,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->arg_types));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret_type,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ret_type));
  if (!mapped_arg_types.UnchangedOrSameAs(self->arg_types)) {
    self->arg_types = std::move(mapped_arg_types).ValueUnchecked();
  }
  if (!mapped_ret_type.UnchangedOrSameAs(self->ret_type)) {
    self->ret_type = std::move(mapped_ret_type).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TupleTypeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  const TupleTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->fields));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TupleTypeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  const TupleTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, mapped_fields,
                                    mutator->MutateExpected(self->fields));
  if (mapped_fields.UnchangedOrSameAs(self->fields)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TupleTypeNode> copy = ffi::make_object<TupleTypeNode>(*self);
  copy->fields = std::move(mapped_fields).ValueOrUnchanged(std::move(copy->fields));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TupleTypeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                      ffi::AnyView value) noexcept {
  TupleTypeNode* self = const_cast<TupleTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Type>>, mapped_fields,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->fields));
  if (!mapped_fields.UnchangedOrSameAs(self->fields)) {
    self->fields = std::move(mapped_fields).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TensorMapTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TensorMapTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TensorMapTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

Type Type::Missing() {
  static Type missing = []() {
    Type type(ffi::UnsafeInit{});
    type.data_ = ffi::make_object<TypeNode>();
    return type;
  }();
  return missing;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TypeNode::RegisterReflection();
  refl::TypeAttrDef<TypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TypeMaybeInplaceMutate));
  refl::GlobalDef()
      .def("ir.TypeMissing", []() { return Type::Missing(); })
      .def("ir.TypeIsMissing", [](Type type) { return type.IsMissing(); });
}

bool Type::IsMissing() const { return this->same_as(Type::Missing()); }

OpaqueType::OpaqueType() : Type(ffi::UnsafeInit{}) { data_ = ffi::make_object<OpaqueTypeNode>(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  OpaqueTypeNode::RegisterReflection();
  refl::TypeAttrDef<OpaqueTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&OpaqueTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&OpaqueTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&OpaqueTypeMaybeInplaceMutate));

  refl::GlobalDef().def("ir.OpaqueType", []() { return OpaqueType(); });
}

// PrimType
PrimType::PrimType(DLDataType dtype) : Type(ffi::UnsafeInit{}) {
  bool is_opaque_handle = dtype.code == static_cast<uint8_t>(DLDataTypeCode::kDLOpaqueHandle);
  bool is_void = is_opaque_handle && dtype.bits == 0 && dtype.lanes == 0;
  TVM_FFI_CHECK(!is_opaque_handle || is_void, TypeError)
      << "PrimType cannot represent an opaque pointer; use PointerType::VoidPointerTy()";
  data_ = GetCachedPrimTypeNode(dtype);
}

PrimType::PrimType(DLDataTypeCode code, int bits, int lanes)
    : PrimType(DLDataType{static_cast<uint8_t>(code), static_cast<uint8_t>(bits),
                          static_cast<uint16_t>(lanes)}) {}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PrimTypeNode::RegisterReflection();
  refl::TypeAttrDef<PrimTypeNode>()
      .attr(refl::type_attr::kAnyHash, reinterpret_cast<void*>(&PrimTypeAnyHash))
      .attr(refl::type_attr::kAnyEqual, reinterpret_cast<void*>(&PrimTypeAnyEqual))
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&PrimTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&PrimTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&PrimTypeMaybeInplaceMutate));

  refl::GlobalDef().def("ir.PrimType", [](DLDataType dtype) { return PrimType(dtype); });
}

PrimType PrimType::Int(int bits, int lanes) {
  if (lanes == 1) {
    if (bits == 32) {
      thread_local PrimType i32_ty(DLDataType{kDLInt, 32, 1});
      return i32_ty;
    }
    if (bits == 64) {
      thread_local PrimType i64_ty(DLDataType{kDLInt, 64, 1});
      return i64_ty;
    }
  }
  return PrimType(DLDataType{kDLInt, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::UInt(int bits, int lanes) {
  return PrimType(DLDataType{kDLUInt, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Float(int bits, int lanes) {
  if (bits == 32 && lanes == 1) {
    thread_local PrimType f32_ty(DLDataType{kDLFloat, 32, 1});
    return f32_ty;
  }
  return PrimType(DLDataType{kDLFloat, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::BFloat(int bits, int lanes) {
  return PrimType(DLDataType{kDLBfloat, static_cast<uint8_t>(bits), static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Bool(int lanes) {
  if (lanes == 1) {
    thread_local PrimType bool_ty(DLDataType{kDLBool, 8, 1});
    return bool_ty;
  }
  return PrimType(DLDataType{kDLBool, 8, static_cast<uint16_t>(lanes)});
}

PrimType PrimType::Void() { return PrimType(DLDataType{kDLOpaqueHandle, 0, 0}); }

PrimType PrimType::ScalableVector(DLDataTypeCode code, int bits, int lanes) {
  return PrimType(ScalableVectorDType(code, bits, lanes));
}

// PointerType
PointerType::PointerType(Type element_type, ffi::String storage_scope) : Type(ffi::UnsafeInit{}) {
  TVM_FFI_ICHECK(!element_type.IsMissing()) << "PointerType element_type cannot be Type::Missing()";
  ffi::ObjectPtr<PointerTypeNode> n = ffi::make_object<PointerTypeNode>();
  if (storage_scope.empty()) {
    n->storage_scope = "global";
  } else {
    n->storage_scope = std::move(storage_scope);
  }
  n->element_type = std::move(element_type);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  PointerTypeNode::RegisterReflection();
  refl::TypeAttrDef<PointerTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&PointerTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&PointerTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&PointerTypeMaybeInplaceMutate));

  refl::GlobalDef().def("ir.PointerType", [](Type element_type, ffi::String storage_scope = "") {
    return PointerType(element_type, storage_scope);
  });
}

PointerType PointerType::VoidPointerTy(ffi::String storage_scope) {
  return PointerType(PrimType::Void(), std::move(storage_scope));
}

FuncType::FuncType(tvm::ffi::Array<Type> arg_types, Type ret_type, Span span)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<FuncTypeNode> n = ffi::make_object<FuncTypeNode>();
  n->arg_types = std::move(arg_types);
  n->ret_type = std::move(ret_type);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  FuncTypeNode::RegisterReflection();
  refl::TypeAttrDef<FuncTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&FuncTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&FuncTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&FuncTypeMaybeInplaceMutate));

  refl::GlobalDef().def("ir.FuncType", [](tvm::ffi::Array<Type> arg_types, Type ret_type) {
    return FuncType(arg_types, ret_type);
  });
}

TupleType::TupleType(ffi::Array<Type> fields, Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TupleTypeNode> n = ffi::make_object<TupleTypeNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TupleTypeNode::RegisterReflection();
  refl::TypeAttrDef<TupleTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TupleTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TupleTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TupleTypeMaybeInplaceMutate));

  refl::GlobalDef().def("ir.TupleType",
                        [](ffi::Array<Type> fields, Span span) { return TupleType(fields, span); });
}

TupleType TupleType::Empty() { return TupleType(ffi::Array<Type>()); }

TensorMapType::TensorMapType(Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorMapTypeNode> n = ffi::make_object<TensorMapTypeNode>();
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TensorMapTypeNode::RegisterReflection();
  refl::TypeAttrDef<TensorMapTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TensorMapTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TensorMapTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TensorMapTypeMaybeInplaceMutate));

  refl::GlobalDef().def("ir.TensorMapType", [](Span span) { return TensorMapType(span); });
}

}  // namespace tvm
