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
 * \file src/relax/ir/dependent_type.cc
 * \brief Relax type nodes.
 */
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_functor.h>

namespace tvm {
namespace relax {

namespace {

TVMFFIAny AnyTypeVisit(ffi::StructuralVisitorObj*, ffi::AnyView) noexcept {
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny AnyTypeMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny AnyTypeMaybeInplaceMutate(ffi::StructuralMutatorObj*, ffi::AnyView) noexcept {
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny ShapeTypeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const ShapeTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->values));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny ShapeTypeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const ShapeTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<ffi::Array<PrimExpr>>>,
                                    mapped_values, mutator->MutateExpected(self->values));
  if (mapped_values.UnchangedOrSameAs(self->values)) return ffi::Unchanged().CopyToTVMFFIAny();
  ffi::ObjectPtr<ShapeTypeNode> copy = ffi::make_object<ShapeTypeNode>(*self);
  copy->values = std::move(mapped_values).ValueOrUnchanged(std::move(copy->values));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny ShapeTypeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                      ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  ShapeTypeNode* self = const_cast<ShapeTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ShapeTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<ffi::Array<PrimExpr>>>,
                                    mapped_values,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->values));
  if (!mapped_values.UnchangedOrSameAs(self->values)) {
    self->values = std::move(mapped_values).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny TensorTypeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const TensorTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->shape));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->dtype));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->vdevice));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TensorTypeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  const TensorTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Expr>>, mapped_shape,
                                    mutator->MutateExpected(self->shape));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimType>>, mapped_dtype,
                                    mutator->MutateExpected(self->dtype));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<VDevice>>, mapped_vdevice,
                                    mutator->MutateExpected(self->vdevice));
  if (mapped_shape.UnchangedOrSameAs(self->shape) && mapped_dtype.UnchangedOrSameAs(self->dtype) &&
      mapped_vdevice.UnchangedOrSameAs(self->vdevice)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TensorTypeNode> copy = ffi::make_object<TensorTypeNode>(*self);
  copy->shape = std::move(mapped_shape).ValueOrUnchanged(std::move(copy->shape));
  copy->dtype = std::move(mapped_dtype).ValueOrUnchanged(std::move(copy->dtype));
  copy->vdevice = std::move(mapped_vdevice).ValueOrUnchanged(std::move(copy->vdevice));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TensorTypeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                       ffi::AnyView value) noexcept {
  // skips: ndim (scalar)
  TensorTypeNode* self = const_cast<TensorTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<Expr>>, mapped_shape,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->shape));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<PrimType>>, mapped_dtype,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->dtype));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Optional<VDevice>>, mapped_vdevice,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->vdevice));
  if (!mapped_shape.UnchangedOrSameAs(self->shape)) {
    self->shape = std::move(mapped_shape).ValueUnchecked();
  }
  if (!mapped_dtype.UnchangedOrSameAs(self->dtype)) {
    self->dtype = std::move(mapped_dtype).ValueUnchecked();
  }
  if (!mapped_vdevice.UnchangedOrSameAs(self->vdevice)) {
    self->vdevice = std::move(mapped_vdevice).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

TVMFFIAny FuncTypeVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: derive_func (environment-backed callable metadata), purity (scalar)
  const FuncTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->WithDefRegionKind(
      kTVMFFIDefRegionKindPattern, [&]() { return visitor->VisitExpected(self->params); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->ret));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny FuncTypeMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: derive_func (environment-backed callable metadata), purity (scalar)
  const FuncTypeNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<ffi::Array<Type>>>, mapped_params,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern,
                                 [&]() { return mutator->MutateExpected(self->params); }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret,
                                    mutator->MutateExpected(self->ret));
  if (mapped_params.UnchangedOrSameAs(self->params) && mapped_ret.UnchangedOrSameAs(self->ret)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<FuncTypeNode> copy = ffi::make_object<FuncTypeNode>(*self);
  copy->params = std::move(mapped_params).ValueOrUnchanged(std::move(copy->params));
  copy->ret = std::move(mapped_ret).ValueOrUnchanged(std::move(copy->ret));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny FuncTypeMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                     ffi::AnyView value) noexcept {
  // skips: derive_func (environment-backed callable metadata), purity (scalar)
  FuncTypeNode* self = const_cast<FuncTypeNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const FuncTypeNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ffi::Optional<ffi::Array<Type>>>, mapped_params,
      mutator->WithDefRegionKind(kTVMFFIDefRegionKindPattern, [&]() {
        return mutator->MaybeInplaceMutateIfUniqueExpected(self->params);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<Type>, mapped_ret,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->ret));
  if (!mapped_params.UnchangedOrSameAs(self->params)) {
    self->params = std::move(mapped_params).ValueUnchecked();
  }
  if (!mapped_ret.UnchangedOrSameAs(self->ret)) {
    self->ret = std::move(mapped_ret).ValueUnchecked();
  }
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  AnyTypeNode::RegisterReflection();
  ShapeTypeNode::RegisterReflection();
  TensorTypeNode::RegisterReflection();
  FuncTypeNode::RegisterReflection();
  refl::TypeAttrDef<AnyTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&AnyTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&AnyTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&AnyTypeMaybeInplaceMutate));
  refl::TypeAttrDef<ShapeTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&ShapeTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&ShapeTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&ShapeTypeMaybeInplaceMutate));
  refl::TypeAttrDef<TensorTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TensorTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TensorTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TensorTypeMaybeInplaceMutate))
      .def("__subscript_expr_realize__",
           [](Expr value,
              ffi::Array<ffi::Variant<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                                 ffi::Optional<PrimExpr>>,
                                      PrimExpr>>
                  slice,
              Span span) -> ffi::ObjectRef {
             TVM_FFI_CHECK_EQ(slice.size(), 1, IndexError)
                 << "A Relax expression requires exactly one index";
             auto index = slice[0].as<PrimExpr>();
             TVM_FFI_CHECK(index.has_value(), TypeError)
                 << "A Relax expression requires a point index";
             const auto* imm = index.value().as<IntImmNode>();
             TVM_FFI_CHECK(imm != nullptr, TypeError)
                 << "A Relax expression requires a constant integer index";
             return TupleGetItem(value, static_cast<int>(imm->value), span);
           });
  refl::TypeAttrDef<FuncTypeNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&FuncTypeVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&FuncTypeMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&FuncTypeMaybeInplaceMutate));
}

AnyType::AnyType(Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<AnyTypeNode> n = ffi::make_object<AnyTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.AnyType", [](Span span) { return AnyType(span); })
      .def("relax.ObjectType", [](Span span) { return AnyType(span); });
}

// Shape
ShapeType::ShapeType(ffi::Array<PrimExpr> values, Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ShapeTypeNode> n = ffi::make_object<ShapeTypeNode>();
  n->ndim = static_cast<int>(values.size());
  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(PrimType::Int(64), value);
    }
    TVM_FFI_ICHECK(value.ty().MatchesElementType(DLDataTypeCode::kDLInt, 64))
        << "the value in ShapeType can only have dtype of int64";
    return value;
  });
  n->span = span;
  data_ = std::move(n);
}

ShapeType::ShapeType(int ndim, Span span) : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<ShapeTypeNode> n = ffi::make_object<ShapeTypeNode>();
  TVM_FFI_ICHECK(ndim >= -1) << "ndim of ShapeType must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.ShapeType", [](ffi::Optional<ffi::Array<PrimExpr>> values, int ndim, Span span) {
        if (values.has_value()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify values and ndim";
          return ShapeType(values.value(), span);
        } else {
          return ShapeType(ndim, span);
        }
      });
}

// Tensor
TensorType::TensorType(Expr shape, ffi::Optional<PrimType> dtype, ffi::Optional<VDevice> vdevice,
                       Span span)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorTypeNode> n = ffi::make_object<TensorTypeNode>();
  // assign ndim before move
  TVM_FFI_ICHECK(shape.defined()) << "Must provide a shape in this constructor";
  ffi::Optional<ShapeType> shape_ty = MatchType<ShapeType>(shape);
  TVM_FFI_ICHECK(shape_ty) << "We expect shape to contain pre-set shape type";
  TVM_FFI_ICHECK(shape->IsInstance<ShapeExprNode>() || shape->IsInstance<VarNode>())
      << "We require shape to be normalized when constructing TensorType";
  n->ndim = shape_ty.value()->ndim;
  // assign rest of the fields.
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->vdevice = vdevice;
  n->span = span;
  data_ = std::move(n);
}

TensorType::TensorType(ffi::Optional<PrimType> dtype, int ndim, ffi::Optional<VDevice> vdevice,
                       Span span)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<TensorTypeNode> n = ffi::make_object<TensorTypeNode>();
  TVM_FFI_ICHECK(ndim >= -1) << "ndim of TensorType must be >= -1, but got " << ndim;
  n->ndim = ndim;
  n->dtype = dtype;
  n->vdevice = vdevice;
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "relax.TensorType", [](ffi::Optional<Expr> shape, ffi::Optional<PrimType> dtype, int ndim,
                             VDevice vdevice, Span span) {
        if (shape.has_value()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify shape and ndim";
          return TensorType(shape.value(), dtype, vdevice, span);
        } else {
          return TensorType(dtype, ndim, vdevice, span);
        }
      });
}

// Func
FuncType::FuncType(ffi::Array<Type> params, Type ret, bool purity, Span span)
    : Type(ffi::UnsafeInit{}) {
  ffi::ObjectPtr<FuncTypeNode> n = ffi::make_object<FuncTypeNode>();
  n->params = std::move(params);
  n->ret = std::move(ret);
  n->purity = std::move(purity);
  n->span = span;
  data_ = std::move(n);
}

FuncType FuncType::OpaqueFunc(TypeDeriveFunc derive_func, bool purity, Span span) {
  ffi::ObjectPtr<FuncTypeNode> n = ffi::make_object<FuncTypeNode>();
  n->derive_func = std::move(derive_func);
  n->ret = AnyType();
  n->purity = std::move(purity);
  n->span = span;
  return FuncType(n);
}

FuncType FuncType::OpaqueFunc(Type ret, bool purity, Span span) {
  ffi::ObjectPtr<FuncTypeNode> n = ffi::make_object<FuncTypeNode>();
  n->ret = std::move(ret);
  n->purity = std::move(purity);
  n->span = span;
  return FuncType(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.FuncType", [](ffi::Array<Type> params, Type ret, bool purity,
                                Span span) { return FuncType(params, ret, purity, span); })
      .def("relax.FuncTypeOpaqueFunc", [](ffi::Optional<Type> ret,
                                          ffi::Optional<TypeDeriveFunc> derive_func, bool purity,
                                          Span span) {
        if (derive_func.has_value()) {
          TVM_FFI_CHECK(!ret.has_value(), ValueError) << "Cannot specify both ret and derive_func";
          return FuncType::OpaqueFunc(derive_func.value(), purity, span);
        } else {
          return FuncType::OpaqueFunc(ret.value_or(AnyType()), purity, span);
        }
      });
}

// Helper functions
void UpdateType(Expr expr, Type ty) {
  TVM_FFI_ICHECK(expr->ty.IsMissing()) << "To ensure idempotency, "
                                       << "the expression passed to UpdateType "
                                       << "must not have any prior type.  "
                                       << "However, expression " << expr << " has type " << expr->ty
                                       << ", which cannot be overwritten with " << ty;
  expr->ty = ty;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.UpdateType", [](Expr expr, Type ty) { UpdateType(expr, ty); })
      .def("ir.ExprType", [](Expr expr) { return GetType(expr); });
}

}  // namespace relax
}  // namespace tvm
