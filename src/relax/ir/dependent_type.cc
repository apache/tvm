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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/analysis.h>
#include <tvm/relax/type.h>
#include <tvm/relax/type_functor.h>

namespace tvm {
namespace relax {

TVM_FFI_STATIC_INIT_BLOCK() {
  ObjectTypeNode::RegisterReflection();
  ShapeTypeNode::RegisterReflection();
  TensorTypeNode::RegisterReflection();
  FuncTypeNode::RegisterReflection();
}

ObjectType::ObjectType(Span span) {
  ffi::ObjectPtr<ObjectTypeNode> n = ffi::make_object<ObjectTypeNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ObjectType", [](Span span) { return ObjectType(span); });
}

// Shape
ShapeType::ShapeType(ffi::Array<PrimExpr> values, Span span) {
  ffi::ObjectPtr<ShapeTypeNode> n = ffi::make_object<ShapeTypeNode>();
  n->ndim = static_cast<int>(values.size());
  n->values = values.Map([](PrimExpr value) {
    if (value->IsInstance<IntImmNode>()) {
      return tvm::cast(DataType::Int(64), value);
    }
    TVM_FFI_ICHECK(value.dtype() == DataType::Int(64))
        << "the value in ShapeType can only have dtype of int64";
    return value;
  });
  n->span = span;
  data_ = std::move(n);
}

ShapeType::ShapeType(int ndim, Span span) {
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
        if (values.defined()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify values and ndim";
          return ShapeType(values.value(), span);
        } else {
          return ShapeType(ndim, span);
        }
      });
}

// Tensor
TensorType::TensorType(Expr shape, DataType dtype, ffi::Optional<VDevice> vdevice, Span span) {
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

TensorType::TensorType(DataType dtype, int ndim, ffi::Optional<VDevice> vdevice, Span span) {
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
      "relax.TensorType", [](ffi::Optional<Expr> shape, ffi::Optional<DataType> dtype, int ndim,
                             VDevice vdevice, Span span) {
        if (shape.defined()) {
          TVM_FFI_CHECK_EQ(ndim, kUnknownNDim, ValueError) << "Cannot both specify shape and ndim";
          return TensorType(shape.value(), dtype.value_or(DataType::Void()), vdevice, span);
        } else {
          return TensorType(dtype.value_or(DataType::Void()), ndim, vdevice, span);
        }
      });
}

// Func
FuncType::FuncType(ffi::Array<Type> params, Type ret, bool purity, Span span) {
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
  n->ret = ObjectType();
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
        if (derive_func.defined()) {
          TVM_FFI_CHECK(!ret.defined(), ValueError) << "Cannot specify both ret and derive_func";
          return FuncType::OpaqueFunc(derive_func.value(), purity, span);
        } else {
          return FuncType::OpaqueFunc(ret.value_or(ObjectType()), purity, span);
        }
      });
}

// Helper functions
void UpdateType(Expr expr, Type ty) {
  TVM_FFI_ICHECK(!expr->ty.defined()) << "To ensure idempotency, "
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
