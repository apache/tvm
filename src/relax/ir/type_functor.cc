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
 * \file type_functor.cc
 * \brief Implementations of Relax type functors.
 */
#include <tvm/ffi/cast.h>
#include <tvm/relax/type_functor.h>

namespace tvm {
namespace relax {

void TypeVisitor::VisitType_(const AnyTypeNode* op) {}

void TypeVisitor::VisitType_(const PrimTypeNode* op) {}

void TypeVisitor::VisitType_(const ShapeTypeNode* op) {
  if (op->values.defined()) {
    for (PrimExpr value : op->values.value()) {
      this->VisitTypeExprField(value);
    }
  }
}

void TypeVisitor::VisitType_(const TensorTypeNode* op) {
  if (op->shape.defined()) {
    this->VisitTypeExprField(op->shape.value());
  }
}

void TypeVisitor::VisitType_(const distributed::DTensorTypeNode* op) {
  this->VisitType(op->tensor_ty);
}

void TypeVisitor::VisitType_(const TupleTypeNode* op) {
  for (Type field : op->fields) {
    this->VisitType(field);
  }
}

void TypeVisitor::VisitType_(const FuncTypeNode* op) {
  if (op->params.defined()) {
    for (Type param : op->params.value()) {
      this->VisitType(param);
    }
  }
  this->VisitType(op->ret);
}

Type TypeMutator::VisitType_(const AnyTypeNode* op) { return ffi::GetRef<Type>(op); }

Type TypeMutator::VisitType_(const PrimTypeNode* op) { return ffi::GetRef<Type>(op); }

Type TypeMutator::VisitType_(const ShapeTypeNode* op) {
  if (!op->values.defined()) {
    return ffi::GetRef<Type>(op);
  }

  // if no changes are made the original array will be returned.
  ffi::Optional<ffi::Array<PrimExpr>> values = op->values.value().Map(
      [this](const PrimExpr& expr) { return this->VisitTypeExprField(expr); });

  if (values.same_as(op->values)) {
    return ffi::GetRef<Type>(op);
  } else {
    return ShapeType(values.value(), op->span);
  }
}

Type TypeMutator::VisitType_(const TensorTypeNode* op) {
  if (!op->shape.defined()) {
    return ffi::GetRef<Type>(op);
  }

  ffi::Optional<Expr> shape = this->VisitTypeExprField(op->shape.value());
  VDevice vdev = op->vdevice.value_or(VDevice());

  if (shape.same_as(op->shape)) {
    return ffi::GetRef<Type>(op);
  } else {
    return TensorType(shape.value(), op->dtype, vdev, op->span);
  }
}

Type TypeMutator::VisitType_(const distributed::DTensorTypeNode* op) {
  TensorType tensor_ty = this->VisitType(op->tensor_ty).as_or_throw<TensorType>();
  return distributed::DTensorType(tensor_ty, op->device_mesh, op->placement);
}

Type TypeMutator::VisitType_(const TupleTypeNode* op) {
  ffi::Array<Type> fields = op->fields.Map([this](const Type& ty) { return this->VisitType(ty); });

  if (fields.same_as(op->fields)) {
    return ffi::GetRef<Type>(op);
  } else {
    return TupleType(fields, op->span);
  }
}

Type TypeMutator::VisitType_(const FuncTypeNode* op) {
  ffi::Optional<ffi::Array<Type>> params;

  if (op->params.defined()) {
    params = op->params.value().Map([this](const Type& ty) { return this->VisitType(ty); });
  }

  Type ret = this->VisitType(op->ret);

  if (params.same_as(op->params) && ret.same_as(op->ret)) {
    return ffi::GetRef<Type>(op);
  } else {
    TVM_FFI_ICHECK(ret.defined()) << "FuncType must contain ret";
    if (params.defined()) {
      return FuncType(params.value(), ret, op->purity, op->span);
    } else {
      return FuncType::OpaqueFunc(ret, op->purity, op->span);
    }
  }
}

}  // namespace relax
}  // namespace tvm
