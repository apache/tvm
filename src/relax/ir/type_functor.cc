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

void TypeVisitor::VisitType_(const ObjectStructInfoNode* op) {}

void TypeVisitor::VisitType_(const PrimStructInfoNode* op) {
  if (op->value.defined()) {
    this->VisitStructInfoExprField(op->value.value());
  }
}

void TypeVisitor::VisitType_(const ShapeStructInfoNode* op) {
  if (op->values.defined()) {
    for (PrimExpr value : op->values.value()) {
      this->VisitStructInfoExprField(value);
    }
  }
}

void TypeVisitor::VisitType_(const TensorStructInfoNode* op) {
  if (op->shape.defined()) {
    this->VisitStructInfoExprField(op->shape.value());
  }
}

void TypeVisitor::VisitType_(const distributed::DTensorStructInfoNode* op) {
  this->VisitType(op->tensor_sinfo);
}

void TypeVisitor::VisitType_(const TupleStructInfoNode* op) {
  for (StructInfo field : op->fields) {
    this->VisitType(field);
  }
}

void TypeVisitor::VisitType_(const FuncStructInfoNode* op) {
  if (op->params.defined()) {
    for (StructInfo param : op->params.value()) {
      this->VisitType(param);
    }
  }
  this->VisitType(op->ret);
}

StructInfo TypeMutator::VisitType_(const ObjectStructInfoNode* op) {
  return ffi::GetRef<StructInfo>(op);
}

StructInfo TypeMutator::VisitType_(const PrimStructInfoNode* op) {
  if (!op->value.defined()) {
    return ffi::GetRef<StructInfo>(op);
  }

  auto new_expr = VisitStructInfoExprField(op->value.value());
  if (new_expr.same_as(op->value)) {
    return ffi::GetRef<StructInfo>(op);
  } else {
    return PrimStructInfo(new_expr);
  }
}

StructInfo TypeMutator::VisitType_(const ShapeStructInfoNode* op) {
  ffi::Optional<ffi::Array<PrimExpr>> values;

  if (op->values.defined()) {
    // if no changes are made the original array will be returned.
    values = op->values.value().Map(
        [this](const PrimExpr& expr) { return this->VisitStructInfoExprField(expr); });
  }

  if (values.same_as(op->values)) {
    return ffi::GetRef<StructInfo>(op);
  } else {
    return ShapeStructInfo(values.value(), op->span);
  }
}

StructInfo TypeMutator::VisitType_(const TensorStructInfoNode* op) {
  ffi::Optional<Expr> shape;

  if (op->shape.defined()) {
    shape = this->VisitStructInfoExprField(op->shape.value());
  }

  VDevice vdev = op->vdevice.value_or(VDevice());

  if (shape.same_as(op->shape)) {
    return ffi::GetRef<StructInfo>(op);
  } else {
    return TensorStructInfo(shape.value(), op->dtype, vdev, op->span);
  }
}

StructInfo TypeMutator::VisitType_(const distributed::DTensorStructInfoNode* op) {
  TensorStructInfo tensor_ty = Downcast<TensorStructInfo>(this->VisitType(op->tensor_sinfo));
  return distributed::DTensorStructInfo(tensor_ty, op->device_mesh, op->placement);
}

StructInfo TypeMutator::VisitType_(const TupleStructInfoNode* op) {
  ffi::Array<StructInfo> fields =
      op->fields.Map([this](const StructInfo& ty) { return this->VisitType(ty); });

  if (fields.same_as(op->fields)) {
    return ffi::GetRef<StructInfo>(op);
  } else {
    return TupleStructInfo(fields, op->span);
  }
}

StructInfo TypeMutator::VisitType_(const FuncStructInfoNode* op) {
  ffi::Optional<ffi::Array<StructInfo>> params;

  if (op->params.defined()) {
    params = op->params.value().Map([this](const StructInfo& ty) { return this->VisitType(ty); });
  }

  StructInfo ret = this->VisitType(op->ret);

  if (params.same_as(op->params) && ret.same_as(op->ret)) {
    return ffi::GetRef<StructInfo>(op);
  } else {
    TVM_FFI_ICHECK(ret.defined()) << "FuncStructInfo that contains params must contain ret";
    return FuncStructInfo(params.value(), ret, op->purity, op->span);
  }
}

}  // namespace relax
}  // namespace tvm
