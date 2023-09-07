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
 * \file struct_info_functor.cc
 * \brief Implementations of struct info functors.
 */
#include <tvm/relax/struct_info_functor.h>

namespace tvm {
namespace relax {

void StructInfoVisitor::VisitStructInfo_(const ObjectStructInfoNode* op) {}

void StructInfoVisitor::VisitStructInfo_(const PrimStructInfoNode* op) {
  if (op->value.defined()) {
    this->VisitStructInfoExprField(op->value.value());
  }
}

void StructInfoVisitor::VisitStructInfo_(const ShapeStructInfoNode* op) {
  if (op->values.defined()) {
    for (PrimExpr value : op->values.value()) {
      this->VisitStructInfoExprField(value);
    }
  }
}

void StructInfoVisitor::VisitStructInfo_(const TensorStructInfoNode* op) {
  if (op->shape.defined()) {
    this->VisitStructInfoExprField(op->shape.value());
  }
}

void StructInfoVisitor::VisitStructInfo_(const distributed::DTensorStructInfoNode* op) {
  this->VisitStructInfo(op->tensor_sinfo);
}

void StructInfoVisitor::VisitStructInfo_(const TupleStructInfoNode* op) {
  for (StructInfo field : op->fields) {
    this->VisitStructInfo(field);
  }
}

void StructInfoVisitor::VisitStructInfo_(const FuncStructInfoNode* op) {
  if (op->params.defined()) {
    for (StructInfo param : op->params.value()) {
      this->VisitStructInfo(param);
    }
  }
  this->VisitStructInfo(op->ret);
}

StructInfo StructInfoMutator::VisitStructInfo_(const ObjectStructInfoNode* op) {
  return GetRef<StructInfo>(op);
}

StructInfo StructInfoMutator::VisitStructInfo_(const PrimStructInfoNode* op) {
  if (!op->value.defined()) {
    return GetRef<StructInfo>(op);
  }

  auto new_expr = VisitStructInfoExprField(op->value.value());
  if (new_expr.same_as(op->value)) {
    return GetRef<StructInfo>(op);
  } else {
    return PrimStructInfo(new_expr);
  }
}

StructInfo StructInfoMutator::VisitStructInfo_(const ShapeStructInfoNode* op) {
  Optional<Array<PrimExpr>> values;

  if (op->values.defined()) {
    // if no changes are made the original array will be returned.
    values = op->values.value().Map(
        [this](const PrimExpr& expr) { return this->VisitStructInfoExprField(expr); });
  }

  if (values.same_as(op->values)) {
    return GetRef<StructInfo>(op);
  } else {
    return ShapeStructInfo(values.value(), op->span);
  }
}

StructInfo StructInfoMutator::VisitStructInfo_(const TensorStructInfoNode* op) {
  Optional<Expr> shape;

  if (op->shape.defined()) {
    shape = this->VisitStructInfoExprField(op->shape.value());
  }

  VDevice vdev = op->vdevice.value_or(VDevice());

  if (shape.same_as(op->shape)) {
    return GetRef<StructInfo>(op);
  } else {
    return TensorStructInfo(shape.value(), op->dtype, vdev, op->span);
  }
}

StructInfo StructInfoMutator::VisitStructInfo_(const distributed::DTensorStructInfoNode* op) {
  TensorStructInfo tensor_sinfo =
      Downcast<TensorStructInfo>(this->VisitStructInfo(op->tensor_sinfo));
  return distributed::DTensorStructInfo(tensor_sinfo, op->device_mesh, op->placement);
}

StructInfo StructInfoMutator::VisitStructInfo_(const TupleStructInfoNode* op) {
  Array<StructInfo> fields =
      op->fields.Map([this](const StructInfo& sinfo) { return this->VisitStructInfo(sinfo); });

  if (fields.same_as(op->fields)) {
    return GetRef<StructInfo>(op);
  } else {
    return TupleStructInfo(fields, op->span);
  }
}

StructInfo StructInfoMutator::VisitStructInfo_(const FuncStructInfoNode* op) {
  Optional<Array<StructInfo>> params;

  if (op->params.defined()) {
    params = op->params.value().Map(
        [this](const StructInfo& sinfo) { return this->VisitStructInfo(sinfo); });
  }

  StructInfo ret = this->VisitStructInfo(op->ret);

  if (params.same_as(op->params) && ret.same_as(op->ret)) {
    return GetRef<StructInfo>(op);
  } else {
    ICHECK(ret.defined()) << "FuncStructInfo that contains params must contain ret";
    return FuncStructInfo(params.value(), ret, op->purity, op->span);
  }
}

}  // namespace relax
}  // namespace tvm
