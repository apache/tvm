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
 * \file update_pointer_storage_scope.cc
 * \brief A pass to update storage scopes for buffer variables.
 */
#include "update_pointer_storage_scope.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

Var WithStorageScope(const VarNode* buffer_var, String storage_scope) {
  auto* ptr_type = buffer_var->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type) << "The provided variable is not of pointer type";
  return Var(buffer_var->name_hint, PointerType(ptr_type->element_type, storage_scope),
             buffer_var->span);
}

UpdatePointerStorageScope::UpdatePointerStorageScope(
    const std::unordered_map<const VarNode*, String>& new_storage_scopes) {
  for (auto& kv : new_storage_scopes) {
    new_var_remap_[kv.first] = WithStorageScope(kv.first, kv.second);
  }
}

PrimExpr UpdatePointerStorageScope::VisitExpr_(const VarNode* op) {
  auto it = new_var_remap_.find(op);
  if (it == new_var_remap_.end()) {
    return GetRef<Var>(op);
  }
  return it->second;
}

PrimExpr UpdatePointerStorageScope::VisitExpr_(const LoadNode* op) {
  auto remapped = StmtExprMutator::VisitExpr(op->buffer_var);
  return Load(op->dtype, Downcast<Var>(remapped), StmtExprMutator::VisitExpr(op->index),
              StmtExprMutator::VisitExpr(op->predicate));
}

Stmt UpdatePointerStorageScope::VisitStmt_(const AllocateNode* op) {
  auto remapped = Downcast<Var>(StmtExprMutator::VisitExpr(op->buffer_var));
  return Allocate(remapped, op->dtype, op->extents, StmtExprMutator::VisitExpr(op->condition),
                  StmtExprMutator::VisitStmt(op->body));
}

Stmt UpdatePointerStorageScope::VisitStmt_(const StoreNode* op) {
  auto remapped = StmtExprMutator::VisitExpr(op->buffer_var);
  return Store(Downcast<Var>(remapped), StmtExprMutator::VisitExpr(op->value),
               StmtExprMutator::VisitExpr(op->index), StmtExprMutator::VisitExpr(op->predicate));
}

}  // namespace tir
}  // namespace tvm
