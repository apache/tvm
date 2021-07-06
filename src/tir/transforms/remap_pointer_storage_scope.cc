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
 * TODO
 * \file remap_pointer_storage_scope.cc
 */
#include "remap_pointer_storage_scope.h"

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

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

RemapStorageScope::RemapStorageScope(
    const std::unordered_map<const VarNode*, String>& new_storage_scopes) {
  for (auto kv : new_storage_scopes) {
    new_var_remap_[kv.first] = WithStorageScope(kv.first, kv.second);
  }
}

Stmt RemapStorageScope::VisitStmt_(const AttrStmtNode* op) {
  using runtime::StorageScope;
  if (op->attr_key == attr::storage_scope) {
    const VarNode* buf = op->node.as<VarNode>();
    auto it = new_var_remap_.find(buf);
    if (it != new_var_remap_.end()) {
      auto remapped = it->second;
      auto new_scope = GetPtrStorageScope(remapped);
      return AttrStmt(remapped, attr::storage_scope, StringImm(new_scope),
                      StmtMutator::VisitStmt(op->body));
    }
  }
  return StmtMutator::VisitStmt_(op);
}

}  // namespace tir
}  // namespace tvm
