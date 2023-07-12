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
#include <utility>

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

Stmt UpdatePointerStorageScope::VisitStmt_(const AllocateNode* op) {
  auto node = Downcast<Allocate>(StmtExprMutator::VisitStmt_(op));
  if (auto it = new_var_remap_.find(node->buffer_var.get()); it != new_var_remap_.end()) {
    node.CopyOnWrite()->buffer_var = it->second;
  }
  return std::move(node);
}

template <typename Node>
Node UpdatePointerStorageScope::UpdateBufferAccess(Node node) {
  auto new_buffer = GetUpdatedBuffer(node->buffer);
  if (!new_buffer.same_as(node->buffer)) {
    auto writer = node.CopyOnWrite();
    writer->buffer = new_buffer;
  }
  return node;
}

Buffer UpdatePointerStorageScope::GetUpdatedBuffer(Buffer buf) {
  // Use the cached buffer, if it exists.
  auto key = buf.get();
  auto it = new_buffer_remap_.find(key);
  if (it != new_buffer_remap_.end()) {
    return it->second;
  }

  // Update the buffer's var, if needed.
  auto remapped = Downcast<Var>(StmtExprMutator::VisitExpr(buf->data));
  if (!remapped.same_as(buf->data)) {
    auto writer = buf.CopyOnWrite();
    writer->data = remapped;
  }

  // Update the cache and return
  new_buffer_remap_[key] = buf;
  return buf;
}

Stmt UpdatePointerStorageScope::VisitStmt_(const DeclBufferNode* op) {
  auto node = Downcast<DeclBuffer>(StmtExprMutator::VisitStmt_(op));
  return UpdateBufferAccess(node);
}

PrimExpr UpdatePointerStorageScope::VisitExpr_(const BufferLoadNode* op) {
  auto node = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
  return UpdateBufferAccess(node);
}

Stmt UpdatePointerStorageScope::VisitStmt_(const BufferStoreNode* op) {
  auto node = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
  return UpdateBufferAccess(node);
}

}  // namespace tir
}  // namespace tvm
