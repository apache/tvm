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

#include <tvm/ffi/cast.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <utility>

#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tirx {

Var WithStorageScope(const VarNode* buffer_var, ffi::String storage_scope) {
  auto* ptr_type = buffer_var->ty.as<PointerTypeNode>();
  TVM_FFI_ICHECK(ptr_type) << "The provided variable is not of pointer type";
  return Var(buffer_var->name, PointerType(ptr_type->element_type, storage_scope),
             buffer_var->span);
}

UpdatePointerStorageScope::UpdatePointerStorageScope(
    const std::unordered_map<const VarNode*, ffi::String>& new_storage_scopes) {
  for (auto& kv : new_storage_scopes) {
    if (kv.first->ty.as<BufferTypeNode>()) {
      BufferVar buffer = GetBufferVar(kv.first);
      auto type = CopyBufferType(buffer);
      type->storage_scope = kv.second;
      BufferVar replacement = RebuildBufferVar(buffer, std::move(type));
      new_var_remap_[kv.first] = replacement.var();
    } else {
      new_var_remap_[kv.first] = WithStorageScope(kv.first, kv.second);
    }
  }
}

Expr UpdatePointerStorageScope::VisitExpr_(const VarNode* op) {
  auto it = new_var_remap_.find(op);
  if (it == new_var_remap_.end()) {
    return ffi::GetRef<Var>(op);
  }
  return it->second;
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

template <>
TensorLoad UpdatePointerStorageScope::UpdateBufferAccess(TensorLoad node) {
  BufferVar buffer = node->source.as_or_throw<tvm::tirx::BufferVar>();
  BufferVar new_buffer = GetUpdatedBuffer(buffer);
  return new_buffer.same_as(buffer) ? node : BufferLoad(new_buffer, node->indices, node->span);
}

BufferVar UpdatePointerStorageScope::GetUpdatedBuffer(BufferVar buf) {
  auto it = new_var_remap_.find(buf.get());
  if (it != new_var_remap_.end()) {
    return BufferVar(it->second);
  }
  return buf;
}

Stmt UpdatePointerStorageScope::VisitStmt_(const AllocBufferNode* op) {
  auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<AllocBuffer>();
  return UpdateBufferAccess(node);
}

Stmt UpdatePointerStorageScope::VisitStmt_(const DeclBufferNode* op) {
  auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<DeclBuffer>();
  return UpdateBufferAccess(node);
}

Expr UpdatePointerStorageScope::VisitExpr_(const TensorLoadNode* op) {
  auto node = StmtExprMutator::VisitExpr_(op).as_or_throw<TensorLoad>();
  return UpdateBufferAccess(node);
}

Stmt UpdatePointerStorageScope::VisitStmt_(const BufferStoreNode* op) {
  auto node = StmtExprMutator::VisitStmt_(op).as_or_throw<BufferStore>();
  return UpdateBufferAccess(node);
}

}  // namespace tirx
}  // namespace tvm
