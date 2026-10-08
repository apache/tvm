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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
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
    const std::unordered_map<Var, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>&
        new_storage_scopes) {
  for (auto& kv : new_storage_scopes) {
    if (kv.first->ty.as<TensorTypeNode>()) {
      TensorVar buffer = GetTensorVar(kv.first.get());
      auto type = CopyTensorType(buffer);
      type->storage_scope = kv.second;
      TensorVar replacement = RebuildTensorVar(buffer, std::move(type));
      VarRemapSet(kv.first, replacement);
    } else {
      VarRemapSet(kv.first, WithStorageScope(kv.first.get(), kv.second));
    }
  }
}

UnchangedOr<Stmt> UpdatePointerStorageScope::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  const auto* call = op->value.as<CallNode>();
  if (call &&
      (call->op.same_as(tirx::alloc_tensor_op()) || call->op.same_as(tirx::decl_tensor_op()))) {
    if (auto mapped = VarRemapGet(op->var); mapped != nullptr) {
      buffer_scopes_.emplace(call, mapped.as_or_throw<TensorVar>().scope());
      auto result = StmtExprMutator::Mutate_(op, inplace_mode);
      buffer_scopes_.erase(call);
      return result;
    }
  }
  return StmtExprMutator::Mutate_(op, inplace_mode);
}

UnchangedOr<Expr> UpdatePointerStorageScope::Mutate_(const CallNode* op, InplaceMode inplace_mode) {
  auto result = StmtExprMutator::Mutate_(op, inplace_mode);
  if (auto it = buffer_scopes_.find(op); it != buffer_scopes_.end()) {
    Expr value = std::move(result).ValueOrUnchanged(ffi::GetRef<Expr>(op));
    auto call = value.as_or_throw<Call>();
    size_t scope_index = call->op.same_as(tirx::alloc_tensor_op()) ? 2 : 3;
    if (call->args[scope_index].as_or_throw<StringImm>()->value != it->second) {
      auto copy = ffi::make_object<CallNode>(*call.get());
      copy->args.Set(scope_index, StringImm(it->second, call->args[scope_index]->span));
      return ReinferMutatedCallType(Expr(std::move(copy)), op, inplace_mode);
    }
    return value;
  }
  if (!op->op.same_as(tirx::buffer_data_op())) return result;
  return ReinferMutatedCallType(std::move(result), op, inplace_mode);
}

}  // namespace tirx
}  // namespace tvm
