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
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/builtin.h>
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
    const std::unordered_map<Var, ffi::String, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>&
        new_storage_scopes) {
  for (auto& kv : new_storage_scopes) {
    if (kv.first->ty.as<BufferTypeNode>()) {
      BufferVar buffer = GetBufferVar(kv.first.get());
      auto type = CopyBufferType(buffer);
      type->storage_scope = kv.second;
      BufferVar replacement = RebuildBufferVar(buffer, std::move(type));
      VarRemapSet(kv.first, replacement);
    } else {
      VarRemapSet(kv.first, WithStorageScope(kv.first.get(), kv.second));
    }
  }
}

UnchangedOr<Expr> UpdatePointerStorageScope::Mutate_(const CallNode* op, InplaceMode inplace_mode) {
  if (op->op.same_as(builtin::buffer_data()) && op->args.size() == 1) {
    auto arg_u = Mutate(op->args[0]);
    if (arg_u.UnchangedOrSameAs(op->args[0])) return ffi::Unchanged();

    Expr arg = std::move(arg_u).ValueUnchecked();
    BufferVar buffer = arg.as_or_throw<BufferVar>();
    return Call(buffer.DataPointerType(), op->op, {arg}, op->attrs, op->ty_args, op->span);
  }
  return StmtExprMutator::Mutate_(op, inplace_mode);
}

}  // namespace tirx
}  // namespace tvm
