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
 * \file update_pointer_storage_scope.h
 * \brief A pass to update storage scopes for buffer variables.
 */
#ifndef TVM_TIR_TRANSFORMS_UPDATE_POINTER_STORAGE_SCOPE_H_
#define TVM_TIR_TRANSFORMS_UPDATE_POINTER_STORAGE_SCOPE_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>

namespace tvm {
namespace tir {

class UpdatePointerStorageScope : public StmtExprMutator {
 public:
  explicit UpdatePointerStorageScope(
      const std::unordered_map<const VarNode*, String>& new_storage_scopes);

  virtual PrimExpr VisitExpr_(const VarNode*);
  virtual PrimExpr VisitExpr_(const BufferLoadNode*);
  virtual Stmt VisitStmt_(const AllocateNode*);
  virtual Stmt VisitStmt_(const DeclBufferNode*);
  virtual Stmt VisitStmt_(const BufferStoreNode*);

 private:
  template <typename Node>
  Node UpdateBufferAccess(Node node);

  Buffer GetUpdatedBuffer(Buffer buf);

  std::unordered_map<const VarNode*, Var> new_var_remap_;
  std::unordered_map<const BufferNode*, Buffer> new_buffer_remap_;
};

}  // namespace tir
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORMS_UPDATE_POINTER_STORAGE_SCOPE_H_
