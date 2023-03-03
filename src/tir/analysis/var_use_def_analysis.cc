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
 * \file var_use_def_analysis.cc
 * \brief Classes and functions to analyze var defition and usage.
 */
#include <tvm/tir/analysis.h>

#include "../../runtime/thread_storage_scope.h"
#include "../transforms/ir_utils.h"

namespace tvm {
namespace tir {

Stmt VarUseDefAnalysis::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    ICHECK_NE(iv->thread_tag.length(), 0U);
    // thread_extent can appear multiple times
    // use the first appearance as def.
    if (!use_count_.count(iv->var.get())) {
      this->HandleDef(iv->var.get());
      thread_axis_.push_back(iv);
      thread_extent_.push_back(op->value);
    }

    PrimExpr value = op->value;
    if (visit_thread_extent_) {
      value = this->VisitExpr(value);
    }
    Stmt body = this->VisitStmt(op->body);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    return AttrStmt(op->node, op->attr_key, value, body);
  } else {
    return StmtExprMutator::VisitStmt_(op);
  }
}

Stmt VarUseDefAnalysis::VisitStmt_(const LetStmtNode* op) {
  this->HandleDef(op->var.get());
  Stmt body = this->VisitStmt(op->body);
  // eliminate unreferenced let
  if (use_count_.at(op->var.get()) == 0 && SideEffect(op->value) <= CallEffectKind::kReadState &&
      simplify_let_) {
    return body;
  } else {
    PrimExpr value = this->VisitExpr(op->value);
    if (body.same_as(op->body) && value.same_as(op->value)) {
      return GetRef<Stmt>(op);
    } else {
      return LetStmt(op->var, value, body);
    }
  }
}

Stmt VarUseDefAnalysis::VisitStmt_(const ForNode* op) {
  this->HandleDef(op->loop_var.get());
  return StmtExprMutator::VisitStmt_(op);
}

Stmt VarUseDefAnalysis::VisitStmt_(const AllocateNode* op) {
  this->HandleDef(op->buffer_var.get());
  auto storage_scope = runtime::StorageScope::Create(GetPtrStorageScope(op->buffer_var));
  if (storage_scope.rank == runtime::StorageRank::kShared && storage_scope.tag == ".dyn") {
    ICHECK_EQ(use_dyn_shmem_, false) << "Only one dynamic shared memory allocation is allowed.";
    ICHECK_GT(op->extents.size(), 0);
    dyn_shmem_size_ = op->extents[0];
    for (size_t i = 1; i < op->extents.size(); ++i) {
      dyn_shmem_size_ *= op->extents[i];
    }
    dyn_shmem_size_ = dyn_shmem_size_ * (op->dtype.bytes());
    use_dyn_shmem_ = true;
  }
  return StmtExprMutator::VisitStmt_(op);
}

Stmt VarUseDefAnalysis::VisitStmt_(const AllocateConstNode* op) {
  this->HandleDef(op->buffer_var.get());
  return StmtExprMutator::VisitStmt_(op);
}

Stmt VarUseDefAnalysis::VisitStmt_(const StoreNode* op) {
  LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
}

Stmt VarUseDefAnalysis::VisitStmt_(const BufferStoreNode* op) {
  VisitBuffer(op->buffer);
  return StmtExprMutator::VisitStmt_(op);
}

PrimExpr VarUseDefAnalysis::VisitExpr_(const LetNode* op) {
  // Weaker SSA condition
  // A single var can be binded in multiple lets
  // but they have to bind to the same value.
  // This is used to allow cases when we reuse a single let
  // expression to construct a nested expr.
  // (let x = 1 in x + 1) * (let x = 1 in x + 1)
  auto it = let_binding_.find(op->var);
  PrimExpr value = this->VisitExpr(op->value);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, value))
        << "Let cannot bind the same var to two different values";
    return GetRef<PrimExpr>(it->second);
  } else {
    this->HandleDef(op->var.get());
    let_binding_[op->var] = op;
  }
  PrimExpr body = this->VisitExpr(op->body);
  // eliminate unreferenced let
  if (use_count_.at(op->var.get()) == 0 && SideEffect(op->value) <= CallEffectKind::kReadState &&
      simplify_let_) {
    return body;
  } else {
    if (body.same_as(op->body) && value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Let(op->var, value, body);
    }
  }
}

PrimExpr VarUseDefAnalysis::VisitExpr_(const VarNode* op) {
  this->HandleUse(GetRef<PrimExpr>(op));
  return StmtExprMutator::VisitExpr_(op);
}

PrimExpr VarUseDefAnalysis::VisitExpr_(const ReduceNode* op) {
  for (const auto& iv : op->axis) {
    this->HandleDef(iv->var.get());
  }
  return StmtExprMutator::VisitExpr_(op);
}

PrimExpr VarUseDefAnalysis::VisitExpr_(const LoadNode* op) {
  LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
}

PrimExpr VarUseDefAnalysis::VisitExpr_(const BufferLoadNode* op) {
  VisitBuffer(op->buffer);
  return StmtExprMutator::VisitExpr_(op);
}

void VarUseDefAnalysis::VisitBuffer(Buffer buffer) {
  this->HandleUse(buffer->data);
  auto visit_arr = [&](Array<PrimExpr> arr) {
    for (const auto& element : arr) {
      this->VisitExpr(element);
    }
  };

  visit_arr(buffer->shape);
  visit_arr(buffer->strides);
}

void VarUseDefAnalysis::HandleDef(const VarNode* v) {
  ICHECK(!def_count_.count(v)) << "variable " << v->name_hint
                               << " has already been defined, the Stmt is not SSA";
  ICHECK(!use_count_.count(v)) << "variable " << v->name_hint
                               << " has been used before definition!";
  use_count_[v] = 0;
  def_count_[v] = 1;
}

void VarUseDefAnalysis::HandleUse(const PrimExpr& v) {
  ICHECK(v.as<VarNode>());
  Var var = Downcast<Var>(v);
  auto it = use_count_.find(var.get());
  if (it != use_count_.end()) {
    if (it->second >= 0) {
      ++it->second;
    }
  } else {
    undefined_.push_back(var);
    use_count_[var.get()] = -1;
  }
}

Array<Var> UndefinedVars(const Stmt& stmt, const Array<Var>& args) {
  VarUseDefAnalysis m;
  m.simplify_let_ = false;
  for (Var arg : args) {
    m.use_count_[arg.get()] = 0;
  }
  m(stmt);
  return m.undefined_;
}

Array<Var> UndefinedVars(const PrimExpr& expr) {
  VarUseDefAnalysis m;
  m.simplify_let_ = false;
  m(expr);
  return m.undefined_;
}

}  // namespace tir
}  // namespace tvm
