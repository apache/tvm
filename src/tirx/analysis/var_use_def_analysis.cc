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
#include "var_use_def_analysis.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
namespace tvm {
namespace tirx {

VarUseDefAnalyzer::VarUseDefAnalyzer(const ffi::Array<Var>& defined_vars, bool visit_thread_extent)
    : visit_thread_extent_(visit_thread_extent) {
  for (const Var v : defined_vars) {
    use_count_[v.get()] = 0;
  }
}

void VarUseDefAnalyzer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::thread_extent) {
    IterVar iv = op->node.as_or_throw<IterVar>();
    TVM_FFI_ICHECK_NE(iv->thread_tag.length(), 0U);
    // thread_extent can appear multiple times
    // use the first appearance as def.
    if (!use_count_.count(iv->var.get())) {
      this->HandleDef(iv->var);
    }

    if (visit_thread_extent_) {
      this->VisitExpr(op->value);
    }

    this->VisitStmt(op->body);
  } else {
    StmtExprVisitor::VisitStmt_(op);
  }
}

void VarUseDefAnalyzer::VisitStmt_(const BindNode* op) {
  this->HandleDef(op->var);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitStmt_(const ForNode* op) {
  this->HandleDef(op->loop_var);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitStmt_(const AllocBufferNode* op) {
  // VisitBufferDef (called by base) defines the typed buffer Var and visits
  // its dependent BufferType expressions.
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitExpr_(const prim::LetNode* op) {
  // Weaker SSA condition
  // A single var can be binded in multiple lets
  // but they have to bind to the same value.
  // This is used to allow cases when we reuse a single let
  // expression to construct a nested expr.
  // (let x = 1 in x + 1) * (let x = 1 in x + 1)
  auto it = let_binding_.find(op->var.get());
  this->VisitExpr(op->value);
  if (it != let_binding_.end()) {
    TVM_FFI_ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    this->HandleDef(op->var);
    let_binding_[op->var.get()] = op;
  }
  this->VisitExpr(op->body);
}

void VarUseDefAnalyzer::VisitExpr_(const VarNode* op) {
  Var var = ffi::GetRef<Var>(op);
  if (var->ty.as<BufferTypeNode>()) {
    this->VisitBufferUse(BufferVar(var));
  } else {
    this->HandleUse(var);
  }
  StmtExprVisitor::VisitExpr_(op);
}

void VarUseDefAnalyzer::VisitBufferDef(const BufferVar& buffer, bool alloc_data) {
  bool is_first_buffer_definition = !buffer_def_count_.count(buffer.get());
  HandleDef(buffer);
  if (is_first_buffer_definition) {
    auto it = use_count_.find(buffer.get());
    if (it == use_count_.end()) {
      HandleDef(buffer.var());
    }
  }
  // Visit shape/strides/elem_offset as uses of vars from the enclosing scope.
  for (const auto& e : buffer->shape) this->VisitExpr(e);
  for (const auto& e : buffer->strides) this->VisitExpr(e);
  this->VisitExpr(buffer->elem_offset);
}

void VarUseDefAnalyzer::VisitBufferUse(const BufferVar& buffer) {
  HandleUse(buffer);
  HandleUse(buffer.var());
}

void VarUseDefAnalyzer::HandleDef(const Var& var) {
  auto v = var.get();
  TVM_FFI_ICHECK(!def_count_.count(v))
      << "variable " << v->name << " has already been defined, the Stmt is not SSA";
  TVM_FFI_ICHECK(!use_count_.count(v))
      << "variable " << v->name << " has been used before definition!";
  use_count_[v] = 0;
  def_count_[v] = 1;
}

void VarUseDefAnalyzer::HandleUse(const Var& var) {
  auto v = var.get();
  auto it = use_count_.find(v);
  if (it != use_count_.end()) {
    if (it->second >= 0) {
      ++it->second;
    }
  } else {
    undefined_.push_back(ffi::GetRef<Var>(v));
    use_count_[v] = -1;
  }
}

void VarUseDefAnalyzer::HandleDef(const BufferVar& buf) {
  auto ptr = buf.get();
  // Some lowering pipelines may duplicate identical DeclBuffer nodes that
  // reference the same BufferVar object. Treat repeated definition of the same
  // buffer object as idempotent.
  if (buffer_def_count_.count(ptr)) {
    return;
  }
  if (!buffer_use_count_.count(ptr)) {
    buffer_use_count_[ptr] = 0;
  }
  buffer_def_count_[ptr] = 1;
  // BufferVar fields (data, shape, strides) are visited by the caller
  // (VisitBufferDef) via the base class, not here.
}

void VarUseDefAnalyzer::HandleUse(const BufferVar& buf) {
  auto ptr = buf.get();
  auto it = buffer_use_count_.find(ptr);
  if (it != buffer_use_count_.end()) {
    if (it->second >= 0) {
      ++it->second;
    }
  } else {
    undefined_buffers_.push_back(BufferVar(ffi::GetRef<Var>(ptr)));
    buffer_use_count_[ptr] = -1;
  }
  // BufferVar fields (shape, strides, data) are visited at the definition
  // site via VisitBufferDef.  Do not re-visit them at use sites, as the
  // buffer's shape variables may not be in scope at the point of use.
}

ffi::Array<Var> UndefinedVars(const Stmt& stmt, const ffi::Array<Var>& args) {
  VarUseDefAnalyzer m(args);
  m(stmt);
  return m.undefined_;
}

ffi::Array<Var> UndefinedVars(const PrimExpr& expr) {
  VarUseDefAnalyzer m({});
  m(expr);
  return m.undefined_;
}

ffi::Array<Var> UndefinedVars(const PrimExpr& expr, const ffi::Array<Var>& args) {
  VarUseDefAnalyzer m(args);
  m(expr);
  return m.undefined_;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      "tirx.analysis.UndefinedVars", [](ffi::PackedArgs args, ffi::Any* rv) {
        if (auto opt_stmt = args[0].as<Stmt>()) {
          *rv = UndefinedVars(opt_stmt.value(), args[1].cast<ffi::Array<Var>>());
        } else if (auto opt_expr = args[0].as<PrimExpr>()) {
          *rv = UndefinedVars(opt_expr.value(), args[1].cast<ffi::Array<Var>>());
        } else {
          TVM_FFI_THROW(InternalError)
              << "either UndefinedVars(stmt, args) or UndefinedVars(expr, args) is expected";
        }
      });
}
}  // namespace tirx
}  // namespace tvm
