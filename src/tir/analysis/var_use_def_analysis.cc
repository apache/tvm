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
namespace tvm {
namespace tir {

VarUseDefAnalyzer::VarUseDefAnalyzer(const Array<Var>& defined_vars, bool visit_thread_extent)
    : visit_thread_extent_(visit_thread_extent) {
  for (const Var v : defined_vars) {
    use_count_[v.get()] = 0;
  }
}

void VarUseDefAnalyzer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    ICHECK_NE(iv->thread_tag.length(), 0U);
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

void VarUseDefAnalyzer::VisitStmt_(const LetStmtNode* op) {
  this->HandleDef(op->var);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitStmt_(const ForNode* op) {
  this->HandleDef(op->loop_var);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitStmt_(const DeclBufferNode* op) {
  this->HandleDef(op->buffer);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitStmt_(const AllocateNode* op) {
  this->HandleDef(op->buffer_var);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitStmt_(const AllocateConstNode* op) {
  this->HandleDef(op->buffer_var);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitStmt_(const BufferStoreNode* op) {
  HandleUse(op->buffer);
  StmtExprVisitor::VisitStmt_(op);
}

void VarUseDefAnalyzer::VisitExpr_(const LetNode* op) {
  // Weaker SSA condition
  // A single var can be binded in multiple lets
  // but they have to bind to the same value.
  // This is used to allow cases when we reuse a single let
  // expression to construct a nested expr.
  // (let x = 1 in x + 1) * (let x = 1 in x + 1)
  auto it = let_binding_.find(op->var.get());
  this->VisitExpr(op->value);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    this->HandleDef(op->var);
    let_binding_[op->var.get()] = op;
  }
  this->VisitExpr(op->body);
}

void VarUseDefAnalyzer::VisitExpr_(const VarNode* op) {
  this->HandleUse(GetRef<Var>(op));
  StmtExprVisitor::VisitExpr_(op);
}

void VarUseDefAnalyzer::VisitExpr_(const ReduceNode* op) {
  for (const auto& iv : op->axis) {
    this->HandleDef(iv->var);
  }
  StmtExprVisitor::VisitExpr_(op);
}

void VarUseDefAnalyzer::VisitExpr_(const BufferLoadNode* op) {
  HandleUse(op->buffer);
  StmtExprVisitor::VisitExpr_(op);
}

void VarUseDefAnalyzer::VisitBuffer(const Buffer& buffer) {
  this->HandleUse(buffer->data);

  auto visit_arr = [&](Array<PrimExpr> arr) {
    for (const auto& element : arr) {
      this->VisitExpr(element);
    }
  };

  visit_arr(buffer->shape);
  visit_arr(buffer->strides);
}

void VarUseDefAnalyzer::HandleDef(const Var& var) {
  auto v = var.get();
  ICHECK(!def_count_.count(v)) << "variable " << v->name_hint
                               << " has already been defined, the Stmt is not SSA";
  ICHECK(!use_count_.count(v)) << "variable " << v->name_hint
                               << " has been used before definition!";
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
    undefined_.push_back(GetRef<Var>(v));
    use_count_[v] = -1;
  }
}

void VarUseDefAnalyzer::HandleDef(const Buffer& buf) {
  auto ptr = buf.get();
  ICHECK(!buffer_def_count_.count(ptr))
      << "buffer " << ptr->name << " has already been defined, the Stmt is not SSA";
  ICHECK(!buffer_use_count_.count(ptr))
      << "buffer " << ptr->name << " has been used before definition!";
  buffer_use_count_[ptr] = 0;
  buffer_def_count_[ptr] = 1;

  VisitBuffer(buf);
}

void VarUseDefAnalyzer::HandleUse(const Buffer& buf) {
  auto ptr = buf.get();
  auto it = buffer_use_count_.find(ptr);
  if (it != buffer_use_count_.end()) {
    if (it->second >= 0) {
      ++it->second;
    }
  } else {
    undefined_buffers_.push_back(GetRef<Buffer>(ptr));
    buffer_use_count_[ptr] = -1;
  }

  VisitBuffer(buf);
}

Array<Var> UndefinedVars(const Stmt& stmt, const Array<Var>& args) {
  VarUseDefAnalyzer m(args);
  m(stmt);
  return m.undefined_;
}

Array<Var> UndefinedVars(const PrimExpr& expr) {
  VarUseDefAnalyzer m({});
  m(expr);
  return m.undefined_;
}

Array<Var> UndefinedVars(const PrimExpr& expr, const Array<Var>& args) {
  VarUseDefAnalyzer m(args);
  m(expr);
  return m.undefined_;
}

TVM_REGISTER_GLOBAL("tir.analysis.UndefinedVars").set_body([](TVMArgs args, TVMRetValue* rv) {
  if (args.size() == 2 && args[0].IsObjectRef<Stmt>()) {
    *rv = UndefinedVars(args[0].AsObjectRef<Stmt>(), args[1]);
  } else if (args.size() == 2 && args[0].IsObjectRef<PrimExpr>()) {
    *rv = UndefinedVars(args[0].AsObjectRef<PrimExpr>(), args[1]);
  } else {
    LOG(FATAL) << "either UndefinedVars(stmt, args) or UndefinedVars(expr, args) is expected";
  }
});
}  // namespace tir
}  // namespace tvm
