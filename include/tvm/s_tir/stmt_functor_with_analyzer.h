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
 * \file tvm/s_tir/stmt_functor_with_analyzer.h
 * \brief Statement and expression functors with an analyzer context.
 */

#ifndef TVM_S_TIR_STMT_FUNCTOR_WITH_ANALYZER_H_
#define TVM_S_TIR_STMT_FUNCTOR_WITH_ANALYZER_H_

#include <tvm/s_tir/stmt_functor.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>

namespace tvm {
namespace s_tir {

class StmtExprVisitorWithAnalyzer : public tirx::StmtExprVisitorWithAnalyzer {
 public:
  using Parent = tirx::StmtExprVisitorWithAnalyzer;
  using Parent::Visit;
  using Parent::Visit_;
  TVM_DEFINE_OBJECT_FUNCTOR_DEFAULT_CONSTRUCTOR(StmtExprVisitorWithAnalyzer, Parent)
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* op);
  virtual ffi::Optional<VisitInterrupt> Visit_(const SBlockRealizeNode* op) {
    return s_tir::StmtExprVisitor::VisitBlockRealize(this, op);
  }

 protected:
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    SetDispatch<StmtExprVisitorWithAnalyzer, SBlockNode>(vtable);
    SetDispatch<StmtExprVisitorWithAnalyzer, SBlockRealizeNode>(vtable);
  }
};

class StmtExprMutatorWithAnalyzer : public tirx::StmtExprMutatorWithAnalyzer {
 public:
  using Parent = tirx::StmtExprMutatorWithAnalyzer;
  using Parent::Mutate;
  using Parent::Mutate_;
  explicit StmtExprMutatorWithAnalyzer(const sym::Analyzer& analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer.get()) {}
  explicit StmtExprMutatorWithAnalyzer(sym::AnalyzerObj* analyzer)
      : Parent(analyzer, GlobalVTable()) {}
  virtual UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode);
  virtual UnchangedOr<Stmt> Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode) {
    return s_tir::StmtExprMutator::MutateBlockRealize(this, op, inplace_mode);
  }

 protected:
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    SetDispatch<StmtExprMutatorWithAnalyzer, SBlockNode>(vtable);
    SetDispatch<StmtExprMutatorWithAnalyzer, SBlockRealizeNode>(vtable);
  }
  static const VTable* GlobalVTable() {
    static const VTable table = [] {
      VTable table;
      InitVTable(&table);
      table.Finalize();
      return table;
    }();
    return &table;
  }
};

}  // namespace s_tir
}  // namespace tvm
#endif  // TVM_S_TIR_STMT_FUNCTOR_WITH_ANALYZER_H_
