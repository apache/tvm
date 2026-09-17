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

#ifndef TVM_S_TIR_IR_MUTATOR_WITH_ANALYZER_H_
#define TVM_S_TIR_IR_MUTATOR_WITH_ANALYZER_H_

#include <tvm/s_tir/stmt_functor.h>

#include "../../tirx/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {
class IRMutatorWithAnalyzer::Extension {
 public:
  static UnchangedOr<Stmt> MutateBlock(IRMutatorWithAnalyzer* self, const s_tir::SBlockNode* op,
                                       InplaceMode inplace_mode);
  static void InitVTable(VTable* vtable);
};
}  // namespace tirx
namespace s_tir {
class IRMutatorWithAnalyzer : public tirx::IRMutatorWithAnalyzer {
 public:
  using Parent = tirx::IRMutatorWithAnalyzer;
  using Parent::Mutate;
  using Parent::Mutate_;
  explicit IRMutatorWithAnalyzer(const arith::Analyzer& analyzer)
      : IRMutatorWithAnalyzer(analyzer.get()) {}
  explicit IRMutatorWithAnalyzer(arith::AnalyzerObj* analyzer) : Parent(analyzer, GlobalVTable()) {}
  virtual UnchangedOr<tirx::Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
    return Parent::Extension::MutateBlock(this, op, inplace_mode);
  }

 protected:
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    vtable->ClearDispatch<SBlockNode>();
    SetDispatch<IRMutatorWithAnalyzer, SBlockNode>(vtable);
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
#endif
