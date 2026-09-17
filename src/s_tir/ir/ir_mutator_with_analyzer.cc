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

#include "ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {
UnchangedOr<Stmt> IRMutatorWithAnalyzer::Extension::MutateBlock(IRMutatorWithAnalyzer* self,
                                                                const s_tir::SBlockNode* op,
                                                                InplaceMode inplace_mode) {
  return self->constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
    for (const auto& iter_var : op->iter_vars) {
      self->analyzer_->Bind(iter_var->var, iter_var->dom);
      self->iter_vars_.Set(iter_var->var, iter_var->dom);
    }
    return s_tir::StmtExprMutator::MutateBlock(self, op, inplace_mode);
  });
}

void IRMutatorWithAnalyzer::Extension::InitVTable(VTable* vtable) {
  vtable->ClearDispatch<s_tir::SBlockNode>();
  vtable->SetDispatch<s_tir::SBlockNode>(
      [](const ffi::Object* node, ObjectMutator* base, InplaceMode mode) -> UnchangedOr<ffi::Any> {
        return MutateBlock(static_cast<IRMutatorWithAnalyzer*>(base),
                           static_cast<const s_tir::SBlockNode*>(node), mode);
      });
}
TVM_FFI_STATIC_INIT_BLOCK() {
  IRMutatorWithAnalyzer::RegisterExtension(IRMutatorWithAnalyzer::Extension::InitVTable);
}
}  // namespace tirx
}  // namespace tvm
