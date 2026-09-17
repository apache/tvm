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

#include "stmt_simplify.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>

#include "../../tirx/transform/stmt_simplify.h"

namespace tvm {
namespace s_tir {
using namespace tirx;

// Reuse ordinary TIRX simplification, adding scoped constraints for S-TIR blocks.
class StmtSimplifier final : public tirx::StmtSimplifier {
 public:
  using Parent = tirx::StmtSimplifier;
  StmtSimplifier(const sym::Analyzer& analyzer, tirx::StmtSimplifyConfig config)
      : Parent(GlobalVTable(), analyzer, config) {}
  using Parent::Mutate_;
  using Parent::Run;

 public:
  UnchangedOr<Stmt> Mutate_(const SBlockNode* op, InplaceMode inplace_mode) {
    // This small binding step stays local: the shared simplifier is TIRX-only,
    // while the analyzer state is protected by its owning base class.
    return constraint_scope_.WithNewScope([&]() -> UnchangedOr<Stmt> {
      for (const auto& iter_var : op->iter_vars) {
        analyzer_->Bind(iter_var->var, iter_var->dom);
        iter_vars_.Set(iter_var->var, iter_var->dom);
      }
      return s_tir::StmtExprMutator::MutateBlock(this, op, inplace_mode);
    });
  }
  UnchangedOr<Stmt> Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode) {
    return s_tir::StmtExprMutator::MutateBlockRealize(this, op, inplace_mode);
  }

 protected:
  static void InitVTable(VTable* vtable) {
    Parent::InitVTable(vtable);
    SetDispatch<StmtSimplifier, SBlockNode>(vtable);
    SetDispatch<StmtSimplifier, SBlockRealizeNode>(vtable);
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

PrimFunc StmtSimplify(PrimFunc func, const sym::Analyzer& analyzer) {
  auto config = tvm::transform::PassConfigWithDefaults<tirx::StmtSimplifyConfig>();
  return ffi::make_object<StmtSimplifier>(analyzer, config)->Run(std::move(func));
}

namespace transform {
Pass StmtSimplify() {
  auto pass_func = [](PrimFunc func, IRModule, tvm::transform::PassContext ctx) {
    sym::Analyzer analyzer;
    auto config = ctx->GetConfig<tirx::StmtSimplifyConfig>("tirx.StmtSimplify")
                      .value_or(tvm::transform::PassConfigWithDefaults<tirx::StmtSimplifyConfig>());
    return ffi::make_object<s_tir::StmtSimplifier>(analyzer, config)->Run(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "s_tir.StmtSimplify", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("s_tir.transform.StmtSimplify", StmtSimplify);
}
}  // namespace transform
}  // namespace s_tir
}  // namespace tvm
