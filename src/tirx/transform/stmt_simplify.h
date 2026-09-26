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
 * \file stmt_simplify.h
 * \brief Statement-level simplification of TIR PrimFuncs.
 */
#ifndef TVM_TIR_TRANSFORM_STMT_SIMPLIFY_H_
#define TVM_TIR_TRANSFORM_STMT_SIMPLIFY_H_

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>

#include "../ir/ir_mutator_with_analyzer.h"

namespace tvm {
namespace tirx {

struct StmtSimplifyConfigNode : public ffi::Object {
  bool transitively_prove_inequalities;
  bool convert_boolean_to_and_of_ors;
  bool apply_constraints_to_boolean_branches;

  static void RegisterReflection();
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.transform.StmtSimplifyConfig", StmtSimplifyConfigNode,
                                    ffi::Object);

  sym::RewriteSimplifier::Extension GetEnabledExtensions() const;
};

class StmtSimplifyConfig : public ffi::ObjectRef {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(StmtSimplifyConfig, ffi::ObjectRef,
                                                StmtSimplifyConfigNode);
};

class StmtSimplifier : public IRMutatorWithAnalyzer {
 public:
  using IRMutatorWithAnalyzer::Mutate;
  using IRMutatorWithAnalyzer::Mutate_;
  static PrimFunc Apply(PrimFunc func, const sym::Analyzer& analyzer,
                        ffi::Optional<StmtSimplifyConfig> config_opt = std::nullopt);

  explicit StmtSimplifier(const sym::Analyzer& analyzer, StmtSimplifyConfig config)
      : IRMutatorWithAnalyzer(analyzer), config_(config) {}

 protected:
  using Parent = IRMutatorWithAnalyzer;
  StmtSimplifier(const VTable* vtable, const sym::Analyzer& analyzer, StmtSimplifyConfig config)
      : Parent(analyzer.get(), vtable), config_(config) {}
  PrimFunc Run(PrimFunc func);

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) final;

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final;

  UnchangedOr<Stmt> Mutate_(const BindNode* op, InplaceMode inplace_mode) override;

  UnchangedOr<Stmt> Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) override;

  // eliminate useless stores
  UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) override;

 private:
  bool ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs);

  /* \brief Internal utility for checking conditionals
   *
   * Substitutes any known Bind values and then simplifies with the analyzer.
   */
  ffi::Optional<bool> ProveCondition(PrimExpr condition) const;

  StmtSimplifyConfig config_;

  // Pure Bind values kept for substitution into assert conditions.
  // Grows monotonically under SSA — no scope-based cleanup required.
  ffi::Map<Var, PrimExpr> non_inlined_bindings_;
};

/* \brief Simplify statements in the prim func
 *
 * Applies the same behavior as the tirx.transform.StmtSimplify pass.
 */
PrimFunc StmtSimplify(PrimFunc func, const sym::Analyzer& analyzer);

}  // namespace tirx
}  // namespace tvm
#endif  // TVM_TIR_TRANSFORM_STMT_SIMPLIFY_H_
