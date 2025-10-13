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
 * \file src/relax/dataflow_rewriter.h
 * \brief Pattern match/rewriters for Relax
 */
#ifndef TVM_RELAX_IR_DATAFLOW_REWRITER_H_
#define TVM_RELAX_IR_DATAFLOW_REWRITER_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/relax/expr.h>

#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "dataflow_matcher.h"

namespace tvm {
namespace relax {

struct RewriteSpec {
  ffi::Map<Var, Expr> variable_rewrites;
  ffi::Map<GlobalVar, BaseFunc> new_subroutines;

  explicit operator bool() const { return variable_rewrites.size(); }

  void Append(RewriteSpec other);
};

class PatternMatchingRewriterNode : public tvm::transform::PassNode {
 public:
  virtual RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const {
    return RewriteSpec();
  }

  static void RegisterReflection() {
    // PatternMatchingRewriterNode has no fields to register
  }

  IRModule operator()(IRModule mod, const tvm::transform::PassContext& pass_ctx) const override;
  tvm::transform::PassInfo Info() const override;
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.PatternMatchingRewriter", PatternMatchingRewriterNode,
                              PassNode);
};

class PatternMatchingRewriter : public tvm::transform::Pass {
 public:
  static PatternMatchingRewriter FromPattern(
      DFPattern pattern,
      ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
      ffi::Optional<ffi::Array<DFPattern>> additional_bindings = std::nullopt,
      ffi::Map<GlobalVar, BaseFunc> new_subroutines = {});

  static PatternMatchingRewriter FromModule(IRModule mod);

  Expr operator()(Expr expr);
  using Pass::operator();

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(PatternMatchingRewriter, Pass,
                                             PatternMatchingRewriterNode);
};

class ExprPatternRewriterNode : public PatternMatchingRewriterNode {
 public:
  DFPattern pattern;
  ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func;
  ffi::Optional<ffi::Array<DFPattern>> additional_bindings;
  ffi::Map<GlobalVar, BaseFunc> new_subroutines;

  RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const final;

  ffi::Optional<Expr> RewriteExpr(const Expr& expr, const ffi::Map<Var, Expr>& bindings) const;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<ExprPatternRewriterNode>()
        .def_ro("pattern", &ExprPatternRewriterNode::pattern)
        .def_ro("func", &ExprPatternRewriterNode::func);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.ExprPatternRewriter", ExprPatternRewriterNode,
                              PatternMatchingRewriterNode);
};

class ExprPatternRewriter : public PatternMatchingRewriter {
 public:
  ExprPatternRewriter(DFPattern pattern,
                      ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
                      ffi::Optional<ffi::Array<DFPattern>> additional_bindings = std::nullopt,
                      ffi::Map<GlobalVar, BaseFunc> new_subroutines = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ExprPatternRewriter, PatternMatchingRewriter,
                                             ExprPatternRewriterNode);
};

class OrRewriterNode : public PatternMatchingRewriterNode {
 public:
  PatternMatchingRewriter lhs;
  PatternMatchingRewriter rhs;

  RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<OrRewriterNode>()
        .def_ro("lhs", &OrRewriterNode::lhs)
        .def_ro("rhs", &OrRewriterNode::rhs);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.OrRewriter", OrRewriterNode, PatternMatchingRewriterNode);
};

class OrRewriter : public PatternMatchingRewriter {
 public:
  OrRewriter(PatternMatchingRewriter lhs, PatternMatchingRewriter rhs);

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(OrRewriter, PatternMatchingRewriter, OrRewriterNode);
};

class TupleRewriterNode : public PatternMatchingRewriterNode {
 public:
  ffi::Array<DFPattern> patterns;
  ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func;
  ffi::Optional<ffi::Array<DFPattern>> additional_bindings;
  ffi::Map<GlobalVar, BaseFunc> new_subroutines;

  RewriteSpec RewriteBindings(const ffi::Array<Binding>& bindings) const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<TupleRewriterNode>()
        .def_ro("patterns", &TupleRewriterNode::patterns)
        .def_ro("func", &TupleRewriterNode::func);
  }
  TVM_FFI_DECLARE_OBJECT_INFO("relax.dpl.TupleRewriter", TupleRewriterNode,
                              PatternMatchingRewriterNode);

 private:
  struct VarInfo {
    Var var;
    Expr expr;
    ffi::Array<ffi::Optional<ffi::Map<DFPattern, Expr>>> matches;
    std::unordered_set<Var> downstream_usage;
    bool used = false;
  };

  ffi::Map<Var, Expr> GenerateVariableRewrites(const ffi::Array<Binding>& bindings) const;

  std::optional<std::vector<Expr>> TryMatchByBindingIndex(const std::vector<VarInfo>& info_vec,
                                                          const std::vector<size_t>& indices) const;
};

class TupleRewriter : public PatternMatchingRewriter {
 public:
  TupleRewriter(ffi::Array<DFPattern> patterns,
                ffi::TypedFunction<ffi::Optional<Expr>(Expr, ffi::Map<DFPattern, Expr>)> func,
                ffi::Optional<ffi::Array<DFPattern>> additional_bindings = std::nullopt,
                ffi::Map<GlobalVar, BaseFunc> new_subroutines = {});

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TupleRewriter, PatternMatchingRewriter,
                                             TupleRewriterNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_DATAFLOW_REWRITER_H_
