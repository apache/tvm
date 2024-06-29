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

#include <tvm/ir/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/node/reflection.h>
#include <tvm/relax/expr.h>

#include <optional>

#include "dataflow_matcher.h"

namespace tvm {
namespace relax {

struct RewriteSpec {
  Map<Var, Expr> variable_rewrites;
  Map<GlobalVar, BaseFunc> new_subroutines;

  explicit operator bool() const { return variable_rewrites.size(); }

  void Append(RewriteSpec other);
};

class ExprRewriterNode : public tvm::transform::PassNode {
 public:
  virtual RewriteSpec RewriteBindings(const Array<Binding>& bindings) const {
    return RewriteSpec();
  }

  void VisitAttrs(AttrVisitor* visitor) {}

  IRModule operator()(IRModule mod, const tvm::transform::PassContext& pass_ctx) const override;
  tvm::transform::PassInfo Info() const override;

  static constexpr const char* _type_key = "relax.dpl.ExprRewriter";
  TVM_DECLARE_BASE_OBJECT_INFO(ExprRewriterNode, PassNode);
};

class ExprRewriter : public tvm::transform::Pass {
 public:
  static ExprRewriter FromPattern(DFPattern pattern,
                                  TypedPackedFunc<Optional<Expr>(Expr, Map<DFPattern, Expr>)> func,
                                  Optional<Array<DFPattern>> additional_bindings = NullOpt,
                                  Map<GlobalVar, BaseFunc> new_subroutines = {});

  static ExprRewriter FromModule(IRModule mod);

  Expr operator()(Expr expr);
  using Pass::operator();

  TVM_DEFINE_OBJECT_REF_METHODS(ExprRewriter, Pass, ExprRewriterNode);
};

class PatternRewriterNode : public ExprRewriterNode {
 public:
  DFPattern pattern;
  TypedPackedFunc<Optional<Expr>(Expr, Map<DFPattern, Expr>)> func;
  Optional<Array<DFPattern>> additional_bindings;
  Map<GlobalVar, BaseFunc> new_subroutines;

  RewriteSpec RewriteBindings(const Array<Binding>& bindings) const final;

  Optional<Expr> RewriteExpr(const Expr& expr, const Map<Var, Expr>& bindings) const;

  void VisitAttrs(AttrVisitor* visitor) {
    visitor->Visit("pattern", &pattern);
    PackedFunc untyped_func = func;
    visitor->Visit("func", &untyped_func);
  }

  static constexpr const char* _type_key = "relax.dpl.PatternRewriter";
  TVM_DECLARE_BASE_OBJECT_INFO(PatternRewriterNode, ExprRewriterNode);
};

class PatternRewriter : public ExprRewriter {
 public:
  PatternRewriter(DFPattern pattern,
                  TypedPackedFunc<Optional<Expr>(Expr, Map<DFPattern, Expr>)> func,
                  Optional<Array<DFPattern>> additional_bindings = NullOpt,
                  Map<GlobalVar, BaseFunc> new_subroutines = {});

  TVM_DEFINE_OBJECT_REF_METHODS(PatternRewriter, ExprRewriter, PatternRewriterNode);
};

class OrRewriterNode : public ExprRewriterNode {
 public:
  ExprRewriter lhs;
  ExprRewriter rhs;

  RewriteSpec RewriteBindings(const Array<Binding>& bindings) const override;

  void VisitAttrs(AttrVisitor* visitor) {
    visitor->Visit("lhs", &lhs);
    visitor->Visit("rhs", &rhs);
  }

  static constexpr const char* _type_key = "relax.dpl.OrRewriter";
  TVM_DECLARE_BASE_OBJECT_INFO(OrRewriterNode, ExprRewriterNode);
};

class OrRewriter : public ExprRewriter {
 public:
  OrRewriter(ExprRewriter lhs, ExprRewriter rhs);

  TVM_DEFINE_OBJECT_REF_METHODS(OrRewriter, ExprRewriter, OrRewriterNode);
};

class TupleRewriterNode : public ExprRewriterNode {
 public:
  Array<DFPattern> patterns;
  TypedPackedFunc<Optional<Expr>(Expr, Map<DFPattern, Expr>)> func;
  Optional<Array<DFPattern>> additional_bindings;
  Map<GlobalVar, BaseFunc> new_subroutines;

  RewriteSpec RewriteBindings(const Array<Binding>& bindings) const override;

  void VisitAttrs(AttrVisitor* visitor) {
    visitor->Visit("patterns", &patterns);
    PackedFunc untyped_func = func;
    visitor->Visit("func", &untyped_func);
  }

  static constexpr const char* _type_key = "relax.dpl.TupleRewriter";
  TVM_DECLARE_BASE_OBJECT_INFO(TupleRewriterNode, ExprRewriterNode);

 private:
  struct VarInfo {
    Var var;
    Expr expr;
    Array<Optional<Map<DFPattern, Expr>>> matches;
    std::unordered_set<Var> downstream_usage;
    bool used = false;
  };

  Map<Var, Expr> GenerateVariableRewrites(const Array<Binding>& bindings) const;

  std::optional<std::vector<Expr>> TryMatchByBindingIndex(const std::vector<VarInfo>& info_vec,
                                                          const std::vector<size_t>& indices) const;
};

class TupleRewriter : public ExprRewriter {
 public:
  TupleRewriter(Array<DFPattern> patterns,
                TypedPackedFunc<Optional<Expr>(Expr, Map<DFPattern, Expr>)> func,
                Optional<Array<DFPattern>> additional_bindings = NullOpt,
                Map<GlobalVar, BaseFunc> new_subroutines = {});

  TVM_DEFINE_OBJECT_REF_METHODS(TupleRewriter, ExprRewriter, TupleRewriterNode);
};

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_IR_DATAFLOW_REWRITER_H_
