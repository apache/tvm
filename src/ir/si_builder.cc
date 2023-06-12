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
 * \file tvm/ir/si_builder.cc
 * \brief Implementation for building a source info during rewriting expressions.
 */
#include <tvm/ir/si_builder.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/stmt_functor.h>

#include <vector>

namespace tvm {

using RelayExprSet = std::unordered_set<relay::Expr, ObjectPtrHash, ObjectPtrEqual>;
using PrimExprSet = std::unordered_set<PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
using StmtSet = std::unordered_set<tir::Stmt, ObjectPtrHash, ObjectPtrEqual>;

class RelayCollectSpans : public relay::ExprVisitor {
 public:
  explicit RelayCollectSpans(const RelayExprSet& inputs = {}) : inputs_(inputs) {}

  // From entry to inputs, recursively collect spans. The spans of inputs are included.
  Span CollectSpans(const relay::Expr& entry);

  void VisitExpr(const relay::Expr& expr) final;

 private:
  Array<Span> spans_;
  const RelayExprSet& inputs_;
};

void RelayCollectSpans::VisitExpr(const relay::Expr& expr) {
  if (visit_counter_.count(expr.get())) {
    return;
  }
  if (expr->span.defined()) {
    spans_.push_back(expr->span);
  }
  if (inputs_.find(expr) != inputs_.end()) {
    // becuase it returns directly, it should be recorded as visted manually.
    visit_counter_.insert({expr.get(), 1});
    return;
  }
  relay::ExprVisitor::VisitExpr(expr);
}

Span RelayCollectSpans::CollectSpans(const relay::Expr& entry) {
  VisitExpr(entry);
  return SequentialSpan(spans_);
}

class RelayRecursivelyFill : public relay::ExprMutator {
 public:
  explicit RelayRecursivelyFill(const Span& span, const RelayExprSet& inputs = {})
      : span_(span), inputs_(inputs) {}

  // From entry until inputs, recursively fill spans into expressions. Inputs are not filled.
  void Fill(const relay::Expr& entry);

  relay::Expr VisitExpr(const relay::Expr& expr) final;

 private:
  const Span& span_;
  const RelayExprSet& inputs_;
};

relay::Expr RelayRecursivelyFill::VisitExpr(const relay::Expr& expr) {
  if (inputs_.find(expr) != inputs_.end()) {
    return expr;
  }
  // Skip op node. Align with python frontend
  if (!expr.as<OpNode>()) {
    expr->span = span_;
  }

  return relay::ExprMutator::VisitExpr(expr);
}

void RelayRecursivelyFill::Fill(const relay::Expr& entry) { Mutate(entry); }

class TirCollectSpans : public tir::StmtExprVisitor {
 public:
  explicit TirCollectSpans(const PrimExprSet& expr_inputs = {}, const StmtSet& stmt_inputs = {})
      : expr_inputs_(expr_inputs), stmt_inputs_(stmt_inputs) {}

  void VisitExpr(const PrimExpr& expr) final;
  void VisitStmt(const tir::Stmt& stmt) final;

  bool IsInput(const PrimExpr& expr);
  bool IsInput(const tir::Stmt& stmt);

  // From entry to inputs, recursively collect spans. The spans of inputs are included.
  Span CollectSpans(const PrimExpr& expr);
  // From entry to inputs, recursively collect spans. The spans of inputs are included.
  Span CollectSpans(const tir::Stmt& stmt);

 private:
  Array<Span> spans_;
  std::unordered_map<const Object*, size_t> visit_counter_;
  const PrimExprSet& expr_inputs_;
  const StmtSet& stmt_inputs_;
};

Span TirCollectSpans::CollectSpans(const PrimExpr& expr) {
  operator()(expr);
  return SequentialSpan(spans_);
}

Span TirCollectSpans::CollectSpans(const tir::Stmt& stmt) {
  operator()(stmt);
  return SequentialSpan(spans_);
}

bool TirCollectSpans::IsInput(const PrimExpr& expr) {
  return expr_inputs_.find(expr) != expr_inputs_.end();
}

bool TirCollectSpans::IsInput(const tir::Stmt& stmt) {
  return stmt_inputs_.find(stmt) != stmt_inputs_.end();
}

void TirCollectSpans::VisitExpr(const PrimExpr& expr) {
  if (visit_counter_.count(expr.get())) {
    return;
  }
  if (expr->span.defined()) {
    spans_.push_back(expr->span);
  }
  if (IsInput(expr)) {
    // becuase it returns directly, it should be recorded as visted manually.
    visit_counter_.insert({expr.get(), 1});
    return;
  }
  StmtExprVisitor::VisitExpr(expr);
}

void TirCollectSpans::VisitStmt(const tir::Stmt& stmt) {
  if (visit_counter_.count(stmt.get())) {
    return;
  }
  if (stmt->span.defined()) {
    spans_.push_back(stmt->span);
  }
  if (IsInput(stmt)) {
    // becuase it returns directly, it should be recorded as visted manually.
    visit_counter_.insert({stmt.get(), 1});
    return;
  }
  StmtExprVisitor::VisitStmt(stmt);
}

class TirRecursivelyFill : public tir::StmtExprMutator {
 public:
  TirRecursivelyFill(const Span& span, const PrimExprSet& expr_inputs = {},
                     const StmtSet& stmt_inputs = {})
      : span_(span), expr_inputs_(expr_inputs), stmt_inputs_(stmt_inputs) {}

  // From entry until inputs, recursively fill spans into expressions. Inputs are not filled.
  tir::Stmt Fill(const tir::Stmt& s) { return operator()(s); }
  // From entry until inputs, recursively fill spans into expressions. Inputs are not filled.
  PrimExpr Fill(const PrimExpr& e) { return operator()(e); }

  bool IsInput(const PrimExpr& expr);
  bool IsInput(const tir::Stmt& stmt);

  PrimExpr VisitExpr(const PrimExpr& expr) final;
  tir::Stmt VisitStmt(const tir::Stmt& stmt) final;

 private:
  const Span& span_;
  const PrimExprSet& expr_inputs_;
  const StmtSet& stmt_inputs_;
};

tir::Stmt TirRecursivelyFill::VisitStmt(const tir::Stmt& stmt) {
  if (IsInput(stmt)) {
    return stmt;
  }
  stmt->span = span_;
  return StmtExprMutator::VisitStmt(stmt);
}

bool TirRecursivelyFill::IsInput(const PrimExpr& expr) {
  return expr_inputs_.find(expr) != expr_inputs_.end();
}

bool TirRecursivelyFill::IsInput(const tir::Stmt& stmt) {
  return stmt_inputs_.find(stmt) != stmt_inputs_.end();
}

PrimExpr TirRecursivelyFill::VisitExpr(const PrimExpr& expr) {
  if (IsInput(expr)) {
    return expr;
  }
  expr->span = span_;
  return StmtExprMutator::VisitExpr(expr);
}

struct SIBuilder::Impl {
  virtual ~Impl() {}
  virtual Span Build() const { return Span(); }
  virtual void RecursivelyFillSpan(const relay::Expr& entry, const RelayExprSet& inputs) const {}
  virtual void RecursivelyFillSpan(const PrimExpr& entry, const PrimExprSet& inputs) const {}
  virtual void RecursivelyFillSpan(const tir::Stmt& entry, const PrimExprSet& inputs) const {}
  virtual void RecursivelyFillSpan(const tir::Stmt& entry, const StmtSet& inputs) const {}
  virtual void CollectSpansSpan(const relay::Expr& entry, const RelayExprSet& inputs) {}
  virtual void CollectSpansSpan(const PrimExpr& entry, const PrimExprSet& inputs) {}
  virtual void CollectSpansSpan(const tir::Stmt& entry, const PrimExprSet& inputs) {}
  virtual void CollectSpansSpan(const tir::Stmt& entry, const StmtSet& inputs) {}
};

SIBuilder::~SIBuilder() = default;

Span SIBuilder::Build() const { return impl_->Build(); }

template <>
void SIBuilder::RecursivelyFillSpan(const relay::Expr& entry, const RelayExprSet& inputs) const {
  impl_->RecursivelyFillSpan(entry, inputs);
}

template <>
void SIBuilder::RecursivelyFillSpan(const PrimExpr& entry, const PrimExprSet& inputs) const {
  impl_->RecursivelyFillSpan(entry, inputs);
}

void SIBuilder::RecursivelyFillSpan(const tir::Stmt& entry, const PrimExprSet& inputs) const {
  impl_->RecursivelyFillSpan(entry, inputs);
}

void SIBuilder::RecursivelyFillSpan(const tir::Stmt& entry, const StmtSet& inputs) const {
  impl_->RecursivelyFillSpan(entry, inputs);
}

std::unique_ptr<SIBuilder::Impl> SIBuilder::CreateImpl(const Span& span) {
  struct Impl : public SIBuilder::Impl {
    explicit Impl(const Span& span) : span_(span) {}
    Span Build() const final { return span_; }
    void RecursivelyFillSpan(const relay::Expr& entry, const RelayExprSet& inputs) const final {
      RelayRecursivelyFill(Build(), inputs).Fill(entry);
    }
    void RecursivelyFillSpan(const PrimExpr& entry, const PrimExprSet& inputs) const final {
      TirRecursivelyFill(Build(), inputs).Fill(entry);
    }
    void RecursivelyFillSpan(const tir::Stmt& entry, const PrimExprSet& inputs) const final {
      TirRecursivelyFill(Build(), inputs).Fill(entry);
    }
    void RecursivelyFillSpan(const tir::Stmt& entry, const StmtSet& inputs) const final {
      TirRecursivelyFill(Build(), {}, inputs).Fill(entry);
    }
    void CollectSpansSpan(const relay::Expr& entry, const RelayExprSet& inputs) final {
      span_ = RelayCollectSpans(inputs).CollectSpans(entry);
    }
    void CollectSpansSpan(const PrimExpr& entry, const PrimExprSet& inputs) final {
      span_ = TirCollectSpans(inputs).CollectSpans(entry);
    }
    void CollectSpansSpan(const tir::Stmt& entry, const PrimExprSet& inputs) final {
      span_ = TirCollectSpans(inputs).CollectSpans(entry);
    }
    void CollectSpansSpan(const tir::Stmt& entry, const StmtSet& inputs) final {
      span_ = TirCollectSpans({}, inputs).CollectSpans(entry);
    }

   private:
    Span span_;
  };

  const bool enable_si_builder = transform::PassContext::Current()
                                     ->GetConfig<Bool>("ir.enable_si_builder", Bool(false))
                                     .value();

  if (enable_si_builder) {
    return std::make_unique<Impl>(span);
  }

  return std::make_unique<SIBuilder::Impl>();
}

SIBuilder::SIBuilder(const Span& span) : impl_(CreateImpl(span)) {}
SIBuilder::SIBuilder(const Array<Span>& spans) : impl_(CreateImpl(SequentialSpan(spans))) {}
SIBuilder::SIBuilder(const std::initializer_list<Span>& init)
    : impl_(CreateImpl(SequentialSpan(Array<Span>(init)))) {}

template <>
SIBuilder::SIBuilder(const relay::Expr& expr, const Array<relay::Expr>& inputs)
    : impl_(CreateImpl(Span())) {
  impl_->CollectSpansSpan(expr, RelayExprSet(inputs.begin(), inputs.end()));
}

template <>
SIBuilder::SIBuilder(const PrimExpr& expr, const Array<PrimExpr>& inputs)
    : impl_(CreateImpl(Span())) {
  impl_->CollectSpansSpan(expr, PrimExprSet(inputs.begin(), inputs.end()));
}

SIBuilder::SIBuilder(const tir::Stmt& s, const Array<PrimExpr>& inputs)
    : impl_(CreateImpl(Span())) {
  impl_->CollectSpansSpan(s, PrimExprSet(inputs.begin(), inputs.end()));
}

SIBuilder::SIBuilder(const tir::Stmt& s, const Array<tir::Stmt>& inputs)
    : impl_(CreateImpl(Span())) {
  impl_->CollectSpansSpan(s, StmtSet(inputs.begin(), inputs.end()));
}

// Register build pipeline related options
TVM_REGISTER_PASS_CONFIG_OPTION("ir.enable_si_builder", Bool);

}  // namespace tvm
