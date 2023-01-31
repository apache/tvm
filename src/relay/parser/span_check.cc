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
 * \file span_check.cc
 * \brief A utility for checking and reporting malformed span information.
 */
#include "./span_check.h"

#include <tvm/relay/transform.h>

namespace tvm {
namespace relay {

using tvm::relay::transform::CreateFunctionPass;
using tvm::transform::PassContext;

void SpanChecker::VisitExpr(const Expr& e) {
  this->expression = e;
  VisitSpan(e->span);
  span_stack.push_back(e->span);
  ExprVisitor::VisitExpr(e);
  this->expression = e;
  span_stack.pop_back();
}

// TODO(@jroesch, @junru): we need to deal with unique spans for global/var.
void SpanChecker::VisitExpr_(const VarNode* op) {}
void SpanChecker::VisitExpr_(const GlobalVarNode* op) {}
void SpanChecker::VisitExpr_(const ConstantNode* op) {}

void SpanChecker::VisitExpr_(const TupleNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const FunctionNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const CallNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const LetNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const IfNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const OpNode* op) {}

void SpanChecker::VisitExpr_(const TupleGetItemNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const RefCreateNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const RefReadNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const RefWriteNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const ConstructorNode* op) {}  // ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitExpr_(const MatchNode* op) { ExprVisitor::VisitExpr_(op); }

void SpanChecker::VisitSpan(const Span& sp) {
  if (!sp.defined()) {
    Span span;
    for (auto spans = this->span_stack.rbegin(); spans != this->span_stack.rend(); spans++) {
      span = this->span_stack.back();
      if (span.defined()) {
        diag_ctx.Emit(Diagnostic::Warning(span) << "found null-span, i-nodes deep from this span.");
        return;
      }
    }
    auto warning = Diagnostic::Warning(span);
    warning << "\tAll spans are null\n";
    warning << "\t" << this->expression;
    diag_ctx.Emit(warning);
  }
}

void SpanChecker::VisitType(const Type& t) {}
void SpanChecker::VisitClause(const Clause& c) {}
void SpanChecker::VisitPattern(const Pattern& c) {}

Pass SpanCheck() {
  return CreateFunctionPass(
      [](const Function& func, const IRModule& mod, const PassContext& ctx) {
        ICHECK(ctx->diag_ctx) << "Diagnostic context must be set.";
        SpanChecker checker(ctx->diag_ctx.value());
        checker.VisitExpr(func);
        ctx->diag_ctx.value().Render();
        return func;
      },
      0, "SpanCheck", {});
}

TVM_REGISTER_GLOBAL("relay.parser.SpanCheck").set_body_typed([]() { return SpanCheck(); });

}  // namespace relay
}  // namespace tvm
