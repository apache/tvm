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
 * \file span_check.h
 * \brief Check that the Relay IR has correctly attached span information.
 */
#ifndef TVM_RELAY_PARSER_SPAN_CHECK_H_
#define TVM_RELAY_PARSER_SPAN_CHECK_H_

#include <tvm/ir/transform.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relay {

using namespace tvm::relay;
using tvm::transform::Pass;

struct SpanChecker : ExprVisitor {
  Expr expression;
  DiagnosticContext diag_ctx;
  std::vector<Span> span_stack;

  explicit SpanChecker(DiagnosticContext diag_ctx) : diag_ctx(diag_ctx) {}

  void VisitExpr(const Expr& expr) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const GlobalVarNode* op) override;
  void VisitExpr_(const ConstantNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const FunctionNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const LetNode* op) override;
  void VisitExpr_(const IfNode* op) override;
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const TupleGetItemNode* op) override;
  void VisitExpr_(const RefCreateNode* op) override;
  void VisitExpr_(const RefReadNode* op) override;
  void VisitExpr_(const RefWriteNode* op) override;
  void VisitExpr_(const ConstructorNode* op) override;
  void VisitExpr_(const MatchNode* op) override;
  void VisitType(const Type& t) override;
  void VisitClause(const Clause& c) override;
  void VisitPattern(const Pattern& c) override;
  void VisitSpan(const Span& span) override;
};

Pass SpanCheck();

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_PARSER_SPAN_CHECK_H_
