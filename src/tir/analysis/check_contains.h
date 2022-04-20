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
 * \file check_contains.h
 * \brief Interface of the analysis that tells if an expression contains
           a node that satisfies a given predicate.
 */

#ifndef TVM_TIR_ANALYSIS_CHECK_CONTAINS_H_
#define TVM_TIR_ANALYSIS_CHECK_CONTAINS_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>  // For the class StmtExprVisitor

namespace tvm {
namespace tir {

/*!
 * \brief Visitor which tells if a given expression or statement contains a subexpression
          that satisfies a given predicate
 */
class CheckContains : public StmtExprVisitor {
 public:
  // Toplevel (static) functions
  static bool ExprContains(const PrimExpr& expr, std::function<bool(const PrimExpr&)> predicate);
  static bool StmtContains(const Stmt& stmt, std::function<bool(const PrimExpr&)> predicate);

 protected:
  // Constructor
  explicit CheckContains(std::function<bool(const PrimExpr&)> predicate);

  void VisitExpr(const PrimExpr& expr) override;
  void VisitStmt(const Stmt& stmt) override;

 private:
  std::function<bool(const PrimExpr&)> predicate_;
  bool contains_it_ = false;
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_ANALYSIS_CHECK_CONTAINS_H_
