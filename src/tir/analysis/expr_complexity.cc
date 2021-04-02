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
 * \file tir/analysis/expr_complexity.cc
 * \brief Calculate expr complexity.
 */
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
namespace tir {

/*! \brief Count the size of the PrimExpr. */
class PrimExprSizeCounter : public ExprVisitor {
 public:
  PrimExprSizeCounter() = default;

  static size_t Count(const PrimExpr& expr) {
    PrimExprSizeCounter prim_expr_size_counter;
    prim_expr_size_counter.VisitExpr(expr);
    return prim_expr_size_counter.counter_;
  }

 private:
  void VisitExpr(const PrimExpr& expr) final {
    counter_++;
    ExprVisitor::VisitExpr(expr);
  }

  size_t counter_{0};
};

size_t CalculateExprComplexity(const PrimExpr& expr) { return PrimExprSizeCounter::Count(expr); }

}  // namespace tir
}  // namespace tvm
