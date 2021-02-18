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
 * \brief Calcute expr complexity.
 */
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr_functor.h>

namespace tvm {
namespace tir {

#define PLUS_ONE(OP) \
  void VisitExpr_(const OP* op) final { num_symbols_++; }

#define PLUS_ONE_BINARY(OP)             \
  void VisitExpr_(const OP* op) final { \
    num_symbols_++;                     \
    VisitExpr(op->a);                   \
    VisitExpr(op->b);                   \
  }

/*!
 * \brief Calculate the expresion complexity based on number of symbols it contains.
 */
class ExprComplexity : public ExprVisitor {
 public:
  size_t Eval(const PrimExpr& expr) {
    VisitExpr(expr);
    return num_symbols_;
  }

  PLUS_ONE_BINARY(AddNode)
  PLUS_ONE_BINARY(SubNode)
  PLUS_ONE_BINARY(MulNode)
  PLUS_ONE_BINARY(DivNode)
  PLUS_ONE_BINARY(ModNode)
  PLUS_ONE_BINARY(FloorDivNode)
  PLUS_ONE_BINARY(FloorModNode)
  PLUS_ONE_BINARY(MinNode)
  PLUS_ONE_BINARY(MaxNode)
  PLUS_ONE_BINARY(EQNode)
  PLUS_ONE_BINARY(NENode)
  PLUS_ONE_BINARY(LTNode)
  PLUS_ONE_BINARY(LENode)
  PLUS_ONE_BINARY(GTNode)
  PLUS_ONE_BINARY(GENode)
  PLUS_ONE_BINARY(AndNode)
  PLUS_ONE_BINARY(OrNode)
  PLUS_ONE(VarNode)
  PLUS_ONE(FloatImmNode)
  PLUS_ONE(IntImmNode)
  void VisitExpr_(const NotNode* op) final {
    num_symbols_++;
    VisitExpr(op->a);
  }

 private:
  size_t num_symbols_{0};
};

size_t CalculateExprComplexity(const PrimExpr& expr) { return ExprComplexity().Eval(expr); }

}  // namespace tir
}  // namespace tvm
