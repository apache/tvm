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
 *  Copyright (c) 2019 by Contributors
 * \file rewrite_simplify.h
 * \brief Rewrite-rule based simplification.
 */
#ifndef TVM_ARITHMETIC_REWRITE_SIMPLIFY_H_
#define TVM_ARITHMETIC_REWRITE_SIMPLIFY_H_

#include <tvm/arithmetic.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_mutator.h>
#include <unordered_map>
#include "const_fold.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using namespace ir;

/*!
 * \brief Rewrite-based simplifier.
 *
 * This class can be inheritated for other simplifiers.
 */
class RewriteSimplifier::Impl : public IRMutator {
 public:
  explicit Impl(Analyzer* parent)
      : parent_(parent) {}

  void Update(const Var& var, const Expr& info, bool override);
  Expr Mutate_(const Add* op, const Expr& self) override;
  Expr Mutate_(const Sub* op, const Expr& self) override;
  Expr Mutate_(const Mul* op, const Expr& self) override;
  Expr Mutate_(const Div* op, const Expr& self) override;
  Expr Mutate_(const Mod* op, const Expr& self) override;
  Expr Mutate_(const FloorDiv* op, const Expr& self) override;
  Expr Mutate_(const FloorMod* op, const Expr& self) override;
  Expr Mutate_(const Min* op, const Expr& self) override;
  Expr Mutate_(const Max* op, const Expr& self) override;
  Expr Mutate_(const EQ* op, const Expr& self) override;
  Expr Mutate_(const NE* op, const Expr& self) override;
  Expr Mutate_(const LT* op, const Expr& self) override;
  Expr Mutate_(const LE* op, const Expr& self) override;
  Expr Mutate_(const GT* op, const Expr& self) override;
  Expr Mutate_(const GE* op, const Expr& self) override;
  Expr Mutate_(const And* op, const Expr& self) override;
  Expr Mutate_(const Or* op, const Expr& self) override;
  Expr Mutate_(const Not* op, const Expr& self) override;
  Expr Mutate_(const Select* op, const Expr& self) override;
  Expr Mutate_(const Call* op, const Expr& self) override;

 protected:
  /*! \brief internal structure for comparison. */
  enum CompareResult {
    kUnknown,
    kEQ,
    kGT,
    kGE,
    kLT,
    kLE,
    kNE
  };
  // reference to the main analyzer
  Analyzer* parent_;
  // counter to record recursive rewrite depth.
  int recur_depth_{0};
  // internal variable map
  std::unordered_map<Var, Expr, ExprHash, ExprEqual> var_map_;
  // maximum number of recursion allowed during a single pass.
  static const constexpr int kMaxRecurDepth = 5;

  /*!
   * \brief try to compare x against val.
   * \param x The expression to be evaluated.
   * \param val The constant value.
   * \return comparison result.
   */
  CompareResult TryCompare(const Expr& x, int64_t val);

 private:
  // Whether x >= val
  bool CanProveGreaterEqual(const Expr& x, int64_t val) {
    return parent_->CanProveGreaterEqual(x, val);
  }
  // Whether x == val
  bool CanProveEqual(const Expr& x, int64_t val) {
    // TODO(tqchen) refer back to super-analyzer.
    return TryCompare(x, val) == kEQ;
  }

  // Recursive rewrite x
  // we limit maximum depth of recursive rewrite allowed to
  // avoid infinite loop
  Expr RecursiveRewrite(const Expr& x) {
    if (recur_depth_ >= kMaxRecurDepth) return x;
    ++recur_depth_;
    Expr res = Mutate(x);
    --recur_depth_;
    return res;
  }

  template<typename TA>
  PConstWithTypeLike<TA> ZeroWithTypeLike(const Pattern<TA>& pattern) {
    return PConstWithTypeLike<TA>(pattern.derived(), 0);
  }

  template<typename TA>
  PConstWithTypeLike<TA> OneWithTypeLike(const Pattern<TA>& pattern) {
    return PConstWithTypeLike<TA>(pattern.derived(), 1);
  }
};


}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITHMETIC_REWRITE_SIMPLIFY_H_
