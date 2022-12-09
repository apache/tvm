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
 * \file narrow_predicate_expression.cc
 * \brief Utility to deduce bound of expression
 */
#include <tvm/arith/int_solver.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {

using namespace tir;

/* \brief Given a true expression that includes free parameter,
 * generate a true expression without the free parameters.
 *
 * This function provides two guarantees:
 *
 * 1. If the resulting expression evaluates to True, then the original
 * expression also evaluates to True.
 *
 * 2. The resulting expression does not contain any of the free
 * parameters.
 *
 */
// Utility for generating a known true expression from an expression
// with free parameters, and the range of those parameters.
class ExpressionNarrower : public tir::ExprMutator {
 public:
  static PrimExpr Apply(PrimExpr expr, Map<Var, Range> free_parameters) {
    ICHECK(expr.dtype().is_bool()) << "Expected boolean expression, but received " << expr;
    ExpressionNarrower mutator(free_parameters);
    return mutator(expr);
  }

 private:
  explicit ExpressionNarrower(Map<Var, Range> free_parameters)
      : free_parameters_(free_parameters) {}

  using Parent = tir::ExprMutator;
  using Parent::VisitExpr_;

  enum class Context {
    Maximize,
    Minimize,
  };

  template <typename T>
  PrimExpr VisitInequality(T t, Context a_ctx, Context b_ctx) {
    PrimExpr a = [&]() {
      WithContext context(this, a_ctx);
      return this->VisitExpr(t->a);
    }();

    PrimExpr b = [&]() {
      WithContext context(this, b_ctx);
      return this->VisitExpr(t->b);
    }();

    if (contains_unknown_expr_ && t.dtype().is_bool()) {
      contains_unknown_expr_ = false;
      return Bool(CurrentContext() == Context::Minimize);
    } else if (a.same_as(t->a) && b.same_as(t->b)) {
      return std::move(t);
    } else {
      return T(a, b);
    }
  }

  PrimExpr VisitExpr_(const FloorModNode* op) override {
    // FloorMod is non-monotonic, so inserting min/max won't remove
    // the free parameters.
    contains_unknown_expr_ = true;
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const FloorDivNode* op) override {
    auto res_a = this->VisitExpr(op->a);
    auto res_b = this->VisitExpr(op->b);
    if (is_zero(res_b)) {
      contains_unknown_expr_ = true;
      return IntImm(op->dtype, 0);
    } else {
      return floordiv(res_a, res_b);
    }
  }

  PrimExpr VisitExpr_(const GTNode* op) override {
    auto current = CurrentContext();
    return VisitInequality(GetRef<GT>(op), OppositeContext(current), current);
  }

  PrimExpr VisitExpr_(const GENode* op) override {
    auto current = CurrentContext();
    return VisitInequality(GetRef<GE>(op), OppositeContext(current), current);
  }

  PrimExpr VisitExpr_(const LTNode* op) override {
    auto current = CurrentContext();
    return VisitInequality(GetRef<LT>(op), current, OppositeContext(current));
  }

  PrimExpr VisitExpr_(const LENode* op) override {
    auto current = CurrentContext();
    return VisitInequality(GetRef<LE>(op), current, OppositeContext(current));
  }

  PrimExpr VisitExpr_(const EQNode* op) override {
    auto res_a = this->VisitExpr(op->a <= op->b);
    auto res_b = this->VisitExpr(op->b <= op->a);
    return res_a && res_b;
  }

  PrimExpr VisitExpr_(const NENode* op) override {
    auto res_a = this->VisitExpr(op->a < op->b);
    auto res_b = this->VisitExpr(op->b < op->a);
    return res_a || res_b;
  }

  PrimExpr VisitExpr_(const SubNode* op) override {
    auto current = CurrentContext();
    return VisitInequality(GetRef<Sub>(op), current, OppositeContext(current));
  }

  PrimExpr VisitExpr_(const NotNode* op) override {
    auto current = CurrentContext();
    WithContext context(this, OppositeContext(current));
    return !VisitExpr(op->a);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    contains_unknown_expr_ = true;
    return GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) override {
    auto it = free_parameters_.find(GetRef<Var>(op));
    if (it == free_parameters_.end()) {
      return Parent::VisitExpr_(op);
    }

    Range range = (*it).second;

    switch (CurrentContext()) {
      case Context::Minimize:
        return range->min;

      case Context::Maximize:
        return range->min + range->extent - 1;
    }

    return Parent::VisitExpr_(op);
  }

  Context CurrentContext() const {
    if (context_stack_.size()) {
      return context_stack_.back();
    } else {
      return Context::Maximize;
    }
  }

  Context OppositeContext(Context context) const {
    switch (context) {
      case Context::Minimize:
        return Context::Maximize;

      case Context::Maximize:
        return Context::Minimize;

      default:
        LOG(FATAL) << "Unhandled Context, all legal values should be handled";
    }
  }

  struct WithContext {
    WithContext(ExpressionNarrower* self, Context context) : self(self) {
      self->context_stack_.push_back(context);
    }
    ~WithContext() { self->context_stack_.pop_back(); }
    ExpressionNarrower* self;
  };

  std::vector<Context> context_stack_;
  Map<Var, Range> free_parameters_;
  bool contains_unknown_expr_{false};
};

PrimExpr NarrowPredicateExpression(PrimExpr expr, Map<Var, Range> free_parameters) {
  return ExpressionNarrower::Apply(std::move(expr), std::move(free_parameters));
}

TVM_REGISTER_GLOBAL("arith.NarrowPredicateExpression").set_body_typed(NarrowPredicateExpression);

}  // namespace arith
}  // namespace tvm
