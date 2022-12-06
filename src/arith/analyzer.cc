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
 * \file tvm/arith/analyzer.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

Analyzer::Analyzer()
    : const_int_bound(this),
      modular_set(this),
      rewrite_simplify(this),
      canonical_simplify(this),
      int_set(this) {}

void Analyzer::Bind(const Var& var, const PrimExpr& expr, bool allow_override) {
  PrimExpr new_expr = expr;
  new_expr = this->canonical_simplify(new_expr);
  new_expr = this->rewrite_simplify(new_expr);

  this->const_int_bound.Update(var, this->const_int_bound(new_expr), allow_override);
  this->modular_set.Update(var, this->modular_set(new_expr), allow_override);
  this->rewrite_simplify.Update(var, new_expr, allow_override);
  this->canonical_simplify.Update(var, new_expr, allow_override);
  this->int_set.Update(var, this->int_set(new_expr), allow_override);
  this->transitive_comparisons.Bind(var, expr, allow_override);
}

void Analyzer::Bind(const Var& var, const Range& range, bool allow_override) {
  ICHECK(range.defined());
  if (tir::is_one(range->extent)) {
    this->Bind(var, range->min, allow_override);
  } else {
    this->const_int_bound.Bind(var, range, allow_override);
    this->int_set.Bind(var, range, allow_override);
    this->transitive_comparisons.Bind(var, range, allow_override);
  }
  // skip modular_set
  // skip rewrite simplify
}

void Analyzer::Bind(const Map<Var, Range>& variables, bool allow_override) {
  for (const auto& iter : variables) {
    this->Bind(iter.first, iter.second, allow_override);
  }
}

void ConstraintContext::EnterWithScope() {
  ICHECK(recovery_functions_.size() == 0);
  // entering the scope.
  recovery_functions_.push_back(analyzer_->const_int_bound.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->modular_set.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->rewrite_simplify.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->int_set.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->transitive_comparisons.EnterConstraint(constraint_));
}

void ConstraintContext::ExitWithScope() {
  while (recovery_functions_.size()) {
    auto& func = recovery_functions_.back();
    if (func) {
      func();
    }
    recovery_functions_.pop_back();
  }
}

bool Analyzer::CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound) {
  if (const auto* ptr = expr.as<tir::IntImmNode>()) {
    return ptr->value >= lower_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->min_value >= lower_bound) return true;
  return false;
}

bool Analyzer::CanProveLess(const PrimExpr& expr, int64_t upper_bound) {
  if (const auto* ptr = expr.as<tir::IntImmNode>()) {
    return ptr->value < upper_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->max_value < upper_bound) return true;
  return false;
}

bool Analyzer::CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs) {
  const auto* clhs = lhs.as<IntImmNode>();
  const auto* crhs = rhs.as<IntImmNode>();
  if (clhs && crhs) return clhs->value == crhs->value;
  if (lhs->dtype.is_handle() || rhs->dtype.is_handle()) {
    return lhs.same_as(rhs);
  }
  return CanProve(lhs - rhs == 0);
}

bool Analyzer::CanProve(const PrimExpr& expr) {
  // Avoid potentially expensive simplification unless required.
  if (const auto* ptr = expr.as<IntImmNode>()) {
    return ptr->value != 0;
  }

  PrimExpr simplified = Simplify(expr);
  const int64_t* as_int = tir::as_const_int(simplified);
  return as_int && *as_int;
}

PrimExpr Analyzer::Simplify(const PrimExpr& expr, int steps) {
  PrimExpr res = expr;

  for (int i = 0; i < steps; ++i) {
    if (tir::is_const_int(res)) {
      return res;
    }
    if (i % 2 == 0) {
      res = this->rewrite_simplify(res);
    } else {
      res = this->canonical_simplify(res);
    }
  }

  return res;
}

TVM_REGISTER_GLOBAL("arith.CreateAnalyzer").set_body([](TVMArgs args, TVMRetValue* ret) {
  using runtime::PackedFunc;
  using runtime::TypedPackedFunc;
  auto self = std::make_shared<Analyzer>();
  auto f = [self](std::string name) -> PackedFunc {
    if (name == "const_int_bound") {
      return PackedFunc(
          [self](TVMArgs args, TVMRetValue* ret) { *ret = self->const_int_bound(args[0]); });
    } else if (name == "modular_set") {
      return PackedFunc(
          [self](TVMArgs args, TVMRetValue* ret) { *ret = self->modular_set(args[0]); });
    } else if (name == "const_int_bound_update") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        self->const_int_bound.Update(args[0], args[1], args[2]);
      });
    } else if (name == "Simplify") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        if (args.size() == 1) {
          *ret = self->Simplify(args[0]);
        } else if (args.size() == 2) {
          *ret = self->Simplify(args[0], args[1]);
        } else {
          LOG(FATAL) << "Invalid size of argument (" << args.size() << ")";
        }
      });
    } else if (name == "rewrite_simplify") {
      return PackedFunc(
          [self](TVMArgs args, TVMRetValue* ret) { *ret = self->rewrite_simplify(args[0]); });
    } else if (name == "canonical_simplify") {
      return PackedFunc(
          [self](TVMArgs args, TVMRetValue* ret) { *ret = self->canonical_simplify(args[0]); });
    } else if (name == "int_set") {
      return PackedFunc(
          [self](TVMArgs args, TVMRetValue* ret) { *ret = self->int_set(args[0], args[1]); });
    } else if (name == "bind") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        if (args[1].IsObjectRef<Range>()) {
          self->Bind(args[0], args[1].operator Range());
        } else {
          self->Bind(args[0], args[1].operator PrimExpr());
        }
      });
    } else if (name == "enter_constraint_context") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        // can't use make_shared due to noexcept(false) decl in destructor,
        // see https://stackoverflow.com/a/43907314
        auto ctx = std::shared_ptr<With<ConstraintContext>>(
            new With<ConstraintContext>(self.get(), args[0]));
        auto fexit = [ctx](TVMArgs, TVMRetValue*) mutable { ctx.reset(); };
        *ret = PackedFunc(fexit);
      });
    } else if (name == "can_prove_equal") {
      return PackedFunc(
          [self](TVMArgs args, TVMRetValue* ret) { *ret = self->CanProveEqual(args[0], args[1]); });
    }
    return PackedFunc();
  };
  *ret = TypedPackedFunc<PackedFunc(std::string)>(f);
});

}  // namespace arith
}  // namespace tvm
