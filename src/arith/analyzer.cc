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
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace arith {

Analyzer::Analyzer()
    : const_int_bound(this),
      modular_set(this),
      rewrite_simplify(this),
      canonical_simplify(this),
      int_set(this) {
}

void Analyzer::Bind(const Var& var, const PrimExpr& expr, bool override) {
  PrimExpr new_expr = expr;
  new_expr = this->canonical_simplify(new_expr);
  new_expr = this->rewrite_simplify(new_expr);

  this->const_int_bound.Update(var, this->const_int_bound(new_expr), override);
  this->modular_set.Update(var, this->modular_set(new_expr), override);
  this->rewrite_simplify.Update(var, new_expr, override);
  this->canonical_simplify.Update(var, new_expr, override);
}

void Analyzer::Bind(const Var& var, const Range& range, bool override) {
  CHECK(range.defined());
  if (tir::is_one(range->extent)) {
    this->Bind(var, range->min, override);
  } else {
    this->const_int_bound.Bind(var, range, override);
  }
  // skip modular_set
  // skip rewrite simplify
}

void Analyzer::Bind(const Map<Var, Range>& variables, bool override) {
  for (const auto& iter : variables) {
    this->Bind(iter.first, iter.second, override);
  }
}

void ConstraintContext::EnterWithScope() {
  CHECK(exit_ == nullptr);
  // entering the scope.
  auto f0 = analyzer_->const_int_bound.EnterConstraint(constraint_);
  auto f1 = analyzer_->modular_set.EnterConstraint(constraint_);
  auto f2 = analyzer_->rewrite_simplify.EnterConstraint(constraint_);
  // recovery function.
  exit_ = [f0, f1, f2]() {
    if (f2 != nullptr) f2();
    if (f1 != nullptr) f1();
    if (f0 != nullptr) f0();
  };
}

void ConstraintContext::ExitWithScope() {
  CHECK(exit_ != nullptr);
  exit_();
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

bool Analyzer::CanProve(const PrimExpr& expr) {
  if (const auto* ptr = expr.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  auto res = this->rewrite_simplify(expr);
  if (const auto* ptr = res.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  res = this->canonical_simplify(expr);
  if (const auto* ptr = res.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  return false;
}

PrimExpr Analyzer::Simplify(const PrimExpr& expr) {
  if (tir::is_const(expr)) return expr;
  auto res = this->rewrite_simplify(expr);
  if (tir::is_const(res)) return res;
  res = this->canonical_simplify(res);
  return res;
}

TVM_REGISTER_GLOBAL("arith.CreateAnalyzer")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    using runtime::PackedFunc;
    using runtime::TypedPackedFunc;
    auto self = std::make_shared<Analyzer>();
    auto f = [self](std::string name) -> PackedFunc {
      if (name == "const_int_bound") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->const_int_bound(args[0]);
          });
      } else if (name == "modular_set") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->modular_set(args[0]);
        });
      } else if (name == "const_int_bound_update") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            self->const_int_bound.Update(args[0], args[1], args[2]);
        });
      } else if (name == "Simplify") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->Simplify(args[0]);
        });
      } else if (name == "rewrite_simplify") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->rewrite_simplify(args[0]);
        });
      } else if (name == "canonical_simplify") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->canonical_simplify(args[0]);
        });
      } else if (name == "int_set") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            *ret = self->int_set(args[0], args[1]);
        });
      } else if (name == "bind") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            if (args[1].IsObjectRef<Range>()) {
              self->Bind(args[0], args[1].operator Range());
            } else {
              self->Bind(args[0], args[1].operator PrimExpr());
            }
        });
      } else if (name == "enter_constraint_context") {
        return PackedFunc([self](TVMArgs args, TVMRetValue *ret) {
            // can't use make_shared due to noexcept(false) decl in destructor,
            // see https://stackoverflow.com/a/43907314
            auto ctx = std::shared_ptr<With<ConstraintContext> >(
                new With<ConstraintContext>(self.get(), args[0]));
            auto fexit = [ctx](TVMArgs, TVMRetValue*) mutable {
              ctx.reset();
            };
            *ret = PackedFunc(fexit);
        });
      }
      return PackedFunc();
    };
    *ret = TypedPackedFunc<PackedFunc(std::string)>(f);
});

}  // namespace arith
}  // namespace tvm
