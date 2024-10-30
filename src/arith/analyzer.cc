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

#include "./scalable_expression.h"
#include "const_fold.h"
#include "product_normal_form.h"

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

void Analyzer::MarkGlobalNonNegValue(const PrimExpr& value) {
  // decompose value as symbol * scale + offset
  int64_t offset = 0;
  PrimExpr symbol_scale = tir::make_const(value.dtype(), 0);

  auto fcollect_sum = [&](PrimExpr val, int sign) {
    if (const auto* intimm = val.as<IntImmNode>()) {
      offset += intimm->value * sign;
    } else {
      if (sign > 0) {
        symbol_scale = symbol_scale + val;
      } else {
        symbol_scale = symbol_scale - val;
      }
    }
  };
  UnpackSum(value, fcollect_sum);

  // split out the symbol and non-symbolic part
  int64_t cscale = 1;
  PrimExpr symbol = tir::make_const(value.dtype(), 1);
  auto fcollect_prod = [&](PrimExpr val) {
    if (const auto* intimm = val.as<IntImmNode>()) {
      cscale *= intimm->value;
    } else {
      symbol = symbol * val;
    }
  };
  UnpackReduction<tir::MulNode>(symbol_scale, fcollect_prod);
  if (cscale <= 0) return;
  // override the constant int bound by marking it as non-negative
  // NOTE: there might be future opportunities of more bound hint
  // this is a simple step and covers all the current needs
  //
  // We may consider enhance the sub analyzer to directly take
  // MarkPositiveVar so their bounds do not overlap
  if (const auto* var_ptr = symbol.as<VarNode>()) {
    Var var = GetRef<Var>(var_ptr);
    // skip non-index type, keep it to be compatible
    // with any_dim that do not represent any value
    if (!IsIndexType(var.dtype())) return;
    bool allow_override = true;
    // mark the constant bound is sufficient
    // we cannot mark interval set as that will cause relaxation of the var
    // during bound proof which is not our intention
    this->const_int_bound.Update(var, ConstIntBound(-offset, ConstIntBound::kPosInf),
                                 allow_override);
  }
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

bool Analyzer::CanProveLessEqualThanSymbolicShapeValue(const PrimExpr& lhs, const PrimExpr& shape) {
  if (this->CanProve(lhs <= shape, ProofStrength::kSymbolicBound)) return true;
  // no need to do further attempt if shape is already a constant.
  if (tir::is_const_int(shape)) return false;
  // collect constant scale and ignore symbolic part
  // so 32 * n => cscale = 32
  int64_t cscale = 1;
  auto fcollect = [&](const PrimExpr& expr) {
    if (auto* ptr = expr.as<IntImmNode>()) {
      cscale *= ptr->value;
    }
  };
  UnpackReduction<tir::MulNode>(shape, fcollect);
  PrimExpr const_shape_bound = IntImm(shape.dtype(), std::abs(cscale));
  if (this->CanProve(lhs <= const_shape_bound, ProofStrength::kSymbolicBound)) return true;
  return false;
}

bool Analyzer::CanProve(const PrimExpr& expr, ProofStrength strength) {
  // Avoid potentially expensive simplification unless required.
  if (const auto* ptr = expr.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  PrimExpr simplified = Simplify(expr);
  const int64_t* as_int = tir::as_const_int(simplified);
  if (as_int && *as_int) return true;
  if (strength >= ProofStrength::kSymbolicBound) {
    // NOTE: we intentionally only pattern match common bound predicate i < bound
    // and put this implementation at the top-level.
    // This is to avoid repeatitive calling of this function
    // that causes speed issues.
    // This strategy can only be called from top-level and not from sub-analyzers.
    Optional<PrimExpr> pos_diff;
    int lower_bound = 0;
    if (const auto* ptr_lt = expr.as<tir::LTNode>()) {
      pos_diff = ptr_lt->b - ptr_lt->a;
      lower_bound = 1;
    }
    if (const auto* ptr_le = expr.as<tir::LENode>()) {
      pos_diff = ptr_le->b - ptr_le->a;
      lower_bound = 0;
    }
    if (const auto* ptr_gt = expr.as<tir::GTNode>()) {
      pos_diff = ptr_gt->a - ptr_gt->b;
      lower_bound = 1;
    }
    if (const auto* ptr_ge = expr.as<tir::GENode>()) {
      pos_diff = ptr_ge->a - ptr_ge->b;
      lower_bound = 0;
    }
    if (pos_diff) {
      IntSet iset = this->int_set(this->Simplify(pos_diff.value()));
      if (iset.HasLowerBound()) {
        ConstIntBound relaxed_lower_bound = this->const_int_bound(this->Simplify(iset.min()));
        if (relaxed_lower_bound->min_value >= lower_bound) return true;
      }
    }
  }

  // Current analysis may not be powerful enough to prove expressions containing
  // the same symbolic value multiple times. However, when the symbolic values are
  // "T.vscale" and the compile target uses a scalable architecture extension like
  // SVE, we can make some assumptions about the value of vscale and iterate over a
  // space of pre-defined values to attempt to prove the expression.
  Target curr_target = Target::Current();
  if (ContainsVscaleCall(simplified)) {
    if (TargetHasSVE(curr_target)) {
      return CanProveVscaleExpressionFromKnownValues(this, simplified, kAArch64VScaleValues);
    }
    LOG(WARNING)
        << "The expression contains scalable values. An attempt to prove by substituting "
           "with known values of vscale was not performed. This proof currently only supports "
           "AArch64 SVE targets, but the target was "
        << curr_target;
  }
  return false;
}

PrimExpr Analyzer::Simplify(const PrimExpr& expr, int steps) {
  PrimExpr res = expr;

  // Always starts with a canonical simplification, as some structural property
  // of an expression might be destroyed by rewrite simplification.
  res = this->canonical_simplify(res);

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
    } else if (name == "get_rewrite_simplify_stats") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        *ret = self->rewrite_simplify.GetStatsCounters();
      });
    } else if (name == "reset_rewrite_simplify_stats") {
      return PackedFunc(
          [self](TVMArgs args, TVMRetValue* ret) { self->rewrite_simplify.ResetStatsCounters(); });
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
    } else if (name == "can_prove") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        int strength = args[1];
        *ret = self->CanProve(args[0], static_cast<ProofStrength>(strength));
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
    } else if (name == "get_enabled_extensions") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        *ret = static_cast<std::int64_t>(self->rewrite_simplify.GetEnabledExtensions());
      });
    } else if (name == "set_enabled_extensions") {
      return PackedFunc([self](TVMArgs args, TVMRetValue* ret) {
        std::int64_t flags = args[0];
        self->rewrite_simplify.SetEnabledExtensions(
            static_cast<RewriteSimplifier::Extension>(flags));
      });
    }
    return PackedFunc();
  };
  *ret = TypedPackedFunc<PackedFunc(std::string)>(f);
});

}  // namespace arith
}  // namespace tvm
