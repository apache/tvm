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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include "const_fold.h"
#include "product_normal_form.h"

namespace tvm {
namespace arith {

AnalyzerObj::AnalyzerObj()
    : const_int_bound(this),
      modular_set(this),
      rewrite_simplify(this),
      canonical_simplify(this),
      int_set(this),
      z3_prover(this) {}

void AnalyzerObj::Bind(const Var& var, const PrimExpr& expr, bool allow_override) {
  PrimExpr new_expr = expr;
  new_expr = this->canonical_simplify(new_expr);
  new_expr = this->rewrite_simplify(new_expr);

  this->const_int_bound.Update(var, this->const_int_bound(new_expr), allow_override);
  this->modular_set.Update(var, this->modular_set(new_expr), allow_override);
  this->rewrite_simplify.Update(var, new_expr, allow_override);
  this->canonical_simplify.Update(var, new_expr, allow_override);
  this->int_set.Update(var, this->int_set(new_expr), allow_override);
  this->transitive_comparisons.Bind(var, expr, allow_override);
  this->z3_prover.Bind(var, expr, allow_override);
}

void AnalyzerObj::Bind(const Var& var, const Range& range, bool allow_override) {
  TVM_FFI_ICHECK(range.defined());
  if (tirx::is_one(range->extent)) {
    this->Bind(var, range->min, allow_override);
  } else {
    this->const_int_bound.Bind(var, range, allow_override);
    this->int_set.Bind(var, range, allow_override);
    this->transitive_comparisons.Bind(var, range, allow_override);
    this->z3_prover.Bind(var, range, allow_override);
  }
  // skip modular_set
  // skip rewrite simplify
}

void AnalyzerObj::MarkGlobalNonNegValue(const PrimExpr& value) {
  // decompose value as symbol * scale + offset
  int64_t offset = 0;
  PrimExpr symbol_scale = tirx::make_const(value.dtype(), 0);

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
  PrimExpr symbol = tirx::make_const(value.dtype(), 1);
  auto fcollect_prod = [&](PrimExpr val) {
    if (const auto* intimm = val.as<IntImmNode>()) {
      cscale *= intimm->value;
    } else {
      symbol = symbol * val;
    }
  };
  UnpackReduction<tirx::MulNode>(symbol_scale, fcollect_prod);
  if (cscale <= 0) return;
  // override the constant int bound by marking it as non-negative
  // NOTE: there might be future opportunities of more bound hint
  // this is a simple step and covers all the current needs
  //
  // We may consider enhance the sub analyzer to directly take
  // MarkPositiveVar so their bounds do not overlap
  if (const auto* var_ptr = symbol.as<VarNode>()) {
    Var var = ffi::GetRef<Var>(var_ptr);
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

void AnalyzerObj::Bind(const ffi::Map<Var, Range>& variables, bool allow_override) {
  for (const auto& iter : variables) {
    this->Bind(iter.first, iter.second, allow_override);
  }
}

void ConstraintContext::EnterWithScope() {
  TVM_FFI_ICHECK(recovery_functions_.size() == 0);
  // entering the scope.
  recovery_functions_.push_back(analyzer_->const_int_bound.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->modular_set.EnterConstraint(constraint_));
  recovery_functions_.push_back(
      analyzer_->rewrite_simplify.EnterConstraint(constraint_, is_assume_));
  recovery_functions_.push_back(analyzer_->int_set.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->transitive_comparisons.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->z3_prover.EnterConstraint(constraint_, is_assume_));
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

bool AnalyzerObj::CanProveGreaterEqual(const PrimExpr& expr, int64_t lower_bound) {
  if (const auto* ptr = expr.as<tirx::IntImmNode>()) {
    return ptr->value >= lower_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->min_value >= lower_bound) return true;
  return false;
}

bool AnalyzerObj::CanProveLess(const PrimExpr& expr, int64_t upper_bound) {
  if (const auto* ptr = expr.as<tirx::IntImmNode>()) {
    return ptr->value < upper_bound;
  }
  auto bd = this->const_int_bound(this->rewrite_simplify(expr));
  if (bd->max_value < upper_bound) return true;
  return false;
}

bool AnalyzerObj::CanProveEqual(const PrimExpr& lhs, const PrimExpr& rhs) {
  const auto* clhs = lhs.as<IntImmNode>();
  const auto* crhs = rhs.as<IntImmNode>();
  if (clhs && crhs) return clhs->value == crhs->value;
  if (lhs->dtype.is_handle() || rhs->dtype.is_handle()) {
    return lhs.same_as(rhs);
  }
  return CanProve(lhs - rhs == 0);
}

bool AnalyzerObj::CanProveLessEqualThanSymbolicShapeValue(const PrimExpr& lhs,
                                                          const PrimExpr& shape) {
  if (this->CanProve(lhs <= shape, ProofStrength::kSymbolicBound)) return true;
  // no need to do further attempt if shape is already a constant.
  if (tirx::is_const_int(shape)) return false;
  // collect constant scale and ignore symbolic part
  // so 32 * n => cscale = 32
  int64_t cscale = 1;
  auto fcollect = [&](const PrimExpr& expr) {
    if (auto* ptr = expr.as<IntImmNode>()) {
      cscale *= ptr->value;
    }
  };
  UnpackReduction<tirx::MulNode>(shape, fcollect);
  PrimExpr const_shape_bound = IntImm(shape.dtype(), std::abs(cscale));
  if (this->CanProve(lhs <= const_shape_bound, ProofStrength::kSymbolicBound)) return true;
  return false;
}

bool AnalyzerObj::CanProve(const PrimExpr& expr, ProofStrength strength) {
  // Avoid potentially expensive simplification unless required.
  if (const auto* ptr = expr.as<IntImmNode>()) {
    return ptr->value != 0;
  }
  PrimExpr simplified = Simplify(expr);
  const int64_t* as_int = tirx::as_const_int(simplified);
  if (as_int && *as_int) return true;
  if (strength >= ProofStrength::kSymbolicBound) {
    // NOTE: we intentionally only pattern match common bound predicate i < bound
    // and put this implementation at the top-level.
    // This is to avoid repeatitive calling of this function
    // that causes speed issues.
    // This strategy can only be called from top-level and not from sub-analyzers.
    ffi::Optional<PrimExpr> pos_diff;
    int lower_bound = 0;
    if (const auto* ptr_lt = expr.as<tirx::LTNode>()) {
      pos_diff = ptr_lt->b - ptr_lt->a;
      lower_bound = 1;
    }
    if (const auto* ptr_le = expr.as<tirx::LENode>()) {
      pos_diff = ptr_le->b - ptr_le->a;
      lower_bound = 0;
    }
    if (const auto* ptr_gt = expr.as<tirx::GTNode>()) {
      pos_diff = ptr_gt->a - ptr_gt->b;
      lower_bound = 1;
    }
    if (const auto* ptr_ge = expr.as<tirx::GENode>()) {
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

    if (z3_prover.CanProve(simplified)) {
      return true;
    }
  }
  return false;
}

PrimExpr AnalyzerObj::Simplify(const PrimExpr& expr, int steps) {
  PrimExpr res = expr;

  // Always starts with a canonical simplification, as some structural property
  // of an expression might be destroyed by rewrite simplification.
  res = this->canonical_simplify(res);

  for (int i = 0; i < steps; ++i) {
    if (tirx::is_const_int(res)) {
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

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<AnalyzerObj>();
  refl::GlobalDef()
      .def("arith.Analyzer", []() { return Analyzer(); })
      .def("arith.AnalyzerConstIntBound",
           [](Analyzer analyzer, const PrimExpr& expr) { return analyzer->const_int_bound(expr); })
      .def("arith.AnalyzerConstIntBoundUpdate",
           [](Analyzer analyzer, const Var& var, const ConstIntBound& info, bool allow_override) {
             analyzer->const_int_bound.Update(var, info, allow_override);
           })
      .def("arith.AnalyzerConstIntBoundIsBound",
           [](Analyzer analyzer, const Var& var) { return analyzer->const_int_bound.IsBound(var); })
      .def("arith.AnalyzerModularSetUpdate",
           [](Analyzer analyzer, const Var& var, const ModularSet& info, bool allow_override) {
             analyzer->modular_set.Update(var, info, allow_override);
           })
      .def("arith.AnalyzerIntSetUpdate",
           [](Analyzer analyzer, const Var& var, const IntSet& info, bool allow_override) {
             analyzer->int_set.Update(var, info, allow_override);
           })
      .def("arith.AnalyzerModularSet",
           [](Analyzer analyzer, const PrimExpr& expr) { return analyzer->modular_set(expr); })
      .def("arith.AnalyzerSimplify", [](Analyzer analyzer, const PrimExpr& expr,
                                        int steps) { return analyzer->Simplify(expr, steps); })
      .def("arith.AnalyzerRewriteSimplify",
           [](Analyzer analyzer, const PrimExpr& expr) { return analyzer->rewrite_simplify(expr); })
      .def("arith.AnalyzerGetRewriteSimplifyStats",
           [](Analyzer analyzer) { return analyzer->rewrite_simplify.GetStatsCounters(); })
      .def("arith.AnalyzerResetRewriteSimplifyStats",
           [](Analyzer analyzer) { analyzer->rewrite_simplify.ResetStatsCounters(); })
      .def("arith.AnalyzerCanonicalSimplify",
           [](Analyzer analyzer, const PrimExpr& expr) {
             return analyzer->canonical_simplify(expr);
           })
      .def("arith.AnalyzerIntSet",
           [](Analyzer analyzer, const PrimExpr& expr,
              ffi::Optional<ffi::Map<Var, IntSet>> opt_dom_map) {
             if (opt_dom_map.has_value()) {
               return analyzer->int_set(expr, opt_dom_map.value());
             }
             return analyzer->int_set(expr);
           })
      .def_packed("arith.AnalyzerBind",
                  [](ffi::PackedArgs args, ffi::Any* ret) {
                    TVM_FFI_ICHECK(args.size() == 3 || args.size() == 4)
                        << "AnalyzerBind expects 3 or 4 arguments, but got " << args.size();
                    Analyzer analyzer = args[0].cast<Analyzer>();
                    bool allow_override = args.size() >= 4 && args[3].cast<bool>();
                    if (auto opt_range = args[2].try_cast<Range>()) {
                      analyzer->Bind(args[1].cast<Var>(), opt_range.value(), allow_override);
                    } else {
                      analyzer->Bind(args[1].cast<Var>(), args[2].cast<PrimExpr>(), allow_override);
                    }
                  })
      .def("arith.AnalyzerCanProve",
           [](Analyzer analyzer, const PrimExpr& expr, int strength) {
             return analyzer->CanProve(expr, static_cast<ProofStrength>(strength));
           })
      .def("arith.AnalyzerSetMaximumRewriteSteps",
           [](Analyzer analyzer, int64_t maximum) {
             analyzer->rewrite_simplify.SetMaximumRewriteSteps(maximum);
           })
      .def("arith.AnalyzerEnterConstraintContext",
           [](Analyzer analyzer, const PrimExpr& constraint) {
             // can't use make_shared due to noexcept(false) decl in destructor,
             // see https://stackoverflow.com/a/43907314
             auto ctx = std::shared_ptr<With<ConstraintContext>>(
                 new With<ConstraintContext>(analyzer, constraint));
             auto fexit = [ctx](ffi::PackedArgs, ffi::Any*) mutable { ctx.reset(); };
             return ffi::Function::FromPacked(fexit);
           })
      .def_method("arith.AnalyzerCanProveEqual", &AnalyzerObj::CanProveEqual)
      .def("arith.AnalyzerTryCompare",
           [](Analyzer analyzer, const PrimExpr& lhs, const PrimExpr& rhs,
              bool propagate_inequalities) {
             return static_cast<int64_t>(
                 analyzer->transitive_comparisons.TryCompare(lhs, rhs, propagate_inequalities));
           })
      .def("arith.AnalyzerGetSMTLIB2",
           [](Analyzer analyzer, ffi::Optional<PrimExpr> expr) {
             return analyzer->z3_prover.GetSMTLIB2(expr);
           })
      .def("arith.AnalyzerSetZ3TimeoutMs", [](Analyzer analyzer, int64_t timeout_ms) {
        analyzer->z3_prover.SetTimeoutMs(static_cast<unsigned>(timeout_ms));
      })
      .def("arith.AnalyzerSetZ3RLimit", [](Analyzer analyzer, int64_t rlimit) {
        analyzer->z3_prover.SetRLimit(static_cast<unsigned>(rlimit));
      })
      .def("arith.AnalyzerGetZ3Stats",
           [](Analyzer analyzer) { return analyzer->z3_prover.GetStats(); })
      .def("arith.AnalyzerGetEnabledExtensions",
           [](Analyzer analyzer) {
             return static_cast<std::int64_t>(analyzer->rewrite_simplify.GetEnabledExtensions());
           })
      .def("arith.AnalyzerSetEnabledExtensions", [](Analyzer analyzer, int64_t flags) {
        analyzer->rewrite_simplify.SetEnabledExtensions(
            static_cast<RewriteSimplifier::Extension>(flags));
      });
}

}  // namespace arith
}  // namespace tvm
