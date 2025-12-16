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
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/builtin.h>

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
      int_set(this),
      z3_prover(this) {}

std::unique_ptr<Analyzer> Analyzer::Clone() const {
  auto cloned = std::make_unique<Analyzer>();
  // Copy per-sub-analyzer states
  cloned->const_int_bound.CopyFrom(this->const_int_bound);
  cloned->modular_set.CopyFrom(this->modular_set);
  cloned->rewrite_simplify.CopyFrom(this->rewrite_simplify);
  cloned->canonical_simplify.CopyFrom(this->canonical_simplify);
  cloned->int_set.CopyFrom(this->int_set);
  cloned->transitive_comparisons.CopyFrom(this->transitive_comparisons);
  cloned->z3_prover.CopyFrom(this->z3_prover);
  return cloned;
}

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
  this->z3_prover.Bind(var, expr, allow_override);
}

void Analyzer::Bind(const Var& var, const Range& range, bool allow_override) {
  ICHECK(range.defined());
  if (tir::is_one(range->extent)) {
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

void Analyzer::Bind(const ffi::Map<Var, Range>& variables, bool allow_override) {
  for (const auto& iter : variables) {
    this->Bind(iter.first, iter.second, allow_override);
  }
}

void ConstraintContext::EnterWithScope() {
  ICHECK(recovery_functions_.size() == 0);
  // entering the scope.
  recovery_functions_.push_back(analyzer_->const_int_bound.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->modular_set.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->rewrite_simplify.EnterConstraint(constraint_, is_assume_));
  recovery_functions_.push_back(analyzer_->int_set.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->transitive_comparisons.EnterConstraint(constraint_));
  recovery_functions_.push_back(analyzer_->z3_prover.EnterConstraint(constraint_));
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
  if (as_int && *as_int) { return true; }

  // Structured boolean reasoning for Or/And (and their bitwise counterparts on bool)
  // Evaluate children with the same proof strength.
  if (const auto* not_node = simplified.as<tir::NotNode>()) {
    PrimExpr a = not_node->a;
    // Try direct complements on common comparators
    if (const auto* p = a.as<tir::LTNode>()) {
      return CanProve(tir::GE(p->a, p->b), strength);
    }
    if (const auto* p = a.as<tir::LENode>()) {
      return CanProve(tir::GT(p->a, p->b), strength);
    }
    if (const auto* p = a.as<tir::GTNode>()) {
      return CanProve(tir::LE(p->a, p->b), strength);
    }
    if (const auto* p = a.as<tir::GENode>()) {
      return CanProve(tir::LT(p->a, p->b), strength);
    }
    if (const auto* p = a.as<tir::EQNode>()) {
      return CanProve(tir::NE(p->a, p->b), strength);
    }
    if (const auto* p = a.as<tir::NENode>()) {
      return CanProve(tir::EQ(p->a, p->b), strength);
    }
    // De Morgan on canonical boolean nodes
    if (const auto* or_node = a.as<tir::OrNode>()) {
      PrimExpr lhs = tir::Not(or_node->a);
      PrimExpr rhs = tir::Not(or_node->b);
      return CanProve(tir::And(lhs, rhs), strength);
    }
    if (const auto* and_node = a.as<tir::AndNode>()) {
      PrimExpr lhs = tir::Not(and_node->a);
      PrimExpr rhs = tir::Not(and_node->b);
      return CanProve(tir::Or(lhs, rhs), strength);
    }
    // De Morgan on bitwise boolean calls
    if (const auto* c = a.as<tir::CallNode>()) {
      using namespace tir;
      if (c->op.same_as(builtin::bitwise_or()) && c->args.size() == 2 && a.dtype().is_bool()) {
        PrimExpr lhs = tir::Not(c->args[0]);
        PrimExpr rhs = tir::Not(c->args[1]);
        return CanProve(tir::And(lhs, rhs), strength);
      }
      if (c->op.same_as(builtin::bitwise_and()) && c->args.size() == 2 && a.dtype().is_bool()) {
        PrimExpr lhs = tir::Not(c->args[0]);
        PrimExpr rhs = tir::Not(c->args[1]);
        return CanProve(tir::Or(lhs, rhs), strength);
      }
    }
    if (const auto* inner_not = a.as<tir::NotNode>()) {
      // Double negation
      return CanProve(inner_not->a, strength);
    }
    // Fallback: if `a` simplifies to constant false, then Not(a) is true
    PrimExpr a_simpl = Simplify(a);
    const int64_t* a_const = tir::as_const_int(a_simpl);
    if (a_const && *a_const == 0) { return true; }
    // Otherwise, cannot conclude true
  }
  if (const auto* or_node = simplified.as<tir::OrNode>()) {
    if (CanProve(or_node->a, strength)) {
      return true;
    }
    if (CanProve(or_node->b, strength)) {
      return true;
    }
  }
  if (const auto* and_node = simplified.as<tir::AndNode>()) {
    bool lhs = CanProve(and_node->a, strength);
    bool rhs = CanProve(and_node->b, strength);
    if (lhs && rhs) {
      return true;
    }
  }
  if (const auto* call = simplified.as<tir::CallNode>()) {
    using namespace tir;
    if (call->op.same_as(builtin::bitwise_or()) && call->args.size() == 2 &&
        simplified.dtype().is_bool()) {
      if (CanProve(call->args[0], strength) || CanProve(call->args[1], strength)) {
        return true;
      }
    }
    if (call->op.same_as(builtin::bitwise_and()) && call->args.size() == 2 &&
        simplified.dtype().is_bool()) {
      bool lhs = CanProve(call->args[0], strength);
      bool rhs = CanProve(call->args[1], strength);
      if (lhs && rhs) {
        return true;
      }
    }
    if (call->op.same_as(builtin::bitwise_not()) && call->args.size() == 1 &&
        simplified.dtype().is_bool()) {
      // Treat as logical not and reuse Not handling by constructing tir::Not
      return CanProve(tir::Not(call->args[0]), strength);
    }
  }
  if (strength >= ProofStrength::kSymbolicBound) {
    // NOTE: we intentionally only pattern match common bound predicate i < bound
    // and put this implementation at the top-level.
    // This is to avoid repeatitive calling of this function
    // that causes speed issues.
    // This strategy can only be called from top-level and not from sub-analyzers.
    ffi::Optional<PrimExpr> pos_diff;
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
      PrimExpr simplified_diff = this->Simplify(pos_diff.value());
      IntSet iset = this->int_set(simplified_diff);
      if (iset.HasLowerBound()) {
        ConstIntBound relaxed_lower_bound = this->const_int_bound(this->Simplify(iset.min()));
        if (relaxed_lower_bound->min_value >= lower_bound) return true;
      }
      if (iset.HasUpperBound()) {
        ConstIntBound relaxed_upper_bound = this->const_int_bound(this->Simplify(iset.max()));
        if (relaxed_upper_bound->max_value < lower_bound) return false;
      }
    }
  }

  // Current analysis may not be powerful enough to prove expressions containing
  // the same symbolic value multiple times. However, when the symbolic values are
  // "T.vscale" and the compile target uses a scalable architecture extension like
  // VLA, we can make some assumptions about the value of vscale and iterate over a
  // space of pre-defined values to attempt to prove the expression.
  Target curr_target = Target::Current();
  if (ContainsVscaleCall(simplified)) {
    if (TargetHasVLA(curr_target)) {
      auto kVScaleValues = GetVScaleValues(curr_target);
      if(CanProveVscaleExpressionFromKnownValues(this, simplified, kVScaleValues)) {
        return true;
      }
    }
    // LOG(WARNING)
    //     << "The expression contains scalable values. An attempt to prove by substituting "
    //        "with known values of vscale was not performed. This proof currently only supports "
    //        "VLA targets, but the target was "
    //     << curr_target;
  }
  if(z3_prover.CanProve(simplified)) {
    // auto msg = z3_prover.GetSMTLIB2(simplified);
    // std::stringstream ss;
    // ss << msg;
    // std::stringstream out;
    // std::string tmp;
    // while(std::getline(ss, tmp)) {
    //   out << "    " << tmp << "\n";
    // }
    // LOG(INFO) << "Proved by Z3: " << simplified << "\n" << out.str();
    return true;
  }
  // if(strength >= ProofStrength::kSymbolicBound && z3_prover.CanProve(simplified)) {
  //   // The following debug logging is very useful when diagnosing issues with the Z3 prover.
  //   auto msg = z3_prover.GetSMTLIB2(simplified);
  //   std::stringstream ss;
  //   ss << msg;
  //   std::stringstream out;
  //   std::string tmp;
  //   while(std::getline(ss, tmp)) {
  //     out << "    " << tmp << "\n";
  //   }
  //   LOG(INFO) << "Proved by Z3: " << simplified << "\n" << out.str();
  //   return true;
  // }
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

namespace {
using FnFactory = tvm::ffi::TypedFunction<tvm::ffi::Function(std::string)>;
static FnFactory BuildAnalyzerFactory(std::shared_ptr<tvm::arith::Analyzer> self) {
  using tvm::ffi::Function;
  return FnFactory([self](std::string name) -> Function {
    if (name == "const_int_bound") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->const_int_bound(args[0].cast<PrimExpr>());
      });
    } else if (name == "modular_set") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->modular_set(args[0].cast<PrimExpr>());
      });
    } else if (name == "clone") {
      return Function([self](tvm::ffi::PackedArgs, tvm::ffi::Any* ret) {
        auto cloned_unique = self->Clone();
        auto cloned = std::shared_ptr<Analyzer>(cloned_unique.release());
        *ret = BuildAnalyzerFactory(cloned);
      });
    } else if (name == "const_int_bound_update") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        self->const_int_bound.Update(args[0].cast<Var>(), args[1].cast<ConstIntBound>(),
                                     args[2].cast<bool>());
      });
    } else if (name == "const_int_bound_is_bound") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->const_int_bound.IsBound(args[0].cast<Var>());
      });
    } else if (name == "Simplify") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        if (args.size() == 1) {
          *ret = self->Simplify(args[0].cast<PrimExpr>());
        } else if (args.size() == 2) {
          *ret = self->Simplify(args[0].cast<PrimExpr>(), args[1].cast<int>());
        } else {
          LOG(FATAL) << "Invalid size of argument (" << args.size() << ")";
        }
      });
    } else if (name == "rewrite_simplify") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->rewrite_simplify(args[0].cast<PrimExpr>());
      });
    } else if (name == "get_rewrite_simplify_stats") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->rewrite_simplify.GetStatsCounters();
      });
    } else if (name == "reset_rewrite_simplify_stats") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        self->rewrite_simplify.ResetStatsCounters();
      });
    } else if (name == "canonical_simplify") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->canonical_simplify(args[0].cast<PrimExpr>());
      });
    } else if (name == "int_set") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->int_set(args[0].cast<PrimExpr>(), args[1].cast<tvm::ffi::Map<Var, IntSet>>());
      });
    } else if (name == "bind") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        if (auto opt_range = args[1].try_cast<Range>()) {
          self->Bind(args[0].cast<Var>(), opt_range.value());
        } else {
          self->Bind(args[0].cast<Var>(), args[1].cast<PrimExpr>());
        }
      });
    } else if (name == "can_prove") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        int strength = args[1].cast<int>();
        *ret = self->CanProve(args[0].cast<PrimExpr>(), static_cast<ProofStrength>(strength));
      });
    } else if (name == "enter_constraint_context") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        auto ctx = std::shared_ptr<With<ConstraintContext>>(
            new With<ConstraintContext>(self.get(), args[0].cast<PrimExpr>()));
        auto fexit = [ctx](tvm::ffi::PackedArgs, tvm::ffi::Any*) mutable { ctx.reset(); };
        *ret = tvm::ffi::Function::FromPacked(fexit);
      });
    } else if (name == "can_prove_equal") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->CanProveEqual(args[0].cast<PrimExpr>(), args[1].cast<PrimExpr>());
      });
    } else if (name == "get_enabled_extensions") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = static_cast<std::int64_t>(self->rewrite_simplify.GetEnabledExtensions());
      });
    } else if (name == "set_enabled_extensions") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        int64_t flags = args[0].cast<int64_t>();
        self->rewrite_simplify.SetEnabledExtensions(
            static_cast<RewriteSimplifier::Extension>(flags));
      });
    } else if (name == "get_smtlib2") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        auto expr = args[0].cast<ffi::Optional<PrimExpr>>();
        *ret = self->z3_prover.GetSMTLIB2(expr);
      });
    } else if (name == "get_z3_stats") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        *ret = self->z3_prover.GetStats();
      });
    } else if (name == "set_z3_timeout_ms") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        unsigned timeout_ms = args[0].cast<unsigned>();
        self->z3_prover.SetTimeoutMs(timeout_ms);
      });
    } else if (name == "set_z3_max_step") {
      return Function([self](tvm::ffi::PackedArgs args, tvm::ffi::Any* ret) {
        unsigned max_step = args[0].cast<unsigned>();
        self->z3_prover.SetMaxStep(max_step);
      });
    }
    return Function();
  });
}
}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("arith.CreateAnalyzer", [](ffi::PackedArgs, ffi::Any* ret) {
    auto self = std::make_shared<Analyzer>();
    *ret = BuildAnalyzerFactory(self);
  });
}

}  // namespace arith
}  // namespace tvm
