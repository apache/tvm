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
 * \file tvm/arith/solve_linear_inequality.cc
 * \brief Solve linear inequalities.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/arith/pattern.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "int_operator.h"

namespace tvm {
namespace arith {

using namespace tvm::runtime;
using namespace tvm::tir;

struct ExprLess {
  bool operator()(const PrimExpr& l, const PrimExpr& r) const {
    return CalculateExprComplexity(l) < CalculateExprComplexity(r);
  }
};

void DebugPrint(const std::vector<PrimExpr>& current_ineq_set,
                const std::vector<PrimExpr>& next_ineq_set, const std::vector<PrimExpr>& rest,
                const std::vector<std::pair<int64_t, PrimExpr>>& coef_pos,
                const std::vector<std::pair<int64_t, PrimExpr>>& coef_neg) {
  std::cout << "Current ineq set:\n[";
  for (auto& ineq : current_ineq_set) {
    std::cout << ineq << ", ";
  }
  std::cout << "]\n";

  std::cout << "Next ineq set:\n[";
  for (auto& ineq : next_ineq_set) {
    std::cout << ineq << ", ";
  }
  std::cout << "]\n";

  std::cout << "coef_pos:\n[";
  for (auto& coef : coef_pos) {
    std::cout << "(" << coef.first << ", " << coef.second << "), ";
  }
  std::cout << "]\n";

  std::cout << "coef_neg:\n[";
  for (auto& coef : coef_neg) {
    std::cout << "(" << coef.first << ", " << coef.second << "), ";
  }
  std::cout << "]\n";
}

/*!
 * \brief normalize to the form `expr <= 0`
 */
class NormalizeComparisons : public ExprMutator {
 public:
  PrimExpr VisitExpr_(const EQNode* op) override { return Make<EQ>(op->a, op->b); }
  PrimExpr VisitExpr_(const NENode* op) override { return Make<NE>(op->a, op->b); }
  PrimExpr VisitExpr_(const LTNode* op) override { return Make<LT>(op->a, op->b); }
  PrimExpr VisitExpr_(const LENode* op) override { return Make<LE>(op->a, op->b); }
  PrimExpr VisitExpr_(const GTNode* op) override { return Make<LT>(op->b, op->a); }
  PrimExpr VisitExpr_(const GENode* op) override { return Make<LE>(op->b, op->a); }

 private:
  template <class T>
  PrimExpr Make(const PrimExpr& a, const PrimExpr& b) {
    // rewrite LT to LE for ints
    if (std::is_same<T, LT>::value && (a.dtype().is_int() || a.dtype().is_uint())) {
      return LE(analyzer_.Simplify(a - b + 1), make_zero(a.dtype()));
    }
    return T(analyzer_.Simplify(a - b), make_zero(a.dtype()));
  }
  arith::Analyzer analyzer_;
};

void AddInequality(std::vector<PrimExpr>* inequality_set, const PrimExpr& new_ineq,
                   Analyzer* analyzer) {
  if (analyzer->CanProve(new_ineq) ||
      std::find_if(inequality_set->begin(), inequality_set->end(), [&](const PrimExpr& e) {
        return StructuralEqual()(e, new_ineq);
      }) != inequality_set->end()) {
    // redundant: follows from the vranges
    // or has already been added
    return;
  }
  if (const LENode* new_le = new_ineq.as<LENode>()) {
    for (auto iter = inequality_set->begin(); iter != inequality_set->end();) {
      const LENode* le = iter->as<LENode>();
      if (le && analyzer->CanProve(new_le->a - le->a <= 0)) {
        return;
      } else if (le && analyzer->CanProve(le->a - new_le->a <= 0)) {
        iter = inequality_set->erase(iter);
      } else {
        ++iter;
      }
    }
  }

  inequality_set->push_back(new_ineq);
}

void ClassifyByPolarity(const Var& var, const std::vector<PrimExpr>& current_ineq_set,
                        std::vector<PrimExpr>* next_ineq_set, std::vector<PrimExpr>* rest,
                        std::vector<std::pair<int64_t, PrimExpr>>* coef_pos,
                        std::vector<std::pair<int64_t, PrimExpr>>* coef_neg, Analyzer* analyzer) {
  // Take formulas from current_ineq_set and classify them according to polarity wrt var
  // and store to coef_pos and coef_neg respectively.
  for (const PrimExpr& ineq : current_ineq_set) {
    if (const LENode* le = ineq.as<LENode>()) {
      Array<PrimExpr> coef = arith::DetectLinearEquation(le->a, {var});
      if (!coef.empty() && is_const_int(coef[0])) {
        int64_t coef0 = *as_const_int(coef[0]);
        if (coef0 == 0) {
          // zero polarity, straight to next_ineq_set
          AddInequality(next_ineq_set, ineq, analyzer);
        } else if (coef0 > 0) {
          coef_pos->push_back({coef0, coef[1]});
        } else if (coef0 < 0) {
          coef_neg->push_back({coef0, coef[1]});
        }
        continue;
      }
    } else if (const EQNode* eq = ineq.as<EQNode>()) {
      Array<PrimExpr> coef = arith::DetectLinearEquation(eq->a, {var});
      if (!coef.empty() && is_const_int(coef[0])) {
        int64_t coef0 = *as_const_int(coef[0]);
        if (coef0 == 0) {
          // zero polarity, straight to next_ineq_set
          AddInequality(next_ineq_set, ineq, analyzer);
        } else if (coef0 > 0) {
          // Equalities may be considered as pairs of two inequalities
          coef_pos->push_back({coef0, coef[1]});
          coef_neg->push_back({-coef0, -coef[1]});
        } else if (coef0 < 0) {
          coef_pos->push_back({-coef0, -coef[1]});
          coef_neg->push_back({coef0, coef[1]});
        }
        continue;
      }
    }

    // if nothing worked, put it in rest
    rest->push_back(ineq);
  }
}

void MoveEquality(std::vector<PrimExpr>* upper_bounds, std::vector<PrimExpr>* lower_bounds,
                  std::vector<PrimExpr>* equalities) {
  // those exist in both upper & lower bounds will be moved to equalities
  for (auto ub = upper_bounds->begin(); ub != upper_bounds->end();) {
    auto lb = std::find_if(lower_bounds->begin(), lower_bounds->end(),
                           [&](const PrimExpr& e) { return StructuralEqual()(e, *ub); });
    if (lb != lower_bounds->end()) {
      equalities->push_back(*lb);
      lower_bounds->erase(lb);
      ub = upper_bounds->erase(ub);
    } else {
      ++ub;
    }
  }
}

PartialSolvedInequalities SolveLinearInequalities(const IntConstraints& system_to_solve) {
  arith::Analyzer analyzer;
  analyzer.Bind(system_to_solve->ranges);

  // The algorithm consists in doing the following things for each variable v
  // - Take formulas from `current_ineq_set_to_solve` and
  //   classify them according to polarity wrt v.
  // - Combine each formula of positive polarity (wrt v)
  //   with each formula of negative polarity.
  // - Put the resulting combinations into `next_ineq_set_to_solve`
  //   along with unclassifiable formulas.
  // - Replace `current_ineq_set_to_solve` with `next_ineq_set_to_solve`
  //   and move to the next variable.

  // normalized inequality
  std::vector<PrimExpr> current_ineq_set_to_solve;
  std::vector<PrimExpr> next_ineq_set_to_solve;
  // A vector of pairs (c, e), c > 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_pos;
  // A vector of pairs (c, e), c < 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_neg;

  // formulas we don't know what to do with
  std::vector<PrimExpr> rest;

  // Simplify each inequality into the form `expr <= 0` and add to current formulas
  for (const PrimExpr& ineq : system_to_solve->relations) {
    AddInequality(&current_ineq_set_to_solve,
                  NormalizeComparisons()(analyzer.Simplify(ineq, kSimplifyRewriteCanonicalRewrite)),
                  &analyzer);
  }

  Map<Var, IntGroupBounds> res_bounds;
  for (const Var& v : system_to_solve->variables) {
    ICHECK(!res_bounds.count(v))
        << "Variable " << v
        << " appears more than one time in the `variables` which might be a bug";

    next_ineq_set_to_solve.clear();
    coef_pos.clear();
    coef_neg.clear();

    // Add bounds from vranges
    if (system_to_solve->ranges.count(v)) {
      const Range& range = system_to_solve->ranges[v];
      PrimExpr range_lbound = analyzer.Simplify(range->min, kSimplifyRewriteCanonicalRewrite);
      PrimExpr range_ubound =
          analyzer.Simplify(range->min + range->extent - 1, kSimplifyRewriteCanonicalRewrite);
      coef_neg.push_back({-1, range_lbound});
      coef_pos.push_back({1, -range_ubound});
    }

    ClassifyByPolarity(v, current_ineq_set_to_solve, &next_ineq_set_to_solve, &rest, &coef_pos,
                       &coef_neg, &analyzer);

    // Combine each positive inequality with each negative one (by adding them together)
    int64_t gcd_x, gcd_y;
    for (const auto& pos : coef_pos) {
      for (const auto& neg : coef_neg) {
        auto first_gcd = ExtendedEuclidean(pos.first, -neg.first, &gcd_x, &gcd_y);
        PrimExpr c_pos = make_const(v.dtype(), neg.first / first_gcd);
        PrimExpr c_neg = make_const(v.dtype(), pos.first / first_gcd);
        // eliminate the current variable
        PrimExpr new_lhs = c_neg * neg.second - c_pos * pos.second;
        PrimExpr new_ineq = LE(new_lhs, make_zero(pos.second.dtype()));
        // we need rewrite_simplify -> canonical_simplify -> rewrite_simplify
        // to help simplify things like (((y + 10) - (-1*(y - 20))) <= 0) => y - 5 <= 0
        // with steps = 2 it's (y*2) - 10 <= 0
        new_ineq =
            NormalizeComparisons()(analyzer.Simplify(new_ineq, kSimplifyRewriteCanonicalRewrite));
        AddInequality(&next_ineq_set_to_solve, new_ineq, &analyzer);
      }
    }

    // Now we have to generate resulting (in)equalities for the variable v

    // Find the common denominator in a sense
    // We will generate formulas of the form coef_lcm*v <= bound
    int64_t coef_lcm = 1;
    for (const auto& pos : coef_pos) {
      coef_lcm = LeastCommonMultiple(coef_lcm, pos.first);
    }
    for (const auto& neg : coef_neg) {
      coef_lcm = LeastCommonMultiple(coef_lcm, -neg.first);
    }

    // The resulting lower and upper bounds
    std::vector<PrimExpr> upper_bounds;
    std::vector<PrimExpr> lower_bounds;
    upper_bounds.reserve(coef_pos.size());
    lower_bounds.reserve(coef_neg.size());

    for (const auto& pos : coef_pos) {
      PrimExpr bound = make_const(v.dtype(), -coef_lcm / pos.first) * pos.second;
      bound = analyzer.Simplify(bound, kSimplifyRewriteCanonicalRewrite);
      // Don't add if any of the existing bounds is better
      if (std::any_of(upper_bounds.begin(), upper_bounds.end(),
                      [&bound, &analyzer](const PrimExpr& o) {
                        return analyzer.CanProve(o - bound <= 0);
                      })) {
        continue;
      }
      // Erase all worse bounds
      for (auto iter = upper_bounds.begin(); iter != upper_bounds.end();) {
        if (analyzer.CanProve(*iter - bound >= 0)) {
          iter = upper_bounds.erase(iter);
        } else {
          ++iter;
        }
      }
      // Add the upper bound
      upper_bounds.push_back(bound);
    }
    for (const auto& neg : coef_neg) {
      PrimExpr bound = make_const(v.dtype(), -coef_lcm / neg.first) * neg.second;
      bound = analyzer.Simplify(bound, kSimplifyRewriteCanonicalRewrite);
      // Don't add if any of the existing bounds is better
      if (std::any_of(lower_bounds.begin(), lower_bounds.end(),
                      [&bound, &analyzer](const PrimExpr& o) {
                        return analyzer.CanProve(o - bound >= 0);
                      })) {
        continue;
      }
      // Erase all worse bounds
      for (auto iter = lower_bounds.begin(); iter != lower_bounds.end();) {
        if (analyzer.CanProve(*iter - bound <= 0)) {
          iter = lower_bounds.erase(iter);
        } else {
          ++iter;
        }
      }
      // Add the lower bound
      lower_bounds.push_back(bound);
    }

    std::vector<PrimExpr> equal;
    equal.reserve(std::min(upper_bounds.size(), lower_bounds.size()));
    MoveEquality(&upper_bounds, &lower_bounds, &equal);
    std::vector<PrimExpr> equal_list(equal.begin(), equal.end());
    std::sort(equal_list.begin(), equal_list.end(), ExprLess());

    // Write it to the result.
    IntGroupBounds bnds(make_const(v.dtype(), coef_lcm),
                        Array<PrimExpr>(lower_bounds.begin(), lower_bounds.end()),
                        Array<PrimExpr>(equal_list.begin(), equal_list.end()),
                        Array<PrimExpr>(upper_bounds.begin(), upper_bounds.end()));
    res_bounds.Set(v, bnds);

    std::swap(current_ineq_set_to_solve, next_ineq_set_to_solve);
  }

  // Everything that is left goes to res.relations
  Array<PrimExpr> other_conditions;
  for (const PrimExpr& e : current_ineq_set_to_solve) {
    PrimExpr e_simp = analyzer.Simplify(e, kSimplifyRewriteCanonicalRewrite);
    if (is_const_int(e_simp, 0)) {
      // contradiction detected
      other_conditions = {const_false()};
      break;
    } else if (is_const_int(e_simp, 1)) {
      continue;
    } else {
      other_conditions.push_back(e_simp);
    }
  }

  for (const PrimExpr& e : rest) {
    other_conditions.push_back(e);
  }

  return {res_bounds, other_conditions};
}

#ifdef _MSC_VER
#pragma optimize("g", off)
#endif
IntConstraints SolveInequalitiesToRange(const IntConstraints& inequalities) {
  // Resulting ranges will contain ranges for the new variables and for the variables that are
  // not in the inequalities->variables but are in inequalities->ranges
  // It will be useful when solving Jacobian axes jac_xxx)
  Map<Var, Range> res_ranges;
  // we get a set of equality, lower, upper bound of each variable.
  auto solved_system = SolveLinearInequalities(inequalities);

  Map<Var, IntGroupBounds> solved_bounds = solved_system.first;
  Array<PrimExpr> solved_other_relations = solved_system.second;

  Array<PrimExpr> res_relations;

  // this keeps being updated during determining the range of each variable.
  Map<Var, Range> vranges;
  for (std::pair<Var, Range> vr : inequalities->ranges) {
    vranges.Set(vr.first, vr.second);
  }

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = inequalities->variables.rbegin(); it != inequalities->variables.rend(); ++it) {
    arith::Analyzer analyzer;
    analyzer.Bind(vranges);

    const Var& var = *it;
    ICHECK(solved_bounds.count(var));
    auto bnd = solved_bounds[var];
    if (is_one(bnd->coef) && !bnd->equal.empty()) {
      // There is an equation of the form `v == expr`, so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity,
      // so it must be the simplest one.
      // The MSVC compiler optimization must be disabled for the expression `bnd->equal[0]` which
      // triggers an internal compiler error.
      Range best_range(bnd->equal[0],
                       analyzer.Simplify(bnd->equal[0] + 1, kSimplifyRewriteCanonicalRewrite));
      res_ranges.Set(var, best_range);
      vranges.Set(var, best_range);
    } else {
      if (vranges.count(var) > 0) {
        bnd = bnd + vranges[var];
      }

      auto best_range = bnd.FindBestRange(vranges);

      if (best_range.defined()) {
        if (analyzer.CanProveGreaterEqual(-best_range->extent, 0)) {
          // range.extent <= 0 implies the input inequality system is unsolvable
          return IntConstraints(/*variables=*/{}, /*ranges=*/{},
                                /*relations=*/{tir::make_zero(DataType::Bool())});
        }
        res_ranges.Set(var, best_range);
        vranges.Set(var, best_range);
      }
    }
  }

  // Add the original conditions to the resulting conditions
  arith::Analyzer analyzer;
  analyzer.Bind(vranges);
  for (const PrimExpr& old_cond :
       AsConditions(inequalities->variables, solved_bounds, solved_other_relations)) {
    if (!analyzer.CanProve(old_cond)) {
      // those not represented in vranges (res_ranges)
      res_relations.push_back(old_cond);
    }
  }

  IntConstraints system(inequalities->variables, res_ranges, res_relations);

  return system;
}
#ifdef _MSC_VER
#pragma optimize("g", on)
#endif

IntConstraintsTransform SolveInequalitiesDeskewRange(const IntConstraints& inequalities) {
  // Resulting ranges will contain ranges for the new variables and for the variables that are
  // not in the inequalities->variables but are in inequalities->ranges (jac_xxx)
  Map<Var, Range> res_ranges;
  // we get a set of equality, lower, upper bound of each variable.
  auto solved_system = SolveLinearInequalities(inequalities);
  Map<Var, IntGroupBounds> solved_bounds = solved_system.first;
  Array<PrimExpr> solved_other_relations = solved_system.second;

  arith::Analyzer analyzer;

  Map<Var, PrimExpr> res_src_to_dst;
  Map<Var, PrimExpr> res_dst_to_src;
  Array<Var> res_variables;
  Array<PrimExpr> res_relations;

  // this keeps being updated during determining the range of each variable.
  Map<Var, Range> vranges;
  for (std::pair<Var, Range> vr : inequalities->ranges) {
    vranges.Set(vr.first, vr.second);
  }
  analyzer.Bind(vranges);

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = inequalities->variables.rbegin(); it != inequalities->variables.rend(); ++it) {
    const Var& var = *it;
    auto bnd = solved_bounds[var];
    // Note that we replace old vars with new ones
    bnd = bnd.Substitute(res_src_to_dst);

    if (is_one(bnd->coef) && !bnd->equal.empty()) {
      // There is an equation of the form `v == expr`,
      // so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity,
      // so it must be the simplest one.
      res_src_to_dst.Set(var, bnd->equal[0]);
    } else {
      if (vranges.count(var) > 0) {
        bnd = bnd + vranges[var];
      }

      auto best_range = bnd.FindBestRange(vranges);

      Var new_var = var.copy_with_suffix(".shifted");
      if (!best_range.defined()) {
        res_src_to_dst.Set(var, var);
        res_dst_to_src.Set(var, var);
        res_variables.push_back(var);
      } else if (is_const_int(best_range->extent, 1)) {
        // Don't create an itervar, just replace it everywhere with its min
        res_src_to_dst.Set(var, best_range->min);
      } else if (analyzer.CanProveGreaterEqual(-best_range->extent, 0)) {
        // range.extent <= 0 implies the input inequality system is unsolvable
        return IntConstraintsTransform(inequalities,
                                       IntConstraints(
                                           /*variables=*/{},
                                           /*ranges=*/{},
                                           /*relations=*/{tir::make_zero(DataType::Bool())}),
                                       {}, {});
      } else {
        // created new_var starts from 0
        res_src_to_dst.Set(var, new_var + best_range->min);
        // Note that we are substituting old with new, so best_range contains new var,
        // that is we have to substitute new with old in best_range here
        res_dst_to_src.Set(new_var,
                           analyzer.Simplify(var - Substitute(best_range->min, res_dst_to_src)));

        // Add the new var to the resulting axis
        auto range = Range(make_zero(new_var.dtype()), best_range->extent);
        res_variables.push_back(new_var);
        res_ranges.Set(new_var, range);

        vranges.Set(new_var, range);
        analyzer.Bind(new_var, range);
      }
    }
  }

  // Add the original conditions (with variables substituted) to the resulting conditions
  for (const PrimExpr& old_cond :
       AsConditions(inequalities->variables, solved_bounds, solved_other_relations)) {
    PrimExpr new_cond = analyzer.Simplify(Substitute(old_cond, res_src_to_dst));
    if (!is_const_int(new_cond, 1)) {
      // those not represented in vranges (res_ranges)
      res_relations.push_back(new_cond);
    }
  }

  // Reverse the axis so that it matches the order of the original variables
  res_variables = Array<Var>(res_variables.rbegin(), res_variables.rend());

  IntConstraints new_inequalities(res_variables, res_ranges, res_relations);
  IntConstraintsTransform transform(inequalities, new_inequalities, res_src_to_dst, res_dst_to_src);

  return transform;
}

TVM_REGISTER_GLOBAL("arith.SolveInequalitiesAsCondition")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      IntConstraints problem;
      PartialSolvedInequalities ret_ineq;
      if (args.size() == 1) {
        problem = args[0];
        ret_ineq = SolveLinearInequalities(problem);
      } else if (args.size() == 3) {
        problem = IntConstraints(args[0], args[1], args[2]);
        ret_ineq = SolveLinearInequalities(problem);
      } else {
        LOG(FATAL) << "arith.SolveInequalitiesAsCondition expects 1 or 3 arguments, gets "
                   << args.size();
      }
      *ret = AsConditions(problem->variables, ret_ineq.first, ret_ineq.second);
    });

TVM_REGISTER_GLOBAL("arith.SolveInequalitiesToRange").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args.size() == 1) {
    *ret = SolveInequalitiesToRange(args[0]);
  } else if (args.size() == 3) {
    IntConstraints problem(args[0], args[1], args[2]);
    *ret = SolveInequalitiesToRange(problem);
  } else {
    LOG(FATAL) << "arith.SolveInequalitiesToRange expects 1 or 3 arguments, gets " << args.size();
  }
});

TVM_REGISTER_GLOBAL("arith.SolveInequalitiesDeskewRange")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      if (args.size() == 1) {
        *ret = SolveInequalitiesDeskewRange(args[0]);
      } else if (args.size() == 3) {
        IntConstraints problem(args[0], args[1], args[2]);
        *ret = SolveInequalitiesDeskewRange(problem);
      } else {
        LOG(FATAL) << "arith.SolveInequalitiesDeskewRange expects 1 or 3 arguments, gets "
                   << args.size();
      }
    });

}  // namespace arith
}  // namespace tvm
