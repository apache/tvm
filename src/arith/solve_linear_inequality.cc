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
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/analysis.h>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/arith/util.h>
#include <tvm/tir/op.h>
#include <tvm/arith/pattern.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/runtime/data_type.h>

// TODO
#include "../tir/pass/ir_util.h"
// TODO: testing
// https://github.com/sergei-grechanik/tvm/blob/8cad2d1e62272b3e192bfe08b896e07bc9550e94/tests/python/unittest/test_pass_zero_elimination.py#L367

namespace tvm {
namespace arith {

using namespace tvm::runtime;
using namespace tvm::te;

/*
struct ExprLess {
  bool operator()(const PrimExpr& l, const PrimExpr& r) const {
    // FIXME:
    // After https://github.com/apache/incubator-tvm/pull/5206
    // we no longer have ExprLess,
    // it was comparing VarNode* raw pointers
    return Compare(l, r) < 0;
  }
};
*/

void DebugPrint(std::unordered_set<PrimExpr, StructuralHash, StructuralEqual>& current_ineq_set,
                std::unordered_set<PrimExpr, StructuralHash, StructuralEqual>& next_ineq_set,
                std::vector<PrimExpr>& rest,
                std::vector<std::pair<int64_t, PrimExpr> >& coef_pos,
                std::vector<std::pair<int64_t, PrimExpr> >& coef_neg) {
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

// normalize to the form `expr <= 0`
class NormalizeComparisons : public ExprMutator {
 public:
  PrimExpr VisitExpr_(const EQNode* op) override { return Make<EQNode>(op->a, op->b); }
  PrimExpr VisitExpr_(const NENode* op) override { return Make<NENode>(op->a, op->b); }
  PrimExpr VisitExpr_(const LTNode* op) override { return Make<LTNode>(op->a, op->b); }
  PrimExpr VisitExpr_(const LENode* op) override { return Make<LENode>(op->a, op->b); }
  PrimExpr VisitExpr_(const GTNode* op) override { return Make<LTNode>(op->b, op->a); }
  PrimExpr VisitExpr_(const GENode* op) override { return Make<LENode>(op->b, op->a); }

 private:
  template <class TNode>
  PrimExpr Make(const PrimExpr& a, const PrimExpr& b) {
    LOG(INFO) << "a = " << a << " b = " << b;
    // rewrite LT to LE for ints
    if (std::is_same<TNode, LTNode>::value && (a.dtype().is_int() || a.dtype().is_uint())) {
      return LENode::make(analyzer_.Simplify(a - b + 1), make_zero(a.dtype()));
    }
    return TNode::make(analyzer_.Simplify(a - b), make_zero(a.dtype()));
  }
  arith::Analyzer analyzer_;
};

void AddInequality(std::unordered_set<PrimExpr, StructuralHash, StructuralEqual>& inequality_set,
                   const PrimExpr& new_ineq,
                   Analyzer& analyzer) {
  LOG(INFO) << "insert ineq " << new_ineq;
  if (analyzer.CanProve(new_ineq) || inequality_set.find(new_ineq) != inequality_set.end()) {
    // redundant: follows from the vranges
    // or has already been added
    return;
  }
  for (auto iter = inequality_set.begin(); iter != inequality_set.end();) {
    if (const LENode* new_le = new_ineq.as<LENode>()) {
      const LENode* le = iter->as<LENode>();
      if (le && analyzer.CanProve(new_le->a - le->a <= 0)) {
        return;
      } else if (le && analyzer.CanProve(le->a - new_le->a <= 0)) {
        iter = inequality_set.erase(iter);
      } else {
        ++iter;
      }
    } else {
      ++iter;
    }
  }

  inequality_set.insert(new_ineq);
}

void ClassifyByPolarity(const Var &var,
                        std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> &current_ineq_set,
                        std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> &next_ineq_set,
                        std::vector<PrimExpr> &rest,
                        std::vector<std::pair<int64_t, PrimExpr> > &coef_pos,
                        std::vector<std::pair<int64_t, PrimExpr> > &coef_neg,
                        Analyzer &analyzer) {
  // Take formulas from current_ineq_set and classify them according to polarity wrt var
  // and store to coef_pos and coef_neg respectively.
  for (const PrimExpr& ineq : current_ineq_set) {
    if (const LENode* le = ineq.as<LENode>()) {
      Array<PrimExpr> coef = arith::DetectLinearEquation(le->a, {var});
      if (!coef.empty() && is_const(coef[0])) {
        int64_t coef0 = *as_const_int(coef[0]);
        if (coef0 == 0) {
          // zero polarity, straight to next_ineq_set
          AddInequality(next_ineq_set, ineq, analyzer);
        } else if (coef0 > 0) {
          coef_pos.push_back({coef0, coef[1]});
        } else if (coef0 < 0) {
          coef_neg.push_back({coef0, coef[1]});
        }
        continue;
      }
    } else if (const EQNode* eq = ineq.as<EQNode>()) {
      Array<PrimExpr> coef = arith::DetectLinearEquation(eq->a, {var});
      if (!coef.empty() && is_const(coef[0])) {
        int64_t coef0 = *as_const_int(coef[0]);
        if (coef0 == 0) {
          // zero polarity, straight to new_current
          AddInequality(next_ineq_set, ineq, analyzer);
        } else if (coef0 > 0) {
          // Equalities may be considered as pairs of two inequalities
          coef_pos.push_back({coef0, coef[1]});
          coef_neg.push_back({-coef0, -coef[1]});
        } else if (coef0 < 0) {
          coef_pos.push_back({-coef0, -coef[1]});
          coef_neg.push_back({coef0, coef[1]});
        }
        continue;
      }
    }

    // if nothing worked, put it in rest
    rest.push_back(ineq);
  }
}

void MoveEquality(std::unordered_set<PrimExpr, StructuralHash, StructuralEqual>& upper_bounds,
                  std::unordered_set<PrimExpr, StructuralHash, StructuralEqual>& lower_bounds,
                  std::unordered_set<PrimExpr, StructuralHash, StructuralEqual>& equalities) {
  // those exist in both upper & lower bounds will be moved to equalities
  for (auto ub = upper_bounds.begin(); ub != upper_bounds.end();) {
    auto lb = lower_bounds.find(*ub);
    if (lb != lower_bounds.end()) {
      equalities.insert(*lb);
      lower_bounds.erase(lb);
      ub = upper_bounds.erase(ub);
    } else {
      ++ub;
    }
  }
}

IntConstraints SolveLinearInequalities(const IntConstraints& system_to_solve) {
  LOG(INFO) << "solving inequalities " << system_to_solve;

  Map<Var, Range> vranges = ConvertGroupedBoundToRange(system_to_solve->ranges);

  arith::Analyzer analyzer;
  analyzer.Bind(vranges);

  // The algorithm consists in doing the following things for each variable v
  // - Take formulas from `current_ineq_set_to_solve` and classify them according to polarity wrt v
  // - Combine each formula of positive polarity (wrt v) with each formula of negative polarity
  // - Put the resulting combinations into `next_ineq_set_to_solve` along with unclassifiable formulas
  // - Replace `current` with `next_ineq_set_to_solve` and move to the next variable

  // normalized inequality
  // current and next_ineq_set_to_solve are sorted to enable some heuristics
  std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> current_ineq_set_to_solve;
  std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> next_ineq_set_to_solve;
  // A vector of pairs (c, e), c > 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_pos;
  // A vector of pairs (c, e), c < 0, representing formulas of the form c*v + e <= 0
  std::vector<std::pair<int64_t, PrimExpr>> coef_neg;

  // formulas we don't know what to do with
  std::vector<PrimExpr> rest;

  // Simplify each inequality into the form `expr <= 0` and add to current formulas
  for (const PrimExpr& ineq : system_to_solve->relations) {
    // TODO: SuperSimplify(ineq, vranges)
    AddInequality(current_ineq_set_to_solve, NormalizeComparisons()(analyzer.Simplify(ineq)), analyzer);
  }

  DebugPrint(current_ineq_set_to_solve,
             next_ineq_set_to_solve,
             rest,
             coef_pos,
             coef_neg);

  Map<Var, IntGroupedBounds> res_bounds;
  for (const Var& v : system_to_solve->variables) {
    CHECK(!res_bounds.count(v)) <<
      "Variable " << v << " appears more than one time in the `variables` which might be a bug";

    next_ineq_set_to_solve.clear();
    coef_pos.clear();
    coef_neg.clear();

    // Add bounds from vranges
    if (system_to_solve->ranges.count(v)) {
      const Range& range = vranges[v];
      PrimExpr range_lbound = analyzer.Simplify(range->min);
      PrimExpr range_ubound = analyzer.Simplify(range->min + range->extent - 1);
      coef_neg.push_back({-1, range_lbound});
      coef_pos.push_back({1, -range_ubound});
    }

    ClassifyByPolarity(v, current_ineq_set_to_solve, next_ineq_set_to_solve, rest, coef_pos, coef_neg, analyzer);

    DebugPrint(current_ineq_set_to_solve,
               next_ineq_set_to_solve,
               rest,
               coef_pos,
               coef_neg);

    // Combine each positive inequality with each negative one (by adding them together)
    for (const auto& pos : coef_pos) {
      for (const auto& neg : coef_neg) {
        auto first_gcd = gcd(pos.first, -neg.first);
        PrimExpr c_pos = make_const(v.dtype(), neg.first/first_gcd);
        PrimExpr c_neg = make_const(v.dtype(), pos.first/first_gcd);
        // eliminate the current variable
        PrimExpr new_lhs = c_neg*neg.second - c_pos*pos.second;
        PrimExpr new_ineq = LENode::make(new_lhs, make_zero(pos.second.dtype()));
        // it helps to simplify (((y + 10) - (-1*(y - 20))) <= 0) => y - 5 <= 0
        // otherwise it's (y*2) - 10 <= 0
        new_ineq = NormalizeComparisons()(analyzer.rewrite_simplify(analyzer.Simplify(new_ineq)));
        AddInequality(next_ineq_set_to_solve, new_ineq, analyzer);
      }
    }

    // Now we have to generate resulting (in)equalities for the variable v

    // Find the common denominator in a sense
    // We will generate formulas of the form coef_lcm*v <= bound
    int64_t coef_lcm = 1;
    for (const auto& pos : coef_pos) {
      coef_lcm = lcm(coef_lcm, pos.first);
    }
    for (const auto& neg : coef_neg) {
      coef_lcm = lcm(coef_lcm, -neg.first);
    }

    // The resulting lower and upper bounds stored in sorted vectors
    std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> upper_bounds;
    std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> lower_bounds;
    upper_bounds.reserve(coef_pos.size());
    lower_bounds.reserve(coef_neg.size());

    for (const auto& pos : coef_pos) {
      PrimExpr bound = make_const(v.dtype(), -coef_lcm/pos.first)*pos.second;
      bound = analyzer.Simplify(bound);
      // Don't add if any of the existing bounds is better
      if (std::any_of(upper_bounds.begin(), upper_bounds.end(),
                      [&bound, &analyzer](const PrimExpr& o)
                      { return analyzer.CanProve(o - bound <= 0); })) {
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
      upper_bounds.insert(bound);
    }
    for (const auto& neg : coef_neg) {
      PrimExpr bound = make_const(v.dtype(), -coef_lcm/neg.first)*neg.second;
      bound = analyzer.Simplify(bound);
      // Don't add if any of the existing bounds is better
      if (std::any_of(lower_bounds.begin(), lower_bounds.end(),
                      [&bound, &analyzer](const PrimExpr& o)
                      { return analyzer.CanProve(o - bound >= 0); })) {
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
      lower_bounds.insert(bound);
    }

    std::unordered_set<PrimExpr, StructuralHash, StructuralEqual> equal;
    equal.reserve(std::min(upper_bounds.size(), lower_bounds.size()));
    MoveEquality(upper_bounds, lower_bounds, equal);

    // Write it to the result.
    IntGroupedBounds bnds(make_const(v.dtype(), coef_lcm),
        Array<PrimExpr>(lower_bounds.begin(), lower_bounds.end()),
        Array<PrimExpr>(equal.begin(), equal.end()),
        Array<PrimExpr>(upper_bounds.begin(), upper_bounds.end())
    );
    res_bounds.Set(v, bnds);
    LOG(INFO) << "Bound of " << v << bnds;

    std::swap(current_ineq_set_to_solve, next_ineq_set_to_solve);
  }

  // Everything that is left goes to res.relations
  Array<PrimExpr> other_conditions;
  for (const PrimExpr& e : current_ineq_set_to_solve) {
    PrimExpr e_simp = analyzer.Simplify(e);
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

  IntConstraints res(system_to_solve->variables, res_bounds, other_conditions);

  return res;
}

IntConstraints SolveInequalitiesRange(const IntConstraints& inequalities) {
  // Resulting ranges will contain ranges for the new variables and for the variables that are
  // not in the inequalities->variables but are in inequalities->ranges (jac_xxx)
  Map<Var, IntGroupedBounds> res_ranges;
  // we get a set of equality, lower, upper bound of each variable.
  auto solved_system = SolveLinearInequalities(inequalities);
  LOG(INFO) << "solved system = " << solved_system;

  Array<Var> res_variables;
  Array<PrimExpr> res_relations;

  // this keeps being updated during determining the range of each variable.
  Map<Var, Range> vranges = ConvertGroupedBoundToRange(inequalities->ranges);

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = inequalities->variables.rbegin(); it != inequalities->variables.rend(); ++it) {
    arith::Analyzer analyzer;
    analyzer.Bind(vranges);

    const Var& var = *it;
    auto bnd = solved_system->ranges[var];
    if (is_one(bnd->coef) && !bnd->equal.empty()) {
      // There is an equation of the form `v == expr`, so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity, so it must be
      // the simplest one.
      Range best_range(bnd->equal[0], analyzer.Simplify(bnd->equal[0] + 1));
      res_ranges.Set(var, IntGroupedBounds::range(best_range));
      vranges.Set(var, best_range);
    } else {
      if (vranges.count(var) > 0) {
        bnd = bnd + vranges[var];
      }
      LOG(INFO) << "bnd = " << bnd;
      LOG(INFO) << "vranges = " << vranges;

      auto best_range = bnd.FindBestRange(vranges);
      LOG(INFO) << "best range for " << var << " = " << best_range;

      res_ranges.Set(var, IntGroupedBounds::range(best_range));
      vranges.Set(var, best_range);
    }
  }

  // Add the original conditions (with variables substituted) to the resulting conditions
  arith::Analyzer analyzer;
  analyzer.Bind(vranges);
  for (const PrimExpr& old_cond : solved_system.as_conditions()) {
    if (!analyzer.CanProve(old_cond)) {
      // those not represented in vranges (res_ranges)
      res_relations.push_back(old_cond);
    }
  }

  IntConstraints system(inequalities->variables, res_ranges, res_relations);

  return system;
}

// Deskew the given domain
IntConstraintsTransform DeskewRange(const IntConstraints& inequalities) {
  // Resulting ranges will contain ranges for the new variables and for the variables that are
  // not in the inequalities->variables but are in inequalities->ranges (jac_xxx)
  Map<Var, IntGroupedBounds> res_ranges;
  // we get a set of equality, lower, upper bound of each variable.
  auto solved_system = SolveLinearInequalities(inequalities);
  LOG(INFO) << "solved system = " << solved_system;

  arith::Analyzer analyzer;

  Map<Var, PrimExpr> res_old_to_new;
  Map<Var, PrimExpr> res_new_to_old;
  Array<Var> res_variables;
  Array<PrimExpr> res_relations;

  // this keeps being updated during determining the range of each variable.
  Map<Var, Range> vranges = ConvertGroupedBoundToRange(inequalities->ranges);
  analyzer.Bind(vranges);

  // We process variables in the reverse direction to start with the most independent one.
  // This order is needed to compute new ranges.
  for (auto it = inequalities->variables.rbegin(); it != inequalities->variables.rend(); ++it) {
    const Var& var = *it;
    auto bnd = solved_system->ranges[var];
    // Note that we replace old vars with new ones
    bnd = bnd.Substitute(res_old_to_new);

    if (is_one(bnd->coef) && !bnd->equal.empty()) {
      // There is an equation of the form `v == expr`, so this variable can be completely removed.
      // Note that we use the 0-th expression because they are ordered by complexity, so it must be
      // the simplest one.
      // TODO
      res_old_to_new.Set(var, bnd->equal[0]);
    } else {
      if (vranges.count(var) > 0) {
        bnd = bnd + vranges[var];
      }

      auto best_range = bnd.FindBestRange(vranges);
      LOG(INFO) << "best range for " << var << " = " << best_range;

      std::string suffix = ".shifted";
      Var new_var = var.copy_with_suffix(suffix);

      if (is_const_int(best_range->extent, 1)) {
        // Don't create an itervar, just replace it everywhere with its min
        res_old_to_new.Set(var, best_range->min);
      } else {
        // created new_var starts from 0
        res_old_to_new.Set(var, new_var + best_range->min);
        // Note that we are substituting old with new, so best_lower contains new var,
        // that is we have to substitute new with old in best_lower here
        res_new_to_old.Set(new_var,
                           analyzer.Simplify(var - Substitute(best_range->min, res_new_to_old)));

        // Add the new var to the resulting axis
        auto range = Range(make_zero(new_var.dtype()), best_range->extent);
        res_variables.push_back(new_var);
        res_ranges.Set(new_var, IntGroupedBounds::range(range));

        vranges.Set(new_var, range);
        analyzer.Bind(new_var, range);
      }
    }
  }

  // Add the original conditions (with variables substituted) to the resulting conditions
  for (const PrimExpr& old_cond : solved_system.as_conditions()) {
    PrimExpr new_cond = analyzer.Simplify(Substitute(old_cond, res_old_to_new));
    if (!is_const_int(new_cond, 1)) {
      // those not represented in vranges (res_ranges)
      res_relations.push_back(new_cond);
    }
  }

  // Reverse the axis so that it matches the order of the original variables
  res_variables = Array<Var>(res_variables.rbegin(), res_variables.rend());

  IntConstraints new_inequalities(res_variables, res_ranges, res_relations);
  IntConstraintsTransform transform(inequalities, new_inequalities, res_old_to_new, res_new_to_old);

  return transform;
}

TVM_REGISTER_GLOBAL("arith.SolveLinearInequalities")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() == 1) {
      *ret = SolveLinearInequalities(args[0]);
    } else if (args.size() == 3) {
      IntConstraints problem(args[0], args[1], args[2]);
      *ret = SolveLinearInequalities(problem);
    } else {
      LOG(FATAL) << "arith.SolveLinearInequalities expects 1 or 3 arguments, gets " << args.size();
    }
  });

TVM_REGISTER_GLOBAL("arith.SolveInequalitiesRange")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    if (args.size() == 1) {
      *ret = SolveInequalitiesRange(args[0]);
    } else if (args.size() == 3) {
      IntConstraints problem(args[0], args[1], args[2]);
      *ret = SolveInequalitiesRange(problem);
    } else {
      LOG(FATAL) << "arith.SolveInequalitiesRange expects 1 or 3 arguments, gets " << args.size();
    }
  });

TVM_REGISTER_GLOBAL("arith.DeskewRange")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  if (args.size() == 1) {
      *ret = DeskewRange(args[0]);
    } else if (args.size() == 3) {
      IntConstraints problem(args[0], args[1], args[2]);
      *ret = DeskewRange(problem);
    } else {
      LOG(FATAL) << "arith.DeskewRange expects 1 or 3 arguments, gets " << args.size();
    }
  });

}  // namespace arith
}  // namespace tvm
