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
 * \file int_constraints.cc
 * \brief The integer constraints data structures.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_solver.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "../tir/transforms/ir_utils.h"

namespace tvm {
namespace arith {

Array<PrimExpr> AsConditions(const Array<Var>& variables, const Map<Var, IntGroupBounds>& bounds,
                             const Array<PrimExpr>& relations) {
  Array<PrimExpr> res;
  // use variables to keep the order of iteration
  // so as to get rid of any non-determinism.
  ICHECK_EQ(variables.size(), bounds.size());
  for (const auto v : variables) {
    ICHECK(bounds.count(v));
    const auto& bnds = bounds[v];
    PrimExpr lhs = bnds->coef * v;
    for (const PrimExpr& rhs : bnds->equal) {
      res.push_back(lhs == rhs);
    }
    for (const PrimExpr& rhs : bnds->lower) {
      res.push_back(lhs >= rhs);
    }
    for (const PrimExpr& rhs : bnds->upper) {
      res.push_back(lhs <= rhs);
    }
  }
  for (const PrimExpr& e : relations) {
    res.push_back(e);
  }
  return res;
}

IntGroupBounds::IntGroupBounds(PrimExpr coef, Array<PrimExpr> lower, Array<PrimExpr> equal,
                               Array<PrimExpr> upper) {
  ICHECK(coef.dtype().is_int() || coef.dtype().is_uint())
      << "Coefficient in IntGroupBounds must be integers";
  ObjectPtr<IntGroupBoundsNode> node = make_object<IntGroupBoundsNode>();
  node->coef = std::move(coef);
  node->lower = std::move(lower);
  node->equal = std::move(equal);
  node->upper = std::move(upper);
  data_ = std::move(node);
}

IntGroupBounds IntGroupBounds::FromRange(const Range& r) {
  Analyzer analyzer;
  PrimExpr coef = tir::make_const(r->min.dtype(), 1);
  Array<PrimExpr> equal;
  Array<PrimExpr> lower;
  Array<PrimExpr> upper;
  if (tir::is_one(r->extent)) {
    equal.push_back(r->min);
  } else {
    lower.push_back(r->min);
    upper.push_back(analyzer.Simplify(r->min + r->extent - 1));
  }
  return IntGroupBounds(coef, lower, equal, upper);
}

IntGroupBounds IntGroupBounds::operator+(const Range& r) {
  Analyzer analyzer;
  Array<PrimExpr> equal;
  Array<PrimExpr> lower;
  Array<PrimExpr> upper;
  const PrimExpr& coef = operator->()->coef;
  if (tir::is_one(r->extent)) {
    equal.push_back(analyzer.Simplify(r->min * coef));
  } else {
    lower.push_back(analyzer.Simplify(r->min * coef));
    upper.push_back(analyzer.Simplify((r->min + r->extent - 1) * coef));
  }
  for (const auto& eq : operator->()->equal) equal.push_back(eq);
  for (const auto& lb : operator->()->lower) lower.push_back(lb);
  for (const auto& ub : operator->()->upper) upper.push_back(ub);
  return IntGroupBounds(coef, lower, equal, upper);
}

IntGroupBounds IntGroupBounds::Substitute(const Map<Var, PrimExpr>& subst) const {
  auto apply_fun = [&subst](const PrimExpr& e) { return tir::Substitute(e, subst); };
  return IntGroupBounds(tir::Substitute(operator->()->coef, subst),
                        tir::UpdateArray(operator->()->lower, apply_fun),
                        tir::UpdateArray(operator->()->equal, apply_fun),
                        tir::UpdateArray(operator->()->upper, apply_fun));
}

Range IntGroupBounds::FindBestRange(const Map<Var, Range>& vranges_addl) const {
  Analyzer analyzer;
  analyzer.Bind(vranges_addl);

  std::unordered_map<const VarNode*, IntSet> var_intsets;
  for (auto kv : vranges_addl) {
    var_intsets[kv.first.get()] = IntSet::FromRange(kv.second);
  }

  const Array<PrimExpr>& equal = operator->()->equal;
  const PrimExpr& coef = operator->()->coef;

  std::vector<PrimExpr> lowers(equal.begin(), equal.end());
  std::vector<PrimExpr> uppers(equal.begin(), equal.end());
  for (const auto& expr : operator->()->lower) {
    lowers.push_back(expr);
  }
  for (const auto& expr : operator->()->upper) {
    uppers.push_back(expr);
  }

  if (lowers.size() == 1 && uppers.size() == 1 && tir::is_one(coef)) {
    return Range(analyzer.Simplify(lowers[0]), analyzer.Simplify(uppers[0] + 1));
  }

  // Here we will try all pairs of lower and upper bounds and find the best pair, that is, the
  // pair with the minimal difference between the upper and the lower.
  // Note that the bounds are for v, not for v*coef

  // The lower bound of the best pair so far
  PrimExpr best_lower;
  // The difference between the upper and the lower of the best pair, maybe overapproximation
  PrimExpr best_diff_over;

  for (const PrimExpr& low : lowers) {
    for (const PrimExpr& upp : uppers) {
      // Since diff may depend on some other variables, we compute its overapproximation
      Optional<PrimExpr> diff_over;
      PrimExpr diff_1 = analyzer.Simplify(floordiv(upp - low, coef), 3);
      IntSet diff_set1 = EvalSet(diff_1, var_intsets);
      if (diff_set1.HasUpperBound()) {
        diff_over = analyzer.Simplify(diff_set1.max(), 3);
      }

      // low is the lower bound for v*coef, but we need the lower bound for v.
      // We use rounding-up division to compute it. Since we want to use a single formula
      PrimExpr low_divided = analyzer.Simplify(floordiv(low + coef - 1, coef), 3);

      // Compute another difference which may be more precise (or not).
      PrimExpr diff_2 = analyzer.Simplify(floordiv(upp, coef) - low_divided, 3);
      IntSet diff_set2 = EvalSet(diff_2, var_intsets);
      if (diff_set2.HasUpperBound()) {
        PrimExpr diff_over_2 = analyzer.Simplify(diff_set2.max(), 3);
        diff_over = diff_over.defined() ? (analyzer.CanProve(diff_over_2 - diff_over.value() < 0)
                                               ? diff_over_2
                                               : diff_over.value())
                                        : diff_over_2;
      }

      // If it is provable that the new one is strictly better than the current best one,
      // then replace it. Note that we are biased towards earlier pairs which should be simpler.
      if (diff_over.defined() && (!best_diff_over.defined() ||
                                  analyzer.CanProve(diff_over.value() - best_diff_over < 0))) {
        best_lower = low_divided;
        best_diff_over = diff_over.value();
      }
    }
  }

  if (!best_lower.defined()) {
    ICHECK(!best_diff_over.defined());
    return Range();
  }
  return Range::FromMinExtent(best_lower, analyzer.Simplify(best_diff_over + 1));
}

TVM_REGISTER_NODE_TYPE(IntGroupBoundsNode);

TVM_REGISTER_GLOBAL("arith.IntGroupBounds")
    .set_body_typed([](PrimExpr coef, Array<PrimExpr> lower, Array<PrimExpr> equal,
                       Array<PrimExpr> upper) {
      return IntGroupBounds(coef, lower, equal, upper);
    });

TVM_REGISTER_GLOBAL("arith.IntGroupBounds_from_range").set_body_typed(IntGroupBounds::FromRange);

TVM_REGISTER_GLOBAL("arith.IntGroupBounds_FindBestRange")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      ICHECK(args.size() == 1 || args.size() == 2);
      IntGroupBounds bounds = args[0];
      if (args.size() == 1) {
        *ret = bounds.FindBestRange();
      } else if (args.size() == 2) {
        *ret = bounds.FindBestRange(args[1]);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntGroupBoundsNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IntGroupBoundsNode*>(node.get());
      p->stream << "IntGroupBounds(coef=" << op->coef << ", lower=" << op->lower
                << ", equal=" << op->equal << ", upper=" << op->upper << ")";
    });

IntConstraints::IntConstraints(Array<Var> variables, Map<Var, Range> ranges,
                               Array<PrimExpr> relations) {
  ObjectPtr<IntConstraintsNode> node = make_object<IntConstraintsNode>();
  if (!variables.defined()) {
    variables = Array<Var>();
  }
  if (!ranges.defined()) {
    ranges = Map<Var, Range>();
  }
  ICHECK(relations.defined());
  for (const auto& var : variables) {
    ICHECK(var.dtype().is_int() || var.dtype().is_uint())
        << "Variables in IntConstraints must be integers";
  }
  node->variables = std::move(variables);
  node->ranges = std::move(ranges);
  node->relations = std::move(relations);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(IntConstraintsNode);

TVM_REGISTER_GLOBAL("arith.IntConstraints")
    .set_body_typed([](Array<Var> variables, Map<Var, Range> ranges, Array<PrimExpr> relations) {
      return IntConstraints(variables, ranges, relations);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntConstraintsNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IntConstraintsNode*>(node.get());
      p->stream << "IntConstraints(" << op->variables << ", " << op->ranges << ", " << op->relations
                << ")";
    });

IntConstraintsTransform::IntConstraintsTransform(IntConstraints src, IntConstraints dst,
                                                 Map<Var, PrimExpr> src_to_dst,
                                                 Map<Var, PrimExpr> dst_to_src) {
  ObjectPtr<IntConstraintsTransformNode> node = make_object<IntConstraintsTransformNode>();
  node->src = std::move(src);
  node->dst = std::move(dst);
  node->src_to_dst = std::move(src_to_dst);
  node->dst_to_src = std::move(dst_to_src);
  data_ = std::move(node);
}

IntConstraintsTransform IntConstraintsTransform::operator+(
    const IntConstraintsTransform& other) const {
  ICHECK(other->src.same_as(operator->()->dst));
  Map<Var, PrimExpr> dst_to_src;
  Map<Var, PrimExpr> src_to_dst;

  Analyzer ana_first;
  ana_first.Bind(operator->()->src->ranges);
  for (auto p : other->dst_to_src) {
    dst_to_src.Set(p.first, ana_first.Simplify(Substitute(p.second, operator->()->dst_to_src)));
  }

  Analyzer ana_second;
  ana_second.Bind(other->dst->ranges);
  for (auto p : operator->()->src_to_dst) {
    src_to_dst.Set(p.first, ana_second.Simplify(Substitute(p.second, other->src_to_dst)));
  }
  return IntConstraintsTransform(operator->()->src, other->dst, src_to_dst, dst_to_src);
}

TVM_REGISTER_NODE_TYPE(IntConstraintsTransformNode);

TVM_REGISTER_GLOBAL("arith.IntConstraintsTransform")
    .set_body_typed([](IntConstraints src, IntConstraints dst, Map<Var, PrimExpr> src_to_dst,
                       Map<Var, PrimExpr> dst_to_src) {
      return IntConstraintsTransform(src, dst, src_to_dst, dst_to_src);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntConstraintsTransformNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IntConstraintsTransformNode*>(node.get());
      p->stream << "IntConstraintsTransform("
                << "\n\t" << op->src << "\n\t" << op->dst << "\n\t" << op->src_to_dst << "\n\t"
                << op->dst_to_src << "\n)";
    });

}  // namespace arith
}  // namespace tvm
