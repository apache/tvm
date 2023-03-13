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
 * \file partition_finder.cc
 */

#include "./partition_finder.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/bound.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {

using namespace tvm::tir;

PartitionDecision::PartitionDecision(IntervalSet interval, Map<PrimExpr, Bool> cond_map) {
  auto n = make_object<PartitionDecisionNode>();
  n->interval = interval;
  n->cond_map = cond_map;
  this->data_ = n;
}

/*!
 * Each term (cond, cond_value, interval) represents the fact that
 * condition cond is proven to have value cond_value (true or false)
 * in interval.
 */
struct PartitionInterval {
  PrimExpr cond;
  IntSet interval;
  bool cond_value;

  PartitionInterval(PrimExpr cond, IntSet interval, bool cond_value)
      : cond(cond), interval(interval), cond_value(cond_value) {}
};

// Finder try best to find partitions for hinted vars
#define DEFINE_PARTITION_FINDER_VISIT_CMP_OP(OpNodeT) \
  void VisitExpr_(const OpNodeT* op) final {          \
    if (has_partition_hint_) {                        \
      DeduceCondition(GetRef<PrimExpr>(op));          \
      return;                                         \
    }                                                 \
    StmtExprVisitor::VisitExpr_(op);                  \
  }

// Populate partitions data structure, i.e., for a specific variable,
// find an interval in which each condition has fixed true or false value
class PartitionFinder : public StmtExprVisitor {
 public:
  explicit PartitionFinder(Var current_var,
                           const std::unordered_map<const VarNode*, IntSet>& hint_map,
                           const std::unordered_map<const VarNode*, IntSet>& relax_map,
                           bool has_partition_hint, bool deduce_min_max)
      : current_var_(current_var),
        has_partition_hint_(has_partition_hint),
        deduce_min_max_(deduce_min_max),
        hint_map_(hint_map),
        relax_map_(relax_map) {
    for (const auto& kv : hint_map) {
      out_vars_.insert(kv.first);
    }
    for (const auto& kv : relax_map) {
      out_vars_.insert(kv.first);
    }
  }

  void VisitStmt_(const ForNode* op) final {
    auto f_vset_contains = [this](const VarNode* var) { return out_vars_.count(var); };
    if (UsesVar(op->min, f_vset_contains) || UsesVar(op->extent, f_vset_contains)) return;

    const VarNode* var = op->loop_var.get();
    hint_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    relax_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    StmtExprVisitor::VisitStmt_(op);
    relax_map_.erase(var);
    hint_map_.erase(var);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    // handle thread_axis
    if (op->attr_key == tir::attr::thread_extent) {
      const IterVarNode* thread_axis = op->node.as<IterVarNode>();
      ICHECK(thread_axis);
      const VarNode* var = thread_axis->var.get();
      IntSet dom = IntSet::FromRange(Range(make_zero(op->value.dtype()), op->value));
      hint_map_.insert({var, dom});
      relax_map_.insert({var, dom});
      StmtExprVisitor::VisitStmt_(op);
      relax_map_.erase(var);
      hint_map_.erase(var);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(tir::builtin::likely())) {
      DeduceCondition(op->args[0]);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(GENode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(GTNode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(LENode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(LTNode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(EQNode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(NENode);

  void VisitExpr_(const MinNode* op) final {
    if (deduce_min_max_) {
      LOG(INFO) << "deduce " << (op->a) << " < " << op->b;
      DeduceCondition(op->a < op->b);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MaxNode* op) final {
    if (deduce_min_max_) {
      LOG(INFO) << "deduce " << (op->a) << " < " << op->b;
      DeduceCondition(op->a < op->b);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<PartitionInterval> partitions;

 private:
  void DeduceCondition(const PrimExpr& cond) {
    // For cond, find out the interval, if exists, in which we can prove that cond is
    // true. Also find the interval, if exists, in which we can prove that cond is
    // false.
    if (UsesVar(cond, [this](const VarNode* var) { return var == current_var_.get(); })) {
      IntSet interval = DeduceBound(current_var_, cond, hint_map_, relax_map_);
      if (!interval.IsNothing()) {
        // cond is true within interval
        partitions.emplace_back(cond, interval, true);
      }
      PrimExpr inverse_cond = InverseCond(cond);
      if (inverse_cond.defined()) {
        IntSet interval = DeduceBound(current_var_, inverse_cond, hint_map_, relax_map_);
        if (!interval.IsNothing()) {
          // cond is false within interval
          partitions.emplace_back(cond, interval, false);
        }
      }
    }
  }

  PrimExpr InverseCond(const PrimExpr& cond) {
    PrimExpr inverse_cond;
    if (const LTNode* op = cond.as<LTNode>()) {
      // a < b -> a >= b
      inverse_cond = GE(op->a, op->b);
    } else if (const GTNode* op = cond.as<GTNode>()) {
      // a > b -> a <= b
      inverse_cond = LE(op->a, op->b);
    } else if (const LENode* op = cond.as<LENode>()) {
      // a <= b -> a > b
      inverse_cond = GT(op->a, op->b);
    } else if (const GENode* op = cond.as<GENode>()) {
      // a >= b -> a < b
      inverse_cond = LT(op->a, op->b);
    } else if (const EQNode* op = cond.as<EQNode>()) {
      // a == b -> a != b
      inverse_cond = NE(op->a, op->b);
      // a != b -> a == b
    } else if (const NENode* op = cond.as<NENode>()) {
      inverse_cond = EQ(op->a, op->b);
    }
    return inverse_cond;
  }

  Var current_var_;
  bool has_partition_hint_;
  bool deduce_min_max_;
  std::unordered_set<const VarNode*> out_vars_;
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
};

// Returns an interval (in the first component) in which all the conditions
// given in the second component provably have value given by cond_value
PartitionDecision GetIntervalAndCondset(const std::vector<PartitionInterval>& partitions,
                                        const arith::IntervalSet& for_interval,
                                        arith::Analyzer* analyzer, bool cond_value,
                                        bool allow_partial_match) {
  Array<IntSet> sets;
  Map<PrimExpr, Bool> cond_map;

  for (const auto& cand : partitions) {
    if (cand.cond_value == cond_value) {
      arith::IntervalSet interval = Downcast<arith::IntervalSet>(cand.interval);
      arith::IntervalSet intersection = arith::Intersect(analyzer, interval, for_interval);
      if (!intersection->IsEmpty()) {
        sets.push_back(cand.interval);
        cond_map.Set(cand.cond, Bool(cond_value));
      }
    }
  }
  IntSet interval = sets.empty() ? IntSet::Nothing() : Intersect(sets);

  // Try to find the intersection of the cond_intervals until the intersection
  // is nothing when has_partition_hint is true.
  if (interval.IsNothing() && allow_partial_match) {
    arith::IntervalSet cond_intersection = arith::IntervalSet::Everything();
    cond_map.clear();
    for (const auto& cand : partitions) {
      if (cand.cond_value == cond_value) {
        arith::IntervalSet cond_interval = Downcast<arith::IntervalSet>(cand.interval);
        arith::IntervalSet intersection = arith::Intersect(analyzer, cond_interval, for_interval);
        if (!intersection->IsEmpty()) {
          cond_intersection = arith::Intersect(analyzer, cond_intersection, cond_interval);
          // Return the latest interval and cond_set if the cond_intersection is nothing.
          if (!cond_intersection->IsEmpty()) {
            cond_map.Set(cand.cond, Bool(cond_value));
            interval = arith::IntervalSet(analyzer->Simplify(cond_intersection->min_value),
                                          analyzer->Simplify(cond_intersection->max_value));
          } else {
            break;
          }
        }
      }
    }
  }

  return PartitionDecision(Downcast<arith::IntervalSet>(interval), cond_map);
}

Optional<PartitionDecision> SearchBestPartition(
    const Var& var, arith::IntervalSet loop_range,
    const std::unordered_map<const VarNode*, IntSet>& hint_map,
    const std::unordered_map<const VarNode*, IntSet>& relax_map, bool partition_likely_cond_only,
    bool deduce_min_max, arith::Analyzer* analyzer, ObjectRef stmt_or_expr) {
  PartitionFinder finder(var, hint_map, relax_map, !partition_likely_cond_only, deduce_min_max);
  ICHECK(stmt_or_expr.defined());
  if (stmt_or_expr->IsInstance<tir::StmtNode>()) {
    finder(Downcast<Stmt>(stmt_or_expr));
  } else if (stmt_or_expr->IsInstance<PrimExprNode>()) {
    finder(Downcast<PrimExpr>(stmt_or_expr));
  } else {
    LOG(FATAL) << "Illegal argument type of 'stmt_or_expr': " << stmt_or_expr->GetTypeKey();
  }
  const auto& candidate_partitions = finder.partitions;

  if (candidate_partitions.empty()) {
    return NullOpt;
  }

  {
    PartitionDecision partition = GetIntervalAndCondset(candidate_partitions, loop_range, analyzer,
                                                        true, !partition_likely_cond_only);
    if (!partition->interval.IsNothing()) {
      return partition;
    }
  }

  {
    // if such interval doesn't exist, find an interval in which all
    // conditions on var are false
    PartitionDecision partition = GetIntervalAndCondset(candidate_partitions, loop_range, analyzer,
                                                        false, !partition_likely_cond_only);
    if (!partition->interval.IsNothing()) {
      return partition;
    }
  }

  // we couldn't find an interval in which the conditions are
  // provably true or false.  Therefore, we can't partition the loop
  // based on those conds
  return NullOpt;
}

class PartitionedIntBoundEstimator {
 public:
  explicit PartitionedIntBoundEstimator(
      const std::vector<Var>& vars,
      const std::unordered_map<const VarNode*, arith::IntSet>& dom_map) {
    hint_map_ = dom_map;
    vars_ = vars;
  }

  ConstIntBound Visit(PrimExpr expr) {
    LOG(INFO) << expr << " " << rec_depth_;
    expr = analyzer_.Simplify(expr);
    size_t cur_rec_depth = rec_depth_;
    while (rec_depth_ < vars_.size()) {
      Var var = vars_[rec_depth_];
      Optional<ConstIntBound> bound = TryPartition(expr, var);
      if (bound.defined()) {
        rec_depth_ = cur_rec_depth;
        return bound.value();
      }
      rec_depth_ += 1;
    }

    rec_depth_ = cur_rec_depth;
    return analyzer_.const_int_bound(expr);
  }

 private:
  std::vector<Var> vars_;
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  arith::Analyzer analyzer_;
  size_t rec_depth_ = 0;

  Optional<ConstIntBound> TryPartition(PrimExpr expr, Var var) {
    auto hint_it = hint_map_.find(var.get());
    ICHECK(hint_it != hint_map_.end());
    IntervalSet cur_interval = Downcast<IntervalSet>(hint_it->second);

    LOG(INFO) << "Try partition " << expr << " " << var << " " << cur_interval;
    Range cur_range = Range(cur_interval.min(), cur_interval.max() + 1);

    Optional<PartitionDecision> opt_partition = arith::SearchBestPartition(
        /*var=*/var, /*loop_range=*/cur_interval, /*hint_map=*/hint_map_, /*relax_map=*/{},
        /*process_likely_only=*/false,
        /*deduce_min_max=*/true,
        /*analyzer=*/&analyzer_,
        /*stmt_or_exor=*/expr);

    if (!opt_partition.defined()) {
      return NullOpt;
    }

    PartitionDecision partition = opt_partition.value();

    arith::IntervalSet middle_interval = partition->interval;
    // middle_interval is the subrange of the loop variable range for which a
    // set of conditions are true (or false resp.)
    // The part of the loop variable range that is before (after resp.) that
    // subrange is prefixed with pre- (post- resp.)

    // Calculating pre-subrange and generating code for it.
    // pre-subrange = [min, body_begin)
    Optional<ConstIntBound> pre_bound = NullOpt;
    if (middle_interval->HasLowerBound()) {
      PrimExpr extent = analyzer_.Simplify(middle_interval.min() - cur_interval.min());
      if (!analyzer_.CanProve(extent <= 0)) {
        bool non_empty = analyzer_.CanProve(extent > 0);
        PrimExpr min =
            non_empty ? cur_interval.min() : tvm::min(middle_interval.min(), cur_interval.min());
        Range pre_range = Range::FromMinExtent(min, extent);
        hint_it->second = IntervalSet::FromRange(pre_range);
        analyzer_.Bind(var, pre_range, true);

        if (!non_empty) ++rec_depth_;
        pre_bound = Visit(expr);
        if (!non_empty) --rec_depth_;

        hint_it->second = cur_interval;
        analyzer_.Bind(var, cur_range, true);
      }
    }

    // Calculating post-subrange and generating code for it.
    // post-subrange = [post_doubt_begin, max+1)
    Optional<ConstIntBound> post_bound = NullOpt;
    if (middle_interval->HasUpperBound()) {
      PrimExpr extent = analyzer_.Simplify(cur_interval.max() - middle_interval.max());
      if (!analyzer_.CanProve(extent <= 0)) {
        bool non_empty = analyzer_.CanProve(extent > 0);
        PrimExpr min =
            non_empty ? middle_interval.max() : tvm::max(middle_interval.max(), cur_interval.max());
        Range post_range = Range::FromMinExtent(min, extent);
        hint_it->second = IntervalSet::FromRange(post_range);
        analyzer_.Bind(var, post_range, true);

        if (!non_empty) ++rec_depth_;
        post_bound = Visit(expr);
        if (!non_empty) --rec_depth_;

        hint_it->second = cur_interval;
        analyzer_.Bind(var, cur_range, true);
      }
    }

    if (!pre_bound && !post_bound) {
      return NullOpt;
    }

    middle_interval =
        Downcast<arith::IntervalSet>(arith::Intersect({middle_interval, cur_interval}));
    hint_it->second = middle_interval;
    LOG(INFO) << middle_interval;
    analyzer_.Bind(var, Range(middle_interval.min(), middle_interval.max() + 1), true);
    ConstIntBound middle_bound = Visit(expr);
    hint_it->second = cur_interval;
    analyzer_.Bind(var, cur_range, true);

    ConstIntBound merged = middle_bound;
    if (pre_bound.defined()) {
      merged = ConstIntBound(std::min(pre_bound.value()->min_value, merged->min_value),
                             std::max(pre_bound.value()->max_value, merged->max_value));
    }
    if (post_bound.defined()) {
      merged = ConstIntBound(std::min(post_bound.value()->min_value, merged->min_value),
                             std::max(post_bound.value()->max_value, merged->max_value));
    }
    return merged;
  }
};

ConstIntBound EstimatePartitionedConstIntBound(
    const PrimExpr& e,

    const std::vector<Var>& vars,
    const std::unordered_map<const VarNode*, arith::IntSet>& dom_map) {
  PartitionedIntBoundEstimator estimator(vars, dom_map);
  return estimator.Visit(e);
}

}  // namespace arith
}  // namespace tvm
