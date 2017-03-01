/*!
 *  Copyright (c) 2016 by Contributors
 * \file loop_partition.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>
#include "../arithmetic/int_set.h"

namespace tvm {
namespace ir {

using arith::IntSet;
using Halide::Internal::const_true;
using Halide::Internal::const_false;
using Halide::Internal::Interval; // for pos_inf & neg_inf

// a partition means condition is equal to true_value in the interval
struct Partition {
  Expr condition;
  Expr old_expr;
  Expr true_value;
  IntSet interval;
};

bool ExprUseVar(Expr expr, const Variable* var) {
  bool success = false;
  PostOrderVisit(expr, [&var, &success](const NodeRef& node) {
    if (node.get() == var) {
      success = true;
      return;
    }
  });
  return success;
}

inline bool IsConstDomain(Expr min, Expr extent) {
  return is_const(min) && is_const(extent);
}

class PartitionFinder : public IRVisitor {
 public:
  explicit PartitionFinder(VarExpr loop_var,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    const std::unordered_set<const Variable*>& variables)
    : loop_var_(loop_var), dom_map_(dom_map), variables_(variables) {}

  void Visit_(const For* op) {
    if (IsConstDomain(op->min, op->extent)) {
      dom_map_.insert({op->loop_var.get(),
          IntSet::interval(op->min, op->min + op->extent - 1)});
      IRVisitor::Visit_(op);
      dom_map_.erase(op->loop_var.get());
    } else {
      variables_.insert(op->loop_var.get());
      IRVisitor::Visit_(op);
      variables_.erase(op->loop_var.get());
    }
  }

  void Visit_(const IfThenElse* op) {
    if (ExprUseVar(op->condition, loop_var_.get())) {
      for (auto var : variables_) {
        if (ExprUseVar(op->condition, var)) IRVisitor::Visit_(op);
      }

      IntSet interval = DeduceBound(loop_var_, op->condition, dom_map_);
      if (interval.min().same_as(Interval::neg_inf)) {
        IntSet upper_bound = EvalSet(interval.max(), dom_map_);
        interval = IntSet::interval(interval.min(), upper_bound.min());
      } else if (interval.max().same_as(Interval::pos_inf)) {
        IntSet lower_bound = EvalSet(interval.min(), dom_map_);
        interval = IntSet::interval(lower_bound.max(), interval.max());
      } else {
        // Assume the partition is always a infinite set
        LOG(WARNING) << "interval wrong";
      }
      partitions.push_back(Partition{op->condition, op->condition, const_true(), interval});
    }
    IRVisitor::Visit_(op);
  }

  std::vector<Partition> partitions;
 private:
  VarExpr loop_var_;
  std::unordered_map<const Variable*, IntSet> dom_map_;
  std::unordered_set<const Variable*> variables_;
};

class PartitionReplacer : public IRMutator {
 public:
  PartitionReplacer(const Partition& p)
    : p_(p) {}

  Expr Mutate(Expr e) override {
    if (e.same_as(p_.old_expr)) {
      return Mutate(p_.true_value);
    }
    return IRMutator::Mutate(e);
  }

  Stmt Mutate(Stmt s) override { // ? will raise error if no this function
    return IRMutator::Mutate(s);
  }

 private:
  const Partition& p_;
};

// LoopPartitioner will try to partition the loop variable in the IR.
// The loop variable can be divided into two categories:
//
// - whose range is fixed, the min and the extent both are constant.
//
//   For now, we will not do partition on this kind loop variable, we
//   add them into dom_map in order to do deduce for follow-up
//   partitions.
//
// - whose range is variable
//
//   We will try to do partition on this kind loop variable. If success,
//   we will mutate the stmt then return. (only consider the partition
//   on the outmost loop yet). If failed, we will mark them as variable
//   (add them into variables_), then in the follow-up procedure, we know
//   a condition is not able to be deduced if it use this variable.

class LoopPartitioner : public IRMutator {
 public:
  explicit LoopPartitioner() {}
  Expr Mutate(Expr e) override {
    return IRMutator::Mutate(e);
  }
  Stmt Mutate(Stmt s) override {
    return IRMutator::Mutate(s);
  }

  Stmt Mutate_(const For* op, const Stmt& stmt) {
    if (IsConstDomain(op->min, op->extent)) {
      // if the range of loop_var is constant, we will not partition it,
      // instead, we will use the fixed domain to deduce.
      dom_map_.insert({op->loop_var.get(),
          IntSet::interval(op->min, op->min + op->extent - 1)});
      Stmt res = IRMutator::Mutate_(op, stmt);
      dom_map_.erase(op->loop_var.get());
      return res;
    }

    PartitionFinder finder(op->loop_var, dom_map_, variables_);
    finder.Visit(op->body);

    if (finder.partitions.empty()) {
      variables_.insert(op->loop_var.get());
      IRMutator::Mutate_(op, stmt);
      variables_.erase(op->loop_var.get());
      return stmt;
    }

    IntSet universe = IntSet::interval(op->min, op->min + op->extent - 1);
    std::vector<IntSet> sets{universe};
    // merge partitions (take their intersect)
    for (auto p : finder.partitions) {
      sets.push_back(p.interval);
    }
    IntSet true_itrv  = Intersect(sets);

    Stmt simplified_body = op->body;
    for (auto p : finder.partitions) {
      p.interval = true_itrv;
      simplified_body = PartitionReplacer(p).Mutate(simplified_body);
    }

    Stmt simplified_stmt = For::make(op->loop_var, true_itrv.min(),
      true_itrv.max() - true_itrv.min() + 1, op->for_type, op->device_api, simplified_body);
    Stmt s = simplified_stmt;

    Expr pre_doubt_cond = (true_itrv.min() != universe.min());
    IntSet pre_doubt_itrv = IntSet::interval(universe.min(), true_itrv.min());
    Stmt pre_stmt = For::make(op->loop_var, pre_doubt_itrv.min(),
      pre_doubt_itrv.max() - pre_doubt_itrv.min() + 1, op->for_type, op->device_api, op->body);
    s = Block::make(IfThenElse::make(pre_doubt_cond, pre_stmt), s);

    Expr post_doubt_cond = (true_itrv.max() != universe.max());
    IntSet post_doubt_itrv = IntSet::interval(true_itrv.max(), universe.max());
    Stmt post_stmt = For::make(op->loop_var, post_doubt_itrv.min(),
      post_doubt_itrv.max() - post_doubt_itrv.min() + 1, op->for_type, op->device_api, op->body);
    s = Block::make(s, IfThenElse::make(post_doubt_cond, post_stmt));
    return s;
  }

 private:
  std::unordered_map<const Variable*, IntSet> dom_map_;
  std::unordered_set<const Variable*> variables_;
};

Stmt LoopPartition(Stmt stmt) {
  stmt = LoopPartitioner().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
