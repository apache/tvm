/*!
 *  Copyright (c) 2016 by Contributors
 * \file loop_partition.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
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

bool expr_use_var(Expr expr, Expr target) {
  bool success = false;
  PostOrderVisit(expr, [&target, &success](const NodeRef& node) {
    if (node.same_as(target)) {
      success = true;
      return;
    }
  });
  return success;
}

class PartitionFinder : public IRVisitor {
 public:
  explicit PartitionFinder(VarExpr loop_var)
    : loop_var_(loop_var) {}

  void Visit_(const For* op) {
    dom_map_[op->loop_var.get()] = IntSet::interval(op->min, op->min + op->extent - 1);
    IRVisitor::Visit_(op);
  }

  void Visit_(const IfThenElse* op) {
    if (expr_use_var(op->condition, loop_var_)) {
      IntSet interval = DeduceBound(loop_var_, op->condition, dom_map_);
      if (interval.min().same_as(Interval::neg_inf)) {
        IntSet upper_bound = EvalSet(interval.max(), dom_map_);
        interval = IntSet::interval(interval.min(), upper_bound.min());
      } else if (interval.max().same_as(Interval::pos_inf)) {
        IntSet lower_bound = EvalSet(interval.min(), dom_map_);
        interval = IntSet::interval(lower_bound.max(), interval.max());
      } else {
        // Assume the partition is always a infinite set
        LOG(WARNING) << "interval wrong?";
      }
      partitions.push_back(Partition{op->condition, op->condition, const_true(), interval});
    }
    IRVisitor::Visit_(op);
  }

  std::vector<Partition> partitions;
 private:
  VarExpr loop_var_;
  std::unordered_map<const Variable*, IntSet> dom_map_;
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

IntSet intersect(IntSet a, IntSet b) { // need move into IntSet
  // (TODO) temp solution
  return IntSet::interval(b.min(), a.max());
}

IntSet complement(IntSet s, IntSet u) { // need move into IntSet
  // (TODO) temp solution
  return IntSet::interval(s.max() + 1, u.max());
}

class LoopPartitioner : public IRMutator {
 public:
  explicit LoopPartitioner() {}
  Expr Mutate(Expr e) override {
    return IRMutator::Mutate(e);
  }
  Stmt Mutate(Stmt s) override {
    return IRMutator::Mutate(s);
  }

  Stmt Mutate_(const For* op, const Stmt& stmt) { // Simplify for this for loop
    // (TODO) recursive

    PartitionFinder finder(op->loop_var);
    finder.Visit(op->body);

    if (finder.partitions.empty()) {
      // no available partition, return directly
      return stmt;
    }

    IntSet universe = IntSet::interval(op->min, op->min + op->extent - 1);
    Stmt s;
    // (TODO) in fact, we need to consider all partitions, then split
    // the universe into multiple ranges
    for (auto p : finder.partitions) {
      IntSet true_itrv  = intersect(p.interval, universe);
      IntSet doubt_itrv = complement(true_itrv, universe);

      Stmt simplified_body = PartitionReplacer(p).Mutate(op->body);
      Stmt simplified_stmt = For::make(op->loop_var, true_itrv.min(),
        true_itrv.max() - true_itrv.min() + 1, op->for_type, op->device_api, simplified_body);
      Stmt remaining_stmt = For::make(op->loop_var, doubt_itrv.min(),
        doubt_itrv.max() - doubt_itrv.min() + 1, op->for_type, op->device_api, op->body);
      s = Block::make(simplified_stmt, remaining_stmt);
    }
    return s;
  }

 private:

};

Stmt LoopPartition(Stmt stmt) {
  stmt = LoopPartitioner().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
