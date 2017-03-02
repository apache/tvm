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

// a partition means the expr is equal to true in the interval
struct Partition {
  Expr expr;
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
    const std::unordered_map<const Variable*, IntSet>& vars)
    : target_var_(loop_var), out_vars_(vars), hint_map_(vars), relax_map_() {}

  void Visit_(const For* op) {
    for (auto kv : out_vars_) {
      if (ExprUseVar(op->min, kv.first) ||
          ExprUseVar(op->extent, kv.first)) {
        return;
      }
    }

    hint_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    relax_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    IRVisitor::Visit_(op);
    relax_map_.erase(op->loop_var.get());
    hint_map_.erase(op->loop_var.get());
  }

  void Visit_(const IfThenElse* op) {
    if (ExprUseVar(op->condition, target_var_.get())) {
      IntSet interval = DeduceBound(target_var_, op->condition, hint_map_, relax_map_);
      partitions.push_back(Partition{op->condition, interval});
    } else {
      IRVisitor::Visit_(op);
    }
  }

  std::vector<Partition> partitions;
 private:
  VarExpr target_var_;
  const std::unordered_map<const Variable*, IntSet>& out_vars_;
  std::unordered_map<const Variable*, IntSet> hint_map_;
  std::unordered_map<const Variable*, IntSet> relax_map_;
};

class PartitionReplacer : public IRMutator {
 public:
  PartitionReplacer(const std::vector<Partition>& ps)
    : ps_(ps) {}

  Expr Mutate(Expr e) final {
    for (auto p : ps_) {
      if (e.same_as(p.expr)) {
        return Mutate(const_true());
      }
    }
    return IRMutator::Mutate(e);
  }
  using IRMutator::Mutate;

 private:
  const std::vector<Partition>& ps_;
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

  Stmt Mutate_(const For* op, const Stmt& stmt) {
    if (IsConstDomain(op->min, op->extent)) {
      // if the range of loop_var is constant, we will not partition it,
      // instead, we will use the fixed domain to deduce.
      vars_.insert({op->loop_var.get(),
        IntSet::interval(op->min, op->min + op->extent - 1)});
      Stmt res = IRMutator::Mutate_(op, stmt);
      vars_.erase(op->loop_var.get());
      return res;
    }

    PartitionFinder finder(op->loop_var, vars_);
    finder.Visit(op->body);

    if (finder.partitions.empty()) {
      vars_.insert({op->loop_var.get(),
          IntSet::interval(op->min, op->min + op->extent - 1)});
      Stmt res = IRMutator::Mutate_(op, stmt);
      vars_.erase(op->loop_var.get());
      return res;
    }

    IntSet universe = IntSet::interval(op->min, op->min + op->extent - 1);
    std::vector<IntSet> sets{universe};
    // merge partitions (take their intersect)
    for (auto p : finder.partitions) {
      sets.push_back(p.interval);
    }
    IntSet true_itrv  = Intersect(sets);

    Stmt simplified_body = PartitionReplacer(finder.partitions).Mutate(op->body);
    Stmt simplified_stmt = For::make(op->loop_var, true_itrv.min(),
      true_itrv.max() - true_itrv.min() + 1, op->for_type, op->device_api, simplified_body);
    Stmt s = simplified_stmt;

    if (!can_prove(true_itrv.min() == universe.min())) {
      Expr pre_doubt_cond = (true_itrv.min() != universe.min());
      IntSet pre_doubt_itrv = IntSet::interval(universe.min(), true_itrv.min());
      Stmt pre_stmt = For::make(op->loop_var, pre_doubt_itrv.min(),
        pre_doubt_itrv.max() - pre_doubt_itrv.min() + 1, op->for_type, op->device_api, op->body);
      s = Block::make(IfThenElse::make(pre_doubt_cond, pre_stmt), s);
    }

    if (!can_prove(true_itrv.max() == universe.max())) {
      Expr post_doubt_cond = (true_itrv.max() != universe.max());
      IntSet post_doubt_itrv = IntSet::interval(true_itrv.max(), universe.max());
      Stmt post_stmt = For::make(op->loop_var, post_doubt_itrv.min(),
        post_doubt_itrv.max() - post_doubt_itrv.min() + 1, op->for_type, op->device_api, op->body);
      s = Block::make(s, IfThenElse::make(post_doubt_cond, post_stmt));
    }
    return s;
  }

 private:
  std::unordered_map<const Variable*, IntSet> vars_;
};

Stmt LoopPartition(Stmt stmt) {
  stmt = LoopPartitioner().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
