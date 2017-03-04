/*!
 *  Copyright (c) 2017 by Contributors
 * \file loop_partition.cc
 */
#include <tvm/ir.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/arithmetic.h>
#include <unordered_map>
#include <unordered_set>
#include "../arithmetic/int_set_internal.h"

namespace tvm {
namespace ir {

using arith::IntSet;
using arith::DeduceBound;
using arith::Intersect;

// a partition means the expr is equal to true in the interval
struct Partition {
  Expr expr;
  IntSet interval;
};

bool ExprUseVars(Expr expr, const std::unordered_set<const Variable*>& vars) {
  bool success = false;
  PostOrderVisit(expr, [&vars, &success](const NodeRef& node) {
    if (const Variable* v = node.as<Variable>()) {
      if (vars.count(v)) {
        success = true;
        return;
      }
    }
  });
  return success;
}

class PartitionFinder : public IRVisitor {
 public:
  explicit PartitionFinder(VarExpr loop_var,
    const std::unordered_map<const Variable*, IntSet>& dom_map)
      : target_var_(loop_var), out_vars_(dom_map.size()), hint_map_(dom_map) {
        for (const auto& kv : dom_map) out_vars_.insert(kv.first);
      }

  void Visit_(const For* op) {
    if (ExprUseVars(op->min, out_vars_) || ExprUseVars(op->extent, out_vars_)) return;

    hint_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    relax_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    IRVisitor::Visit_(op);
    relax_map_.erase(op->loop_var.get());
    hint_map_.erase(op->loop_var.get());
  }

  void Visit_(const IfThenElse* op) {
    if (ExprUseVars(op->condition, std::unordered_set<const Variable*>({target_var_.get()}))) {
      IntSet interval = DeduceBound(target_var_, op->condition, hint_map_, relax_map_);
      partitions[op->condition.get()] = Partition{op->condition, interval};
    } else {
      IRVisitor::Visit_(op);
    }
  }

  std::unordered_map<const Node*, Partition> partitions;

 private:
  VarExpr target_var_;
  std::unordered_set<const Variable*> out_vars_;
  std::unordered_map<const Variable*, IntSet> hint_map_;
  std::unordered_map<const Variable*, IntSet> relax_map_;
};

class PartitionReplacer : public IRMutator {
 public:
  explicit PartitionReplacer(const std::unordered_map<const Node*, Partition>& ps)
    : ps_(ps) {}

  Expr Mutate(Expr e) override {
    if (ps_.count(e.get())) {
      return Mutate(const_true());
    }
    return IRMutator::Mutate(e);
  }
  using IRMutator::Mutate;

 private:
  const std::unordered_map<const Node*, Partition>& ps_;
};

class LoopPartitioner : public IRMutator {
 public:
  LoopPartitioner() {}

  Stmt Mutate_(const For* op, const Stmt& stmt) {
    if (!is_const(op->min) || !is_const(op->extent)) {
      Stmt s = DoPartition(op, stmt);
      if (s.defined()) return s;
    }
    dom_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    Stmt res = IRMutator::Mutate_(op, stmt);
    dom_map_.erase(op->loop_var.get());
    return res;
  }

 private:
  Stmt DoPartition(const For* op, const Stmt& stmt);

  std::unordered_map<const Variable*, IntSet> dom_map_;
};

Stmt LoopPartitioner::DoPartition(const For* op, const Stmt& stmt) {
  PartitionFinder finder(op->loop_var, dom_map_);
  finder.Visit(op->body);
  const auto& partitions = finder.partitions;

  if (partitions.empty()) return Stmt();

  Expr min = op->min;
  Expr max = op->min + op->extent - 1;
  Array<IntSet> sets;
  // merge partitions (take their intersect)
  for (const auto& kv : partitions) {
    sets.push_back(kv.second.interval);
  }
  IntSet true_itrv  = Intersect(sets);

  Stmt pre_stmt;
  Expr body_begin;
  if (true_itrv.as<arith::IntervalSet>()->i.has_lower_bound()) {
    body_begin = true_itrv.min();
    if (!can_prove(body_begin == min)) {
      if (!can_prove(body_begin - min >= 0)) {
        LOG(WARNING) << "cannot prove: " << (body_begin - min >= 0)
                     << ", when generating the pre doubt loop";
        body_begin = Max::make(body_begin, min);
      }
      // [min, body_begin)
      Stmt body = Substitute(op->body,
        {{Var{op->loop_var}, op->loop_var + min}});
      pre_stmt = For::make(op->loop_var, 0,
        body_begin - min, op->for_type, op->device_api, body);
    }
  } else {
    body_begin = min;
  }

  Stmt post_stmt;
  Expr post_doubt_begin;
  if (true_itrv.as<arith::IntervalSet>()->i.has_upper_bound()) {
    post_doubt_begin = true_itrv.max() + 1;
    if (!can_prove(true_itrv.max() == max)) {
      if (!can_prove(max - post_doubt_begin >= 0)) {
        LOG(WARNING) << "Cannot prove: " << (max - post_doubt_begin >= 0)
                     << ", when generating the post doubt loop";
        post_doubt_begin = Min::make(post_doubt_begin, max);
      }
      // [post_doubt_begin, max]
      Stmt body = Substitute(op->body,
        {{Var{op->loop_var}, op->loop_var + post_doubt_begin}});
      post_stmt = For::make(op->loop_var, 0,
        max - post_doubt_begin + 1, op->for_type, op->device_api, body);
    }
  } else {
    post_doubt_begin = max + 1;
  }

  // [body_begin, post_doubt_begin)
  Stmt simplified_body = PartitionReplacer(partitions).Mutate(op->body);
  Stmt body = Substitute(simplified_body, {{Var{op->loop_var}, op->loop_var + body_begin}});
  Stmt simplified_stmt = For::make(op->loop_var, 0,
    post_doubt_begin - body_begin, op->for_type, op->device_api, body);
  Stmt s = simplified_stmt;
  if (pre_stmt.defined()) {
    s = Block::make(pre_stmt, s);
  }
  if (post_stmt.defined()) {
    s = Block::make(s, post_stmt);
  }

  return Simplify(ConvertSSA(s));
}

Stmt LoopPartition(Stmt stmt) {
  stmt = LoopPartitioner().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
