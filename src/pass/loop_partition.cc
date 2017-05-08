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
  explicit PartitionFinder(VarExpr current_var,
    const std::unordered_map<const Variable*, IntSet>& dom_map)
      : current_var_(current_var), out_vars_(dom_map.size()), hint_map_(dom_map) {
        for (const auto& kv : dom_map) out_vars_.insert(kv.first);
      }

  void Visit_(const For* op) {
    if (ExprUseVars(op->min, out_vars_) || ExprUseVars(op->extent, out_vars_)) return;

    const Variable* var = op->loop_var.get();
    hint_map_.insert({var, IntSet::interval(op->min, op->min + op->extent - 1)});
    relax_map_.insert({var, IntSet::interval(op->min, op->min + op->extent - 1)});
    IRVisitor::Visit_(op);
    relax_map_.erase(var);
    hint_map_.erase(var);
  }

  void Visit_(const AttrStmt* op) {
    // handle thread_axis
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode* thread_axis = op->node.as<IterVarNode>();
      CHECK(thread_axis);
      const Variable* var = thread_axis->var.get();
      IntSet dom = IntSet::range(Range(make_zero(op->value.type()), op->value));
      hint_map_.insert({var, dom});
      relax_map_.insert({var, dom});
      IRVisitor::Visit_(op);
      relax_map_.erase(var);
      hint_map_.erase(var);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Call* op) {
    if (op->is_intrinsic(Call::likely)) {
      Expr cond = op->args[0];
      if (ExprUseVars(cond,
          std::unordered_set<const Variable*>({current_var_.get()}))) {
        IntSet interval =
          DeduceBound(current_var_, cond, hint_map_, relax_map_);
        partitions[cond.get()] = Partition{cond, interval};
      }
    } else {
      IRVisitor::Visit_(op);
    }
  }

  std::unordered_map<const Node*, Partition> partitions;

 private:
  VarExpr current_var_;
  std::unordered_set<const Variable*> out_vars_;
  std::unordered_map<const Variable*, IntSet> hint_map_;
  std::unordered_map<const Variable*, IntSet> relax_map_;
};

class PartitionReplacer : public IRMutator {
 public:
  explicit PartitionReplacer(const std::unordered_map<const Node*, Partition>& ps)
    : ps_(ps) {}
  using IRMutator::Mutate;

  Expr Mutate(Expr e) override {
    if (ps_.count(e.get())) {
      return Mutate(const_true());
    }
    return IRMutator::Mutate(e);
  }

 private:
  const std::unordered_map<const Node*, Partition>& ps_;
};

class LoopPartitioner : public IRMutator {
 public:
  LoopPartitioner() {}

  Stmt Mutate_(const For* op, const Stmt& stmt) {
    if (!is_const(op->min) || !is_const(op->extent)) {
      Stmt s = TryPartition(op, op->loop_var, op->min,
          op->min + op->extent - 1, op->body);
      if (s.defined()) return s;
    }
    dom_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    Stmt res = IRMutator::Mutate_(op, stmt);
    dom_map_.erase(op->loop_var.get());
    return res;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& stmt) {
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Var var = iv->var;
      if ((var.get()->name_hint.compare(0, 9, "blockIdx.") == 0) &&
          !is_const(op->value)) {
        Stmt s = TryPartition(op, var, 0, op->value - 1, op->body);
        if (s.defined()) return s;
      }

      dom_map_.insert({var.get(),
        IntSet::interval(make_zero(var.type()), op->value)});
      Stmt res = IRMutator::Mutate_(op, stmt);
      dom_map_.erase(var.get());
      return res;
    } else {
      return IRMutator::Mutate_(op, stmt);
    }
  }

 private:
  Stmt TryPartition(const Node* op, VarExpr var, Expr min, Expr max, Stmt body);
  Stmt MakeStmt(const Node* op, Expr extent, Stmt body);

  std::unordered_map<const Variable*, IntSet> dom_map_;
};

Stmt LoopPartitioner::TryPartition(const Node* op,
    VarExpr var, Expr min, Expr max, Stmt body) {
  PartitionFinder finder(var, dom_map_);
  finder.Visit(body);
  const auto& partitions = finder.partitions;
  if (partitions.empty()) return Stmt();

  Array<IntSet> sets;
  // merge partitions (take their intersect)
  for (const auto& kv : partitions) sets.push_back(kv.second.interval);
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
      Stmt new_body = Substitute(body, {{Var{var}, var + min}});
      pre_stmt = MakeStmt(op, body_begin - min, new_body);
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
      Stmt new_body = Substitute(body, {{Var{var}, var + post_doubt_begin}});
      post_stmt = MakeStmt(op, max - post_doubt_begin + 1, new_body);
    }
  } else {
    post_doubt_begin = max + 1;
  }

  // [body_begin, post_doubt_begin)
  Stmt simplified_body = PartitionReplacer(partitions).Mutate(body);
  Stmt new_body = Substitute(simplified_body, {{Var{var}, var + body_begin}});
  Stmt simplified_stmt = MakeStmt(op, post_doubt_begin - body_begin, new_body);
  Stmt s = simplified_stmt;
  if (pre_stmt.defined())  s = Block::make(pre_stmt, s);
  if (post_stmt.defined()) s = Block::make(s, post_stmt);

  return ConvertSSA(s);
}

Stmt LoopPartitioner::MakeStmt(const Node *node, Expr extent, Stmt body) {
  if (node->is_type<For>()) {
    const For *for_node = static_cast<const For*>(node);
    return For::make(for_node->loop_var, 0, extent,
      for_node->for_type, for_node->device_api, body);
  } else if (node->is_type<AttrStmt>()) {
    const AttrStmt *attr_stmt = static_cast<const AttrStmt*>(node);
    return AttrStmt::make(attr_stmt->node, attr_stmt->attr_key, extent, body);
  } else {
    LOG(FATAL) << "wrong type when try to make statement";
    return Stmt();
  }
}

class RemoveLikelyTags : public IRMutator {
 public:
  using IRMutator::Mutate;

  Expr Mutate_(const Call *op, const Expr& e) {
    if (op->is_intrinsic(Call::likely)) {
      CHECK_EQ(op->args.size(), 1);
      return IRMutator::Mutate(op->args[0]);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }
};

Stmt LoopPartition(Stmt stmt) {
  stmt = LoopPartitioner().Mutate(stmt);
  stmt = RemoveLikelyTags().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
