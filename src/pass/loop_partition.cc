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
#include "../runtime/thread_storage_scope.h"

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

// Select potential candidate IRs that can be partitioned.
// Rule:
//   - the range should not be const
//   - there exist a condition expression in the scope that use the var
class CandidateSelector : public IRVisitor {
 public:
  using VarIsUsed = bool;
  CandidateSelector() {}

  void Visit_(const For* op) {
    if (!is_const(op->min) || !is_const(op->extent)) {
      const Variable* var = op->loop_var.get();
      record_.insert({var, false});
      IRVisitor::Visit_(op);
      if (record_.at(var)) {
        candidates.insert(op);
      }
      record_.erase(var);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const AttrStmt* op) {
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::make(iv->thread_tag);
      if ((scope.rank == 0) && !is_const(op->value)) {
        record_.insert({var.get(), false});
        IRVisitor::Visit_(op);
        if (record_.at(var.get())) {
          candidates.insert(op);
        }
        record_.erase(var.get());
        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Call* op) {
    if (op->is_intrinsic(Call::likely)) {
      Expr cond = op->args[0];
      PostOrderVisit(cond, [&](const NodeRef& node) {
        const Variable* var = node.as<Variable>();
        if (var && record_.count(var)) record_.at(var) = true;
      });
    }
    IRVisitor::Visit_(op);
  }

  std::unordered_set<const Node*> candidates;

 private:
  std::unordered_map<const Variable*, VarIsUsed> record_;
};

// Find valid partition for specific variable
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

// Eliminate the condition expressions by partitions
class ConditionEliminator : public IRMutator {
 public:
  explicit ConditionEliminator(const std::unordered_map<const Node*, Partition>& ps)
    : ps_(ps) {}

  using IRMutator::Mutate;
  Expr Mutate(Expr e) final {
    if (ps_.count(e.get())) return Mutate(const_true());
    return IRMutator::Mutate(e);
  }

 private:
  const std::unordered_map<const Node*, Partition>& ps_;
};


// Insert the partition branch at the innermost thread scope
class ThreadPartitionInserter : public IRMutator {
 public:
  explicit ThreadPartitionInserter(const std::unordered_map<const Node*, Partition>& ps,
    Expr cond) : ps_(ps), cond_(cond), innermost_thread_scope_(false) {}

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == attr::thread_extent) {
      innermost_thread_scope_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      // add branch code inside the innermost thread scope
      if (innermost_thread_scope_) {
        Stmt simplified_body = ConditionEliminator(ps_).Mutate(op->body);
        Stmt body = IfThenElse::make(cond_, simplified_body, op->body);
        Expr value = this->Mutate(op->value);
        stmt = AttrStmt::make(op->node, op->attr_key, value, body);
      }
      innermost_thread_scope_ = false;
      return stmt;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

 private:
  const std::unordered_map<const Node*, Partition>& ps_;
  Expr cond_;
  bool innermost_thread_scope_;
};

// Try to do partition at the candidate IRs
class LoopPartitioner : public IRMutator {
 public:
  explicit LoopPartitioner(std::unordered_set<const Node*> candidates)
    : candidates_(candidates) {}

  Stmt Mutate_(const For* op, const Stmt& stmt) {
    if (candidates_.count(op)) {
      Stmt s = TryPartition(op, stmt, op->loop_var,
          op->min, op->min + op->extent - 1, op->body);
      if (s.defined()) return s;
    }
    dom_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    Stmt res = IRMutator::Mutate_(op, stmt);
    dom_map_.erase(op->loop_var.get());
    return res;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& stmt) {
    if (candidates_.count(op)) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Var var = iv->var;
      Stmt s = TryPartition(op, stmt, var, 0, op->value - 1, op->body);
      if (s.defined()) return s;

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
  Stmt TryPartition(const Node* op, const Stmt& stmt, VarExpr var,
      Expr min, Expr max, Stmt body);
  inline Stmt MakeStmt(const Node* op, Expr extent, Stmt body);

  std::unordered_set<const Node*> candidates_;
  std::unordered_map<const Variable*, IntSet> dom_map_;
};

Stmt LoopPartitioner::TryPartition(const Node* node, const Stmt& stmt,
    VarExpr var, Expr min, Expr max, Stmt body) {
  bool inside_thread_scope = node->is_type<AttrStmt>();
  PartitionFinder finder(var, dom_map_);
  finder.Visit(body);
  const auto& partitions = finder.partitions;
  if (partitions.empty()) return Stmt();

  Array<IntSet> sets;
  // merge partitions (take their intersect)
  for (const auto& kv : partitions) sets.push_back(kv.second.interval);
  IntSet true_itrv  = Intersect(sets);

  Expr body_begin;
  Stmt pre_stmt;
  if (true_itrv.as<arith::IntervalSet>()->i.has_lower_bound()) {
    body_begin = true_itrv.min();
    if (!can_prove(body_begin == min)) {
      Expr cond = (body_begin - min >= 0);
      if (!can_prove(cond)) {
        LOG(WARNING) << "Cannot prove: " << cond
                     << ", when generating the pre doubt loop";
        body_begin = Max::make(body_begin, min);
      }
      // [min, body_begin)
      if (!inside_thread_scope) {
        Stmt new_body = Substitute(body, {{Var{var}, var + min}});
        pre_stmt = MakeStmt(node, body_begin - min, new_body);
      }
    }
  } else {
    body_begin = min;
  }

  Expr post_doubt_begin;
  Stmt post_stmt;
  if (true_itrv.as<arith::IntervalSet>()->i.has_upper_bound()) {
    post_doubt_begin = true_itrv.max() + 1;
    if (!can_prove(true_itrv.max() == max)) {
      Expr cond = (max - post_doubt_begin >= 0);
      if (!can_prove(cond)) {
        LOG(WARNING) << "Cannot prove: " << cond
                     << ", when generating the post doubt loop";
        post_doubt_begin = Min::make(post_doubt_begin, max);
      }
      // [post_doubt_begin, max]
      if (!inside_thread_scope) {
        Stmt new_body = Substitute(body, {{Var{var}, var + post_doubt_begin}});
        post_stmt = MakeStmt(node, max - post_doubt_begin + 1, new_body);
      }
    }
  } else {
    post_doubt_begin = max + 1;
  }

  // [body_begin, post_doubt_begin)
  Stmt s;
  if (!inside_thread_scope) {
    Stmt simplified_body = ConditionEliminator(partitions).Mutate(body);
    Stmt new_body = Substitute(simplified_body, {{Var{var}, var + body_begin}});
    s = MakeStmt(node, post_doubt_begin - body_begin, new_body);
    if (pre_stmt.defined())  s = Block::make(pre_stmt, s);
    if (post_stmt.defined()) s = Block::make(s, post_stmt);
  } else {
    Expr cond = const_true();
    if (!can_prove(body_begin == min)) cond = cond && (var >= body_begin);
    if (!can_prove(post_doubt_begin == (max + 1))) cond = cond && (var < post_doubt_begin);
    s = ThreadPartitionInserter(partitions, cond).Mutate(stmt);
  }
  s = ConvertSSA(s);
  return s;
}

inline Stmt LoopPartitioner::MakeStmt(const Node *node, Expr extent, Stmt body) {
  const For *for_node = static_cast<const For*>(node);
  return For::make(for_node->loop_var, 0, extent,
    for_node->for_type, for_node->device_api, body);
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
  CandidateSelector selector;
  selector.Visit(stmt);
  stmt = LoopPartitioner(selector.candidates).Mutate(stmt);
  stmt = RemoveLikelyTags().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
