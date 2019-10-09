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
#include "../arithmetic/int_set.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

using arith::IntSet;
using arith::DeduceBound;
using arith::Intersect;

using PartitionKey = std::pair<const Node*, bool>;
struct PartitionKeyHash {
  std::size_t operator()(PartitionKey const& k) const noexcept {
    std::size_t h1 = std::hash<const Node*>{}(k.first);
    std::size_t h2 = std::hash<bool>{}(k.second);
    return h1 ^ h2;
  }
};

// Each mapping (cond, cond_value) -> interval represents the fact that
// condition cond is proven to have value cond_value (true or false) in interval.
using Partition = std::unordered_map<PartitionKey, IntSet, PartitionKeyHash>;


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
class CandidateSelector final : public IRVisitor {
 public:
  using VarIsUsed = bool;
  explicit CandidateSelector(bool split_const_loop)
      : split_const_loop_(split_const_loop) {}

  void Visit_(const For* op) {
    // partition const loop when sets split_const_loop_
    if (!is_const(op->min) || !is_const(op->extent) || split_const_loop_) {
      const Variable* var = op->loop_var.get();
      record_.insert({var, false});
      IRVisitor::Visit_(op);
      if (record_.at(var) && !no_split_) {
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
      if ((scope.rank == 0) && (!is_const(op->value) || split_const_loop_)) {
        record_.insert({var.get(), false});
        IRVisitor::Visit_(op);
        if (record_.at(var.get()) && !no_split_) {
          candidates.insert(op);
        }
        record_.erase(var.get());
        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Block* op) {
    bool temp = no_split_;
    this->Visit(op->first);
    // erase the no split state of first when visit rest.
    std::swap(temp, no_split_);
    this->Visit(op->rest);
    // restore the no split flag.
    no_split_ = no_split_ || temp;
  }

  void Visit_(const Call* op) {
    if (op->is_intrinsic(Call::likely)) {
      in_likely_ = true;
      IRVisitor::Visit_(op);
      in_likely_ = false;
    } else if (op->is_intrinsic(intrinsic::tvm_thread_allreduce)) {
      // no split if the body contains allreduce.
      no_split_ = true;
      return;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Variable* op) {
    if (in_likely_ && record_.count(op)) {
      record_.at(op) = true;
    }
  }

  std::unordered_set<const Node*> candidates;

 private:
  bool in_likely_{false};
  bool no_split_{false};
  bool split_const_loop_{false};
  std::unordered_map<const Variable*, VarIsUsed> record_;
};

// Populate partitions data structure, i.e., for a specific variable,
// find an interval in which each condition
// (currently, "likely" conditions) has fixed true or false value
class PartitionFinder : public IRVisitor {
 public:
  explicit PartitionFinder(VarExpr current_var,
    const std::unordered_map<const Variable*, IntSet>& hint_map,
    const std::unordered_map<const Variable*, IntSet>& relax_map)
      : current_var_(current_var), hint_map_(hint_map),  relax_map_(relax_map) {
        for (const auto& kv : hint_map) {
          out_vars_.insert(kv.first);
        }
        for (const auto& kv : relax_map) {
          out_vars_.insert(kv.first);
        }
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
        // For cond, find out the interval, if exists, in which we can prove that cond is
        // true. Also find the interval, if exists, in which we can prove that cond is
        // false.
        IntSet interval =
                DeduceBound(current_var_, cond, hint_map_, relax_map_);
        if (!interval.is_nothing()) {
          // cond is true within interval
          partitions[{cond.get(), true}] = interval;
        }
        Expr inverse_cond = InverseCond(cond);
        if (inverse_cond.defined()) {
          IntSet interval =
                  DeduceBound(current_var_, inverse_cond, hint_map_, relax_map_);
          if (!interval.is_nothing()) {
            // cond is false within interval
            partitions[{cond.get(), false}] = interval;
          }
        }
      }
    } else {
      IRVisitor::Visit_(op);
    }
  }

  Partition partitions;

 private:
  Expr InverseCond(const Expr& cond) {
    Expr inverse_cond;
    if (const LT* op = cond.as<LT>()) {
      // a < b -> a >= b
      inverse_cond = GE::make(op->a, op->b);
    } else if (const GT* op = cond.as<GT>()) {
      // a > b -> a <= b
      inverse_cond = LE::make(op->a, op->b);
    } else if (const LE* op = cond.as<LE>()) {
      // a <= b -> a > b
      inverse_cond = GT::make(op->a, op->b);
    } else if (const GE* op = cond.as<GE>()) {
      // a >= b -> a < b
      inverse_cond = LT::make(op->a, op->b);
    } else if (const EQ* op = cond.as<EQ>()) {
      // a == b -> a != b
      inverse_cond = NE::make(op->a, op->b);
      // a != b -> a == b
    } else if (const NE* op = cond.as<NE>()) {
      inverse_cond = EQ::make(op->a, op->b);
    }
    return inverse_cond;
  }

  VarExpr current_var_;
  std::unordered_set<const Variable*> out_vars_;
  std::unordered_map<const Variable*, IntSet> hint_map_;
  std::unordered_map<const Variable*, IntSet> relax_map_;
};

// Replace the set of conditions given by ps with cond_value (true or false)
class ConditionEliminator : public IRMutator {
 public:
  explicit ConditionEliminator(const std::unordered_set<const Node*>& ps, bool cond_value = true)
    : ps_(ps), cond_value_(cond_value) {}

  using IRMutator::Mutate;
  Expr Mutate(Expr e) final {
    if (ps_.find(e.get()) != ps_.end()) {
      return Mutate(cond_value_ ? const_true() : const_false());
    }
    return IRMutator::Mutate(e);
  }

 private:
  std::unordered_set<const Node*> ps_;
  bool cond_value_;
};


// Insert the partition branch at the innermost thread scope
class ThreadPartitionInserter : public IRMutator {
 public:
  explicit ThreadPartitionInserter(const std::unordered_set<const Node*>& ps,
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
  const std::unordered_set<const Node*>& ps_;
  Expr cond_;
  bool innermost_thread_scope_;
};

// Try to partition range of iteration variables in order to remove (some)
// likely conditions
class LoopPartitioner : public IRMutator {
 public:
  explicit LoopPartitioner(bool split_const_loop)
      : selector(CandidateSelector(split_const_loop)) {}

  Stmt VisitAndMutate(const Stmt& stmt) {
    selector.Visit(stmt);
    return Mutate(stmt);
  }

  Stmt Mutate_(const For* op, const Stmt& stmt) {
    if (selector.candidates.count(op)) {
      Stmt s = TryPartition(op, stmt, op->loop_var,
          op->min, op->min + op->extent - 1, op->body, false);
      if (s.defined()) return s;
    }

    // normal path when loop partition fails
    // normal loop variable can be put into hint map.
    hint_map_.insert({op->loop_var.get(),
      IntSet::interval(op->min, op->min + op->extent - 1)});
    Stmt res = IRMutator::Mutate_(op, stmt);
    hint_map_.erase(op->loop_var.get());
    return res;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& stmt) {
    if (op->attr_key != attr::thread_extent) {
      return IRMutator::Mutate_(op, stmt);
    }

    const IterVarNode *iv = op->node.as<IterVarNode>();
    CHECK(iv);
    Var var = iv->var;
    if (selector.candidates.count(op)) {
      Stmt s = TryPartition(op, stmt, var, 0, op->value - 1, op->body, true);
      if (s.defined()) return s;
    }

    // normal path when loop parittion fails.
    runtime::ThreadScope scope = runtime::ThreadScope::make(iv->thread_tag);
    Stmt res;
    if (scope.rank == 1) {
      // threadIdx should be put into relax map, in case of divergence.
      relax_map_.insert({var.get(),
        IntSet::interval(make_zero(var.type()), op->value - 1)});
      res = IRMutator::Mutate_(op, stmt);
      relax_map_.erase(var.get());
    } else {
      hint_map_.insert({var.get(),
        IntSet::interval(make_zero(var.type()), op->value - 1)});
      res = IRMutator::Mutate_(op, stmt);
      hint_map_.erase(var.get());
    }
    return res;
  }

 private:
  Stmt TryPartition(const Node* op, const Stmt& stmt, VarExpr var,
      Expr min, Expr max, Stmt body, bool partition_thread_scope);

  std::pair<IntSet, std::unordered_set<const Node*>>
  GetIntervalAndCondset(const Partition &partitions,
                        const arith::IntervalSet &for_interval,
                        bool cond_value);

  inline Stmt MakeFor(const Node* op, Expr extent, Stmt body);

  /* Candidate IRs that may be partitioned potentially */
  std::unordered_map<const Variable*, IntSet> hint_map_;
  std::unordered_map<const Variable*, IntSet> relax_map_;
  arith::Analyzer analyzer_;
  CandidateSelector selector;
};

// Returns an interval (in the first component) in which all the conditions
// given in the second component provably have value given by cond_value
std::pair<IntSet, std::unordered_set<const Node*>>
LoopPartitioner::GetIntervalAndCondset(const Partition &partitions,
                                       const arith::IntervalSet &for_interval,
                                       bool cond_value) {
  Array<IntSet> sets;
  std::unordered_set<const Node*> cond_set;

  for (const auto &kv : partitions) {
    if (kv.first.second == cond_value) {
      arith::IntervalSet interval = Downcast<arith::IntervalSet>(kv.second);
      arith::IntervalSet intersection = arith::Intersect(
          &analyzer_, interval, for_interval);
      if (!intersection->IsEmpty()) {
        sets.push_back(kv.second);
        cond_set.insert(kv.first.first);
      }
    }
  }
  IntSet interval = sets.empty() ? IntSet::nothing() : Intersect(sets);
  return std::make_pair(interval, cond_set);
}

Stmt AppendStmts(const Stmt& a, const Stmt& b) {
  if (!a.defined()) {
    return b;
  } else if (!b.defined()) {
    return a;
  } else {
    return Block::make(a, b);
  }
}

/*
 * Tries to recursively partition the range of the variable (given by var) of
 * the for loop (given by node and stmt) into a
 * number of disjoint ranges such that in some ranges one or more predicates
 * in the loopnest are provably true or false in each range. For example, given the
 * following loop to partition:
 * for (i = 0; i < 4; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(i*10 + j < 36))
 *            A[10*i+j] = B[10*i+j]
 *
 * We first partition range of i, i.e., [0,3] into subranges [0,2] and [3,3] because the
 * likely condition is always true for the first subrange but not always true for the
 * second subrange. Therefore, we'll have
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(1))
 *           A[10*i+j] = B[10*i+j]
 * for (i = 0; i < 1; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely((i+3)*10 + j < 36))
 *            A[10*(i+3)+j] = B[10*(i+3)+j]
 * Which is simplified as:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        A[10*i+j] = B[10*i+j]
 * for (j = 0; j < 10; j++) // loopnest 1
 *    if (likely(j < 6))
 *            A[30+j] = B[30+j]
 * Now, we recursively partition j in loopnest 1 into subranges [0,5] and [6,9] where the
 * condition is true for the first subrange and now always true for the second subrange.
 * for (j = 0; j < 6; j++)
 *    if (likely(1))
 *         A[30+j] = B[30+j]
 * for (j = 0; j < 4; j++) // loop 2
 *    if (likely(j < 0))
 *        A[36+j] = B[36+j]
 * Finally we recursively partition loop 2 above into subrange [0,3] where the
 * condition is false and empty interval where the condition is not false,
 * therefore we generate
 * for (j = 0; j < 4; j++)
 *    if (likely(0))
 *        A[36+j] = B[36+j]
 * which will eventually be simplified to empty code. And because only one loop was generated
 * from loop 2 we stop recursing.
 */
Stmt LoopPartitioner::TryPartition(const Node* node,
                                   const Stmt& stmt,
                                   VarExpr var,
                                   Expr min,
                                   Expr max,
                                   Stmt body,
                                   bool partition_thread_scope) {
  using namespace arith;
  // include hint of var.
  hint_map_.insert({var.get(), IntSet::interval(min, max)});

  PartitionFinder finder(var, hint_map_, relax_map_);
  finder.Visit(body);

  hint_map_.erase(var.get());
  if (finder.partitions.empty()) return Stmt();

  arith::IntervalSet for_interval(min, max);
  bool cond_value;
  IntSet middle_interval;
  std::unordered_set<const Node*> cond_set;
  // find an interval in which all conditions on var are true
  std::tie(middle_interval, cond_set) =
          GetIntervalAndCondset(finder.partitions, for_interval, true);
  if (middle_interval.is_nothing()) {
    // if such interval doesn't exist, find an interval in which all
    // conditions on var are false
    std::tie(middle_interval, cond_set) =
        GetIntervalAndCondset(finder.partitions, for_interval, false);
    if (middle_interval.is_nothing())
      // we couldn't find an interval in which the conditions are provably true or false
      // Therefore, we can't partition the loop based on those conds
      return Stmt();
    cond_value = false;
  } else {
    cond_value = true;
  }

  IntervalSet middle_interval_i = Downcast<IntervalSet>(middle_interval);
  // middle_interval is the subrange of the loop variable range for which a
  // set of conditions are true (or false resp.)
  // The part of the loop variable range that is before (after resp.) that
  // subrange is prefixed with pre- (post- resp.)

  // Calculating pre-subrange and generating code for it.
  // pre-subrange = [min, body_begin)
  Expr body_begin;
  Stmt pre_stmt;
  bool pre_stmt_recurse = true;
  if (middle_interval_i->HasLowerBound()) {
    body_begin = ir::Simplify(middle_interval.min());
    Expr cond = (body_begin - min >= 0);
    if (!analyzer_.CanProve(cond)) {
      LOG(WARNING) << "Cannot prove: " << cond
                   << ", when generating the pre doubt loop";
      body_begin = Max::make(body_begin, min);
      // stop recursing on this interval if we can't prove it has non-negative length
      pre_stmt_recurse = false;
    }
    if (!partition_thread_scope) {
      Stmt pre_body = Substitute(body, {{Var{var}, var + min}});
      pre_stmt = MakeFor(node, body_begin - min, pre_body);
    }
  } else {
    body_begin = min;
  }

  // Calculating post-subrange and generating code for it.
  // post-subrange = [post_doubt_begin, max+1)
  Expr post_doubt_begin;
  Stmt post_stmt;
  bool post_stmt_recurse = true;
  if (middle_interval_i->HasUpperBound()) {
    post_doubt_begin = ir::Simplify(middle_interval.max() + 1);
    // require the extent to be non-negative
    Expr cond = (max - post_doubt_begin + 1 >= 0);
    if (!analyzer_.CanProve(cond)) {
      LOG(WARNING) << "Cannot prove: " << cond
                   << ", when generating the post doubt loop";
      post_doubt_begin = Min::make(post_doubt_begin, max+1);
      // stop recursing on this interval if we can't prove it has non-negative length
      post_stmt_recurse = false;
    }
    if (!partition_thread_scope) {
      Stmt post_body =
        Substitute(body, {{Var{var}, var + post_doubt_begin}});
      post_stmt = MakeFor(node, max - post_doubt_begin + 1, post_body);
    }
  } else {
    post_doubt_begin = max + 1;
  }

  Stmt s;

  // Generating code for middle subrange
  if (!partition_thread_scope) {
    Stmt mid_stmt;
    if (!analyzer_.CanProve(body_begin >= post_doubt_begin)) {
      // [body_begin, post_doubt_begin)
      Stmt simplified_body = ConditionEliminator(cond_set, cond_value).Mutate(body);
      Stmt new_body = Substitute(simplified_body, {{Var{var}, var + body_begin}});
      mid_stmt = MakeFor(node, post_doubt_begin - body_begin, new_body);

      // Recurse for each non-empty subrange only if there are at least
      // two non-empty subranges
      if (pre_stmt.defined() || post_stmt.defined()) {
        mid_stmt = VisitAndMutate(mid_stmt);
        if (pre_stmt.defined() && pre_stmt_recurse) {
          pre_stmt = VisitAndMutate(pre_stmt);
        }
        if (post_stmt.defined() && post_stmt_recurse) {
          post_stmt = VisitAndMutate(post_stmt);
        }
      }
    }
    s = AppendStmts(pre_stmt, mid_stmt);
    s = AppendStmts(s, post_stmt);
  } else {
    Expr cond = const_true();
    if (!analyzer_.CanProve(body_begin == min)) cond = cond && (var >= body_begin);
    if (!analyzer_.CanProve(post_doubt_begin == (max + 1))) cond = cond && (var < post_doubt_begin);
    s = ThreadPartitionInserter(cond_set, cond).Mutate(stmt);
  }
  s = ConvertSSA(s);
  return s;
}

inline Stmt LoopPartitioner::MakeFor(const Node *node, Expr extent, Stmt body) {
  const For *for_node = static_cast<const For*>(node);
  CHECK(for_node);
  if (analyzer_.CanProve(extent == make_const(Int(32), 1))) {
    // If the loop extent is 1, do not create the loop anymore
    return Substitute(body, {{Var{for_node->loop_var}, make_const(Int(32), 0)}});
  } else {
    return For::make(for_node->loop_var, 0, extent,
                     for_node->for_type, for_node->device_api, body);
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

Stmt LoopPartition(Stmt stmt, bool split_const_loop) {
  stmt = LoopPartitioner(split_const_loop).VisitAndMutate(stmt);
  stmt = RemoveLikelyTags().Mutate(stmt);
  return stmt;
}

}  // namespace ir
}  // namespace tvm
