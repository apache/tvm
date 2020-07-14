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
 * \file loop_partition.cc
 */
#include <tvm/arith/analyzer.h>
#include <tvm/arith/bound.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_util.h"

namespace tvm {
namespace tir {

struct LoopPartitionConfigNode : public tvm::AttrsNode<LoopPartitionConfigNode> {
  bool partition_const_loop;

  TVM_DECLARE_ATTRS(LoopPartitionConfigNode, "tir.transform.LoopPartitionConfig") {
    TVM_ATTR_FIELD(partition_const_loop).describe("Split constant loop").set_default(false);
  }
};

class LoopPartitionConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopPartitionConfig, Attrs, LoopPartitionConfigNode);
};

TVM_REGISTER_NODE_TYPE(LoopPartitionConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.LoopPartition", LoopPartitionConfig);

using arith::DeduceBound;
using arith::Intersect;
using arith::IntSet;

using PartitionKey = std::pair<PrimExpr, bool>;
struct PartitionKeyHash {
  std::size_t operator()(PartitionKey const& k) const noexcept {
    std::size_t h1 = ObjectPtrHash{}(k.first);  // NOLINT(whitespace/braces)
    std::size_t h2 = std::hash<bool>{}(k.second);
    return h1 ^ h2;
  }
};

struct PartitionKeyEqual {
  bool operator()(const PartitionKey& k1, const PartitionKey& k2) const {
    // NOLINTNEXTLINE(whitespace/braces)
    return k1.second == k2.second && ObjectPtrEqual{}(k1.first, k2.first);
  }
};

// Each mapping (cond, cond_value) -> interval represents the fact that
// condition cond is proven to have value cond_value (true or false) in interval.
using Partition = std::unordered_map<PartitionKey, IntSet, PartitionKeyHash, PartitionKeyEqual>;

using ExpressionSet = std::unordered_set<PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

bool ExprUseVars(PrimExpr expr, const std::unordered_set<const VarNode*>& vars) {
  bool success = false;
  PostOrderVisit(expr, [&vars, &success](const ObjectRef& node) {
    if (const VarNode* v = node.as<VarNode>()) {
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
class CandidateSelector final : public StmtExprVisitor {
 public:
  using VarIsUsed = bool;
  explicit CandidateSelector(bool partition_const_loop)
      : partition_const_loop_(partition_const_loop) {}

  void VisitStmt_(const ForNode* op) final {
    // partition const loop when sets partition_const_loop_
    if (!is_const_int(op->min) || !is_const_int(op->extent) || partition_const_loop_) {
      const VarNode* var = op->loop_var.get();
      record_.insert({var, false});
      StmtExprVisitor::VisitStmt_(op);
      if (record_.at(var) && !no_split_) {
        candidates.insert(GetRef<Stmt>(op));
      }
      record_.erase(var);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      if ((scope.rank == 0) && (!is_const_int(op->value) || partition_const_loop_)) {
        record_.insert({var.get(), false});
        StmtExprVisitor::VisitStmt_(op);
        if (record_.at(var.get()) && !no_split_) {
          candidates.insert(GetRef<Stmt>(op));
        }
        record_.erase(var.get());
        return;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    bool init_no_split = no_split_;
    for (Stmt stmt : op->seq) {
      // erase the no split state of before visiting the next one.
      bool temp = init_no_split;
      std::swap(temp, no_split_);
      this->VisitStmt(stmt);
      // restore the no split flag.
      no_split_ = no_split_ || temp;
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      in_likely_ = true;
      StmtExprVisitor::VisitExpr_(op);
      in_likely_ = false;
    } else if (op->op.same_as(builtin::tvm_thread_allreduce())) {
      // no split if the body contains allreduce.
      no_split_ = true;
      return;
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    if (in_likely_ && record_.count(op)) {
      record_.at(op) = true;
    }
  }

  std::unordered_set<Stmt, ObjectPtrHash, ObjectPtrEqual> candidates;

 private:
  bool in_likely_{false};
  bool no_split_{false};
  bool partition_const_loop_{false};
  std::unordered_map<const VarNode*, VarIsUsed> record_;
};

// Populate partitions data structure, i.e., for a specific variable,
// find an interval in which each condition
// (currently, "likely" conditions) has fixed true or false value
class PartitionFinder : public StmtExprVisitor {
 public:
  explicit PartitionFinder(Var current_var,
                           const std::unordered_map<const VarNode*, IntSet>& hint_map,
                           const std::unordered_map<const VarNode*, IntSet>& relax_map)
      : current_var_(current_var), hint_map_(hint_map), relax_map_(relax_map) {
    for (const auto& kv : hint_map) {
      out_vars_.insert(kv.first);
    }
    for (const auto& kv : relax_map) {
      out_vars_.insert(kv.first);
    }
  }

  void VisitStmt_(const ForNode* op) final {
    if (ExprUseVars(op->min, out_vars_) || ExprUseVars(op->extent, out_vars_)) return;

    const VarNode* var = op->loop_var.get();
    hint_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    relax_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    StmtExprVisitor::VisitStmt_(op);
    relax_map_.erase(var);
    hint_map_.erase(var);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    // handle thread_axis
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode* thread_axis = op->node.as<IterVarNode>();
      CHECK(thread_axis);
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
    if (op->op.same_as(builtin::likely())) {
      PrimExpr cond = op->args[0];
      if (ExprUseVars(cond, std::unordered_set<const VarNode*>({current_var_.get()}))) {
        // For cond, find out the interval, if exists, in which we can prove that cond is
        // true. Also find the interval, if exists, in which we can prove that cond is
        // false.
        IntSet interval = DeduceBound(current_var_, cond, hint_map_, relax_map_);
        if (!interval.IsNothing()) {
          // cond is true within interval
          partitions[{cond, true}] = interval;
        }
        PrimExpr inverse_cond = InverseCond(cond);
        if (inverse_cond.defined()) {
          IntSet interval = DeduceBound(current_var_, inverse_cond, hint_map_, relax_map_);
          if (!interval.IsNothing()) {
            // cond is false within interval
            partitions[{cond, false}] = interval;
          }
        }
      }
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  Partition partitions;

 private:
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
  std::unordered_set<const VarNode*> out_vars_;
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
};

// Replace the set of conditions given by ps with cond_value (true or false)
class ConditionEliminator : public StmtExprMutator {
 public:
  explicit ConditionEliminator(const ExpressionSet& ps, bool cond_value = true)
      : ps_(ps), cond_value_(cond_value) {}

  PrimExpr VisitExpr(const PrimExpr& e) final {
    if (ps_.find(e) != ps_.end()) {
      return VisitExpr(cond_value_ ? const_true() : const_false());
    }
    return StmtExprMutator::VisitExpr(e);
  }

 private:
  ExpressionSet ps_;
  bool cond_value_;
};

// Insert the partition branch at the innermost thread scope
class ThreadPartitionInserter : public StmtMutator {
 public:
  explicit ThreadPartitionInserter(const ExpressionSet& ps, PrimExpr cond)
      : ps_(ps), cond_(cond), innermost_thread_scope_(false) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      innermost_thread_scope_ = true;
      Stmt stmt = StmtMutator::VisitStmt_(op);
      // add branch code inside the innermost thread scope
      if (innermost_thread_scope_) {
        Stmt simplified_body = ConditionEliminator(ps_)(op->body);
        Stmt body = IfThenElse(cond_, simplified_body, op->body);
        PrimExpr value = this->VisitExpr(op->value);
        stmt = AttrStmt(op->node, op->attr_key, value, body);
      }
      innermost_thread_scope_ = false;
      return stmt;
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

 private:
  const ExpressionSet& ps_;
  PrimExpr cond_;
  bool innermost_thread_scope_;
};

// Try to partition range of iteration variables in order to remove (some)
// likely conditions
class LoopPartitioner : public StmtMutator {
 public:
  explicit LoopPartitioner(bool partition_const_loop)
      : selector(CandidateSelector(partition_const_loop)) {}

  Stmt VisitAndMutate(Stmt stmt) {
    selector(stmt);
    return operator()(std::move(stmt));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    auto fs = GetRef<Stmt>(op);
    if (selector.candidates.count(fs)) {
      Stmt s = TryPartition(fs, op->loop_var, op->min, op->min + op->extent - 1, op->body, false);
      if (s.defined()) return s;
    }

    // normal path when loop partition fails
    // normal loop variable can be put into hint map.
    hint_map_.insert({op->loop_var.get(), IntSet::Interval(op->min, op->min + op->extent - 1)});
    Stmt res = StmtMutator::VisitStmt_(op);
    hint_map_.erase(op->loop_var.get());
    return res;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key != attr::thread_extent) {
      return StmtMutator::VisitStmt_(op);
    }

    const IterVarNode* iv = op->node.as<IterVarNode>();
    CHECK(iv);
    Var var = iv->var;
    auto as = GetRef<Stmt>(op);
    if (selector.candidates.count(as)) {
      Stmt s = TryPartition(as, var, 0, op->value - 1, op->body, true);
      if (s.defined()) return s;
    }

    // normal path when loop parittion fails.
    runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
    Stmt res;
    if (scope.rank == 1) {
      // threadIdx should be put into relax map, in case of divergence.
      relax_map_.insert({var.get(), IntSet::Interval(make_zero(var.dtype()), op->value - 1)});
      res = StmtMutator::VisitStmt_(op);
      relax_map_.erase(var.get());
    } else {
      hint_map_.insert({var.get(), IntSet::Interval(make_zero(var.dtype()), op->value - 1)});
      res = StmtMutator::VisitStmt_(op);
      hint_map_.erase(var.get());
    }
    return res;
  }

 private:
  Stmt TryPartition(const Stmt& stmt, Var var, PrimExpr min, PrimExpr max, Stmt body,
                    bool partition_thread_scope);

  std::pair<IntSet, ExpressionSet> GetIntervalAndCondset(const Partition& partitions,
                                                         const arith::IntervalSet& for_interval,
                                                         bool cond_value);

  inline Stmt MakeFor(const Object* op, PrimExpr extent, Stmt body);

  /* Candidate IRs that may be partitioned potentially */
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
  arith::Analyzer analyzer_;
  CandidateSelector selector;
};

// Returns an interval (in the first component) in which all the conditions
// given in the second component provably have value given by cond_value
std::pair<IntSet, ExpressionSet> LoopPartitioner::GetIntervalAndCondset(
    const Partition& partitions, const arith::IntervalSet& for_interval, bool cond_value) {
  Array<IntSet> sets;
  ExpressionSet cond_set;

  for (const auto& kv : partitions) {
    if (kv.first.second == cond_value) {
      arith::IntervalSet interval = Downcast<arith::IntervalSet>(kv.second);
      arith::IntervalSet intersection = arith::Intersect(&analyzer_, interval, for_interval);
      if (!intersection->IsEmpty()) {
        sets.push_back(kv.second);
        cond_set.insert(kv.first.first);
      }
    }
  }
  IntSet interval = sets.empty() ? IntSet::Nothing() : Intersect(sets);
  return std::make_pair(interval, cond_set);
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
Stmt LoopPartitioner::TryPartition(const Stmt& stmt, Var var, PrimExpr min, PrimExpr max, Stmt body,
                                   bool partition_thread_scope) {
  using namespace arith;
  // include hint of var.
  hint_map_.insert({var.get(), IntSet::Interval(min, max)});

  PartitionFinder finder(var, hint_map_, relax_map_);
  finder(body);

  hint_map_.erase(var.get());
  if (finder.partitions.empty()) return Stmt();

  arith::IntervalSet for_interval(min, max);
  bool cond_value;
  IntSet middle_interval;
  ExpressionSet cond_set;
  // find an interval in which all conditions on var are true
  std::tie(middle_interval, cond_set) =
      GetIntervalAndCondset(finder.partitions, for_interval, true);
  if (middle_interval.IsNothing()) {
    // if such interval doesn't exist, find an interval in which all
    // conditions on var are false
    std::tie(middle_interval, cond_set) =
        GetIntervalAndCondset(finder.partitions, for_interval, false);
    if (middle_interval.IsNothing())
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
  PrimExpr body_begin;
  Stmt pre_stmt;
  bool pre_stmt_recurse = true;
  if (middle_interval_i->HasLowerBound()) {
    body_begin = analyzer_.Simplify(middle_interval.min());
    if (!analyzer_.CanProve(body_begin == min)) {
      PrimExpr cond = (body_begin - min >= 0);
      if (!analyzer_.CanProve(cond)) {
        LOG(WARNING) << "Cannot prove: " << cond << ", when generating the pre doubt loop";
        body_begin = Max(body_begin, min);
        // stop recursing on this interval if we can't prove it has non-negative length
        pre_stmt_recurse = false;
      }
      if (!partition_thread_scope) {
        Stmt pre_body = Substitute(body, {{Var{var}, var + min}});
        pre_stmt = MakeFor(stmt.get(), body_begin - min, pre_body);
      }
    }
  } else {
    body_begin = min;
  }

  // Calculating post-subrange and generating code for it.
  // post-subrange = [post_doubt_begin, max+1)
  PrimExpr post_doubt_begin;
  Stmt post_stmt;
  bool post_stmt_recurse = true;
  if (middle_interval_i->HasUpperBound()) {
    post_doubt_begin = analyzer_.Simplify(middle_interval.max() + 1);
    if (!analyzer_.CanProve(middle_interval.max() == max)) {
      // require the extent to be non-negative
      PrimExpr cond = (max - post_doubt_begin + 1 >= 0);
      if (!analyzer_.CanProve(cond)) {
        LOG(WARNING) << "Cannot prove: " << cond << ", when generating the post doubt loop";
        post_doubt_begin = Min(post_doubt_begin, max + 1);
        // stop recursing on this interval if we can't prove it has non-negative length
        post_stmt_recurse = false;
      }
      if (!partition_thread_scope) {
        Stmt post_body = Substitute(body, {{Var{var}, var + post_doubt_begin}});
        post_stmt = MakeFor(stmt.get(), max - post_doubt_begin + 1, post_body);
      }
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
      Stmt simplified_body = ConditionEliminator(cond_set, cond_value)(body);
      Stmt new_body = Substitute(simplified_body, {{Var{var}, var + body_begin}});
      mid_stmt = MakeFor(stmt.get(), post_doubt_begin - body_begin, new_body);

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
    s = SeqStmt::Flatten(pre_stmt, mid_stmt, post_stmt);
  } else {
    PrimExpr cond = const_true();
    if (!analyzer_.CanProve(body_begin == min)) cond = cond && (var >= body_begin);
    if (!analyzer_.CanProve(post_doubt_begin == (max + 1))) cond = cond && (var < post_doubt_begin);
    s = ThreadPartitionInserter(cond_set, cond)(stmt);
  }
  s = ConvertSSA(s);
  return s;
}

inline Stmt LoopPartitioner::MakeFor(const Object* node, PrimExpr extent, Stmt body) {
  const ForNode* for_node = static_cast<const ForNode*>(node);
  CHECK(for_node);
  if (analyzer_.CanProve(extent == make_const(DataType::Int(32), 1))) {
    // If the loop extent is 1, do not create the loop anymore
    return Substitute(body, {{Var{for_node->loop_var}, make_const(DataType::Int(32), 0)}});
  } else {
    return For(for_node->loop_var, IntImm(for_node->min.dtype(), 0), extent, for_node->for_type,
               for_node->device_api, body);
  }
}

class RemoveLikelyTags : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      CHECK_EQ(op->args.size(), 1);
      return StmtExprMutator::VisitExpr(op->args[0]);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }
};

Stmt LoopPartition(Stmt stmt, bool partition_const_loop) {
  stmt = LoopPartitioner(partition_const_loop).VisitAndMutate(std::move(stmt));
  stmt = RemoveLikelyTags()(std::move(stmt));
  return stmt;
}

namespace transform {

Pass LoopPartition() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<LoopPartitionConfig>("tir.LoopPartition");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<LoopPartitionConfig>();
    }
    n->body = LoopPartition(std::move(n->body), cfg.value()->partition_const_loop);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LoopPartition", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LoopPartition").set_body_typed(LoopPartition);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
