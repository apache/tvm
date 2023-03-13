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
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../arith/partition_finder.h"
#include "../../runtime/thread_storage_scope.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

struct LoopPartitionConfigNode : public tvm::AttrsNode<LoopPartitionConfigNode> {
  bool partition_const_loop;
  bool no_unroll_loop_with_extent_one;
  bool unroll_loop_with_partition_hint_no_interval;

  TVM_DECLARE_ATTRS(LoopPartitionConfigNode, "tir.transform.LoopPartitionConfig") {
    TVM_ATTR_FIELD(partition_const_loop).describe("Split constant loop").set_default(false);
    TVM_ATTR_FIELD(no_unroll_loop_with_extent_one)
        .describe("Don't unroll loops with extent 1")
        .set_default(false);
    TVM_ATTR_FIELD(unroll_loop_with_partition_hint_no_interval)
        .describe("Unroll loops with pragma_loop_partition_hint and no interval")
        .set_default(false);
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
using arith::PartitionDecision;

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
    if (!is_const_int(op->min) || !is_const_int(op->extent) || partition_const_loop_ ||
        partition_hint_vars.count(op->loop_var.get())) {
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
      ICHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      if ((scope.rank == 0) && (!is_const_int(op->value) || partition_const_loop_)) {
        // always treat var with hint to be partitioned
        if (partition_hint_vars.count(var.get())) {
          candidates.insert(GetRef<Stmt>(op));
          StmtExprVisitor::VisitStmt_(op);
          return;
        }
        record_.insert({var.get(), false});
        StmtExprVisitor::VisitStmt_(op);
        if (record_.at(var.get()) && !no_split_) {
          candidates.insert(GetRef<Stmt>(op));
        }
        record_.erase(var.get());
        return;
      }
    } else if (op->attr_key == attr::pragma_loop_partition_hint) {
      if (analyzer_.CanProve(op->value)) {
        const VarNode* var = nullptr;
        if (op->node->IsInstance<VarNode>()) {
          var = op->node.as<VarNode>();
        } else if (op->node->IsInstance<IterVarNode>()) {
          var = op->node.as<IterVarNode>()->var.get();
        }
        ICHECK(var);
        partition_hint_vars.insert(var);
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
      bool prev_in_likely = in_likely_;
      in_likely_ = true;
      StmtExprVisitor::VisitExpr_(op);
      in_likely_ = prev_in_likely;
    } else if (op->op.same_as(builtin::tvm_thread_allreduce())) {
      // no split if the body contains allreduce.
      no_split_ = true;
      return;
    } else if (op->op.same_as(builtin::if_then_else())) {
      bool prev_in_condition = in_condition_;
      in_condition_ = true;
      VisitExpr(op->args[0]);
      in_condition_ = prev_in_condition;
      VisitExpr(op->args[1]);
      VisitExpr(op->args[2]);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    bool prev_in_condition = in_condition_;
    in_condition_ = true;
    VisitExpr(op->condition);
    in_condition_ = prev_in_condition;

    VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
  }

  void VisitExpr_(const VarNode* op) final {
    if (record_.count(op) && (in_likely_ || (in_condition_ && partition_hint_vars.count(op)))) {
      record_.at(op) = true;
    }
  }

  std::unordered_set<Stmt, ObjectPtrHash, ObjectPtrEqual> candidates;
  std::unordered_set<const VarNode*> partition_hint_vars;

 private:
  bool in_likely_{false};
  bool in_condition_{false};
  bool no_split_{false};
  bool partition_const_loop_{false};
  std::unordered_map<const VarNode*, VarIsUsed> record_;
  arith::Analyzer analyzer_;
};

// Replace the set of conditions given by ps with cond_value (true or false)
class ConditionEliminator : public StmtExprMutator {
 public:
  explicit ConditionEliminator(const Map<PrimExpr, Bool>& cond_map) : cond_map_(cond_map) {}

  PrimExpr VisitExpr(const PrimExpr& e) final {
    auto it = cond_map_.find(e);
    if (it != cond_map_.end()) {
      bool cond_value = (*it).second->value;
      return VisitExpr(cond_value ? const_true() : const_false());
    }
    return StmtExprMutator::VisitExpr(e);
  }

 private:
  const Map<PrimExpr, Bool>& cond_map_;
};

// Insert the partition branch at the innermost thread scope
class ThreadPartitionInserter : public StmtMutator {
 public:
  explicit ThreadPartitionInserter(const Map<PrimExpr, Bool>& cond_map, PrimExpr cond)
      : cond_map_(cond_map), cond_(cond), innermost_thread_scope_(false) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      innermost_thread_scope_ = true;
      Stmt stmt = StmtMutator::VisitStmt_(op);
      // add branch code inside the innermost thread scope
      if (innermost_thread_scope_) {
        Stmt simplified_body = ConditionEliminator(cond_map_)(op->body);
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
  const Map<PrimExpr, Bool>& cond_map_;
  PrimExpr cond_;
  bool innermost_thread_scope_;
};

// Try to partition range of iteration variables in order to remove (some)
// likely conditions
class LoopPartitioner : public StmtMutator {
 public:
  explicit LoopPartitioner(bool partition_const_loop, bool no_unroll_loop_with_extent_one,
                           bool unroll_loop_with_partition_hint_no_interval)
      : selector(CandidateSelector(partition_const_loop)),
        no_unroll_loop_with_extent_one_(no_unroll_loop_with_extent_one),
        unroll_loop_with_partition_hint_no_interval_(unroll_loop_with_partition_hint_no_interval) {}

  Stmt VisitAndMutate(Stmt stmt) {
    selector(stmt);
    return operator()(std::move(stmt));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent), true);
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
    ICHECK(iv);
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

  inline Stmt MakeFor(const Object* op, PrimExpr extent, Stmt body);

  /* Candidate IRs that may be partitioned potentially */
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
  arith::Analyzer analyzer_;
  CandidateSelector selector;
  bool no_unroll_loop_with_extent_one_;
  bool unroll_loop_with_partition_hint_no_interval_;
};

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
  // include hint of var.
  arith::IntervalSet for_interval(min, max);
  hint_map_.insert({var.get(), for_interval});

  bool has_partition_hint_ = selector.partition_hint_vars.count(var.get());

  Optional<PartitionDecision> opt_partition = arith::SearchBestPartition(
      /*var=*/var, /*loop_range=*/for_interval, /*hint_map=*/hint_map_, /*relax_map=*/relax_map_,
      /*process_likely_only=*/!has_partition_hint_,
      /*deduce_min_max=*/false,
      /*analyzer=*/&analyzer_,
      /*stmt=*/body);

  hint_map_.erase(var.get());

  if (!opt_partition.defined()) {
    if (has_partition_hint_ && unroll_loop_with_partition_hint_no_interval_ &&
        analyzer_.CanProve(max - min > 0)) {
      auto new_body = VisitAndMutate(body);
      return For(var, min, max - min + 1, ForKind::kUnrolled, new_body);
    }
    return Stmt();
  }

  PartitionDecision partition = opt_partition.value();

  arith::IntervalSet middle_interval = partition->interval;
  // middle_interval is the subrange of the loop variable range for which a
  // set of conditions are true (or false resp.)
  // The part of the loop variable range that is before (after resp.) that
  // subrange is prefixed with pre- (post- resp.)

  // Calculating pre-subrange and generating code for it.
  // pre-subrange = [min, body_begin)
  PrimExpr body_begin;
  Stmt pre_stmt;
  bool pre_stmt_recurse = true;
  if (middle_interval->HasLowerBound()) {
    body_begin = analyzer_.Simplify(middle_interval.min());
    if (!analyzer_.CanProve(body_begin == min)) {
      PrimExpr extent = analyzer_.Simplify(body_begin - min);
      if (!analyzer_.CanProve(extent > 0)) {
        body_begin = tvm::max(body_begin, min);
        // stop recursing on this interval if we can't prove it has non-negative length
        pre_stmt_recurse = false;
      }
      if (!analyzer_.CanProve(extent <= 0)) {
        if (!partition_thread_scope) {
          Stmt pre_body = Substitute(body, {{Var{var}, var + min}});
          pre_stmt = MakeFor(stmt.get(), body_begin - min, pre_body);
        }
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
  if (middle_interval->HasUpperBound()) {
    post_doubt_begin = analyzer_.Simplify(middle_interval.max() + 1);
    if (!analyzer_.CanProve(middle_interval.max() == max)) {
      // require the extent to be non-negative
      PrimExpr extent = analyzer_.Simplify(max - post_doubt_begin + 1);
      if (!analyzer_.CanProve(extent > 0)) {
        post_doubt_begin = tvm::min(post_doubt_begin, max + 1);
        // stop recursing on this interval if we can't prove it has non-negative length
        post_stmt_recurse = false;
      }
      if (!analyzer_.CanProve(extent <= 0)) {
        if (!partition_thread_scope) {
          Stmt post_body = Substitute(body, {{Var{var}, var + post_doubt_begin}});
          post_stmt = MakeFor(stmt.get(), extent, post_body);
        }
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
      Stmt simplified_body = ConditionEliminator(partition->cond_map)(body);
      Stmt new_body = Substitute(simplified_body, {{Var{var}, var + body_begin}});
      mid_stmt = MakeFor(stmt.get(), post_doubt_begin - body_begin, new_body);
      // Recurse until partitions is empty
      mid_stmt = VisitAndMutate(mid_stmt);
      // Recurse for each non-empty subrange only if there are at least
      // two non-empty subranges
      if (pre_stmt.defined() || post_stmt.defined()) {
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
    s = ThreadPartitionInserter(partition->cond_map, cond)(stmt);
  }
  s = ConvertSSA(s);
  return s;
}

inline Stmt LoopPartitioner::MakeFor(const Object* node, PrimExpr extent, Stmt body) {
  const ForNode* for_node = static_cast<const ForNode*>(node);
  ICHECK(for_node);
  if (analyzer_.CanProve(extent == make_const(DataType::Int(32), 1)) &&
      !no_unroll_loop_with_extent_one_ && for_node->annotations.empty()) {
    // If the loop extent is 1, do not create the loop anymore
    return Substitute(body, {{Var{for_node->loop_var}, make_const(DataType::Int(32), 0)}});
  } else {
    ICHECK(for_node->kind != ForKind::kThreadBinding);
    return For(for_node->loop_var, IntImm(for_node->min.dtype(), 0), extent, for_node->kind, body,
               for_node->thread_binding, for_node->annotations);
  }
}

class RemoveLikelyTagsAndHints : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      ICHECK_EQ(op->args.size(), 1);
      return StmtExprMutator::VisitExpr(op->args[0]);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::pragma_loop_partition_hint) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

Stmt LoopPartition(Stmt stmt, bool partition_const_loop, bool no_unroll_loop_with_extent_one,
                   bool unroll_loop_with_partition_hint_no_interval) {
  stmt = LoopPartitioner(partition_const_loop, no_unroll_loop_with_extent_one,
                         unroll_loop_with_partition_hint_no_interval)
             .VisitAndMutate(std::move(stmt));
  stmt = RemoveLikelyTagsAndHints()(std::move(stmt));
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
    n->body = LoopPartition(std::move(n->body), cfg.value()->partition_const_loop,
                            cfg.value()->no_unroll_loop_with_extent_one,
                            cfg.value()->unroll_loop_with_partition_hint_no_interval);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LoopPartition", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LoopPartition").set_body_typed(LoopPartition);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
