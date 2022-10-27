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
 * \file remove_no_op.cc
 * \brief Remove no op from the stmt
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>

#include "../../arith/const_fold.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

// Mark the statement of each stage.
class NoOpRemover : public StmtMutator {
 public:
  Stmt VisitStmt_(const LetStmtNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<LetStmtNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "pragma_debug_skip_region") {
      return MakeEvaluate(0);
    } else if (op->attr_key == attr::async_wait_queue_scope) {
      auto wait_attrs = GetAsyncWaitAttributes(op);
      auto wait_cnt = wait_attrs.second;
      arith::Analyzer ana;
      if (ana.CanProve(wait_cnt < 0)) {
        // A negative wait count can arise if it depends on a loop variable.
        // For example, a wait count 1 - i can be negative after loop unrolling.
        // We assume that such wait is a nop.
        auto inner = op->body.as<AttrStmtNode>();
        ICHECK(inner);
        return StmtMutator::VisitStmt(inner->body);
      }
    }

    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AttrStmtNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<IfThenElseNode>();
    if (op->else_case) {
      if (is_no_op(op->else_case.value())) {
        if (is_no_op(op->then_case)) {
          return MakeEvaluate(op->condition);
        } else {
          return IfThenElse(op->condition, op->then_case);
        }
      } else {
        return stmt;
      }
    } else {
      if (is_no_op(op->then_case)) {
        return MakeEvaluate(op->condition);
      } else {
        return stmt;
      }
    }
  }
  Stmt VisitStmt_(const ForNode* op) final {
    var_range_map_[op->loop_var.get()] = arith::IntSet::FromMinExtent(op->min, op->extent);
    auto extent_range = arith::EvalSet(op->extent, var_range_map_);
    if (!arith::is_neg_inf(extent_range.max()) && !arith::is_pos_inf(extent_range.max()) &&
        analyzer_.CanProve(extent_range.max() <= 0)) {
      return Evaluate(0);
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    var_range_map_.erase(op->loop_var.get());
    op = stmt.as<ForNode>();
    if (is_zero(op->extent)) {
      return Evaluate(0);
    }
    return is_no_op(op->body) ? MakeEvaluate({op->min, op->extent}) : stmt;
  }
  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    return is_no_op(op->body) ? MakeEvaluate(op->extents) : stmt;
  }

  Stmt VisitStmt_(const ProducerRealizeNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ProducerRealizeNode>();
    return is_no_op(op->body) ? op->body : stmt;
  }
  Stmt VisitStmt_(const EvaluateNode* op) final {
    if (SideEffect(op->value) > CallEffectKind::kReadState) return GetRef<Stmt>(op);
    return Evaluate(0);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Stmt ret = StmtMutator::VisitSeqStmt_(op, true);
    op = ret.as<SeqStmtNode>();
    ICHECK(op != nullptr);
    bool need_compact = false;
    for (size_t i = 0; i < op->size(); ++i) {
      if (is_no_op(op->seq[i])) need_compact = true;
    }
    if (need_compact) {
      auto n = CopyOnWrite(op);
      size_t top = 0;
      for (size_t i = 0; i < n->seq.size(); ++i) {
        if (!is_no_op(n->seq[i])) {
          n->seq.Set(top++, n->seq[i]);
        }
      }
      if (top == 1) {
        return n->seq[0];
      } else {
        n->seq.resize(top);
        return Stmt(n);
      }
    } else {
      if (op->size() == 1) {
        return op->seq[0];
      } else {
        return ret;
      }
    }
  }

 private:
  Stmt MakeEvaluate(PrimExpr value) {
    if (SideEffect(value) > CallEffectKind::kReadState) {
      return Evaluate(value);
    } else {
      return Evaluate(0);
    }
  }
  Stmt MakeEvaluate(const Array<PrimExpr>& values) {
    Stmt stmt;
    for (PrimExpr e : values) {
      if (SideEffect(e) > CallEffectKind::kReadState) {
        if (stmt.defined()) {
          stmt = SeqStmt({stmt, Evaluate(e)});
        } else {
          stmt = Evaluate(e);
        }
      }
    }
    return stmt.defined() ? stmt : Evaluate(0);
  }

  std::unordered_map<const VarNode*, arith::IntSet> var_range_map_;
  arith::Analyzer analyzer_;
};

Stmt RemoveNoOp(Stmt stmt) { return NoOpRemover()(std::move(stmt)); }

namespace transform {

Pass RemoveNoOp() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = NoOpRemover()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.RemoveNoOp", {});
}

TVM_REGISTER_GLOBAL("tir.transform.RemoveNoOp").set_body_typed(RemoveNoOp);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
