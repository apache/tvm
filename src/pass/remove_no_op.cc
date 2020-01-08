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
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_functor_ext.h>
#include <unordered_map>

namespace tvm {
namespace ir {

// Mark the statment of each stage.
class NoOpRemover : public StmtMutator {
 public:
  Stmt VisitStmt_(const LetStmt* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<LetStmt>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt VisitStmt_(const AttrStmt* op) final {
    if (op->attr_key == "pragma_debug_skip_region") {
      return MakeEvaluate(0);
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AttrStmt>();
    return is_no_op(op->body) ? MakeEvaluate(op->value) : stmt;
  }
  Stmt VisitStmt_(const IfThenElse* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<IfThenElse>();
    if (op->else_case.defined()) {
      if (is_no_op(op->else_case)) {
        if (is_no_op(op->then_case)) {
          return MakeEvaluate(op->condition);
        } else {
          return IfThenElse::make(op->condition, op->then_case);
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
  Stmt VisitStmt_(const For* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<For>();
    if (is_zero(op->extent)) {
      return Evaluate::make(0);
    }
    return is_no_op(op->body) ? MakeEvaluate({op->min, op->extent}) : stmt;
  }
  Stmt VisitStmt_(const Allocate* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<Allocate>();
    return is_no_op(op->body) ? MakeEvaluate(op->extents) : stmt;
  }
  Stmt VisitStmt_(const ProducerConsumer* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ProducerConsumer>();
    return is_no_op(op->body) ? op->body : stmt;
  }
  Stmt VisitStmt_(const Realize* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<Realize>();
    return is_no_op(op->body) ? op->body : stmt;
  }
  Stmt VisitStmt_(const Evaluate* op) final {
    if (HasSideEffect(op->value)) return GetRef<Stmt>(op);
    return Evaluate::make(0);
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    Stmt ret = StmtMutator::VisitSeqStmt_(op, true);
    op = ret.as<SeqStmtNode>();
    CHECK(op != nullptr);
    bool need_compact = false;
    for (size_t i = 0; i < op->size(); ++i) {
      if (is_no_op(op->seq[i])) need_compact = true;
    }
    if (need_compact) {
      auto n = CopyOnWrite(op);
      size_t top = 0;
      for (size_t i = 0; i < n->seq.size(); ++i) {
        if (!is_no_op(n->seq[i]))  {
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
  Stmt MakeEvaluate(Expr value) {
    if (HasSideEffect(value)) {
      return Evaluate::make(value);
    } else {
      return Evaluate::make(0);
    }
  }
  Stmt MakeEvaluate(const Array<Expr>& values) {
    Stmt stmt;
    for (Expr e : values) {
      if (HasSideEffect(e)) {
        if (stmt.defined()) {
          stmt = SeqStmt({stmt, Evaluate::make(e)});
        } else {
          stmt = Evaluate::make(e);
        }
      }
    }
    return stmt.defined() ? stmt : Evaluate::make(0);
  }
};

Stmt RemoveNoOp(Stmt stmt) {
  return NoOpRemover()(std::move(stmt));
}
}  // namespace ir
}  // namespace tvm
