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
 *
 * \brief Lift specified AttrStmt scope to outer if
 *   the body contains the same scope.
 * \file lift_attr_scope.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt_functor.h>
#include "ir_util.h"

namespace tvm {
namespace tir {

// NOTE: this optimization can only be applied
// to a few specified attr keys
class AttrScopeLifter : public StmtMutator {
 public:
  explicit AttrScopeLifter(std::string attr_key)
      : attr_key_(attr_key) {}

  Stmt Lift(Stmt stmt) {
    stmt = operator()(std::move(stmt));
    if (attr_node_.defined()) {
      stmt = AttrStmtNode::make(
          attr_node_, attr_key_, attr_value_, stmt);
    }
    return stmt;
  }

  // do not go beyond
  Stmt VisitStmt_(const AllocateNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<AllocateNode>();
    if (attr_node_.defined()) {
      Stmt body = AttrStmtNode::make(
          attr_node_, attr_key_, attr_value_, op->body);
      // undefine them
      attr_node_ = ObjectRef();
      attr_value_ = PrimExpr();
      return AllocateNode::make(
        op->buffer_var, op->dtype,
        op->extents, op->condition, body);
    } else {
      return stmt;
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr_key_) {
      attr_node_ = op->node;
      attr_value_ = op->value;
      return op->body;
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* op) final {
    // remember the decorations.
    std::vector<ObjectRef> attr_node;
    std::vector<PrimExpr> attr_value;

    auto fmutate = [&](const Stmt& s) {
      attr_node_ = ObjectRef();
      attr_value_ = PrimExpr();
      Stmt ret = this->VisitStmt(s);
      attr_node.push_back(attr_node_);
      attr_value.push_back(attr_value_);
      return ret;
    };
    Stmt ret = StmtMutator::VisitSeqStmt_(op, true, fmutate);
    if (attr_node.size() == 0) return ret;

    op = ret.as<SeqStmtNode>();
    CHECK(op != nullptr);
    Array<Stmt> reorg;
    // check if all decorations are common.
    for (size_t begin = 0; begin < attr_node.size();) {
      size_t end = begin + 1;
      while (end < attr_node.size() &&
             attr_node[end].same_as(attr_node[begin]) &&
             ValueSame(attr_value[end], attr_value[begin])) {
        ++end;
      }
      // covers everything
      // lift attr to parent.
      if (begin == 0 && end == attr_node.size()) {
        attr_node_ = attr_node[0];
        attr_value_ = attr_value[0];
        return ret;
      }
      // construct subsegments.
      Array<Stmt> seq;
      for (size_t i = begin; i < end; ++i) {
        seq.push_back(op->seq[i]);
      }
      Stmt stmt = SeqStmt::Flatten(seq);
      if (attr_node[begin].defined()) {
        stmt = AttrStmtNode::make(
            attr_node[begin], attr_key_, attr_value[begin], stmt);
      }
      reorg.push_back(stmt);
      begin = end;
    }
    attr_node_ = ObjectRef();
    attr_value_ = PrimExpr();
    return SeqStmt::Flatten(reorg);
  }

  Stmt VisitStmt_(const IfThenElseNode* op) final {
    if (!op->else_case.defined()) {
      return StmtMutator::VisitStmt_(op);
    }
    Stmt then_case = this->VisitStmt(op->then_case);
    ObjectRef first_node;
    PrimExpr first_value;
    std::swap(first_node, attr_node_);
    std::swap(first_value, attr_value_);
    Stmt else_case = this->VisitStmt(op->else_case);
    if (attr_node_.defined() &&
        attr_value_.defined() &&
        first_node.defined() &&
        first_value.defined() &&
        attr_node_.same_as(first_node) &&
        ValueSame(attr_value_, first_value)) {
      if (then_case.same_as(op->then_case) &&
          else_case.same_as(op->else_case)) {
        return GetRef<Stmt>(op);
      } else {
        return IfThenElseNode::make(op->condition, then_case, else_case);
      }
    } else {
      if (first_node.defined()) {
        then_case = AttrStmtNode::make(
            first_node, attr_key_, first_value, then_case);
      }
      if (attr_node_.defined()) {
        else_case = AttrStmtNode::make(
            attr_node_, attr_key_, attr_value_, else_case);
        // undefine them
        attr_node_ = ObjectRef();
        attr_value_ = PrimExpr();
      }
      if (then_case.same_as(op->then_case) &&
          else_case.same_as(op->else_case)) {
        return GetRef<Stmt>(op);
      } else {
        return IfThenElseNode::make(op->condition, then_case, else_case);
      }
    }
  }

 private:
  // value comparison that also compares content of int constant
  static bool ValueSame(const PrimExpr& a, const PrimExpr& b) {
    if (a.same_as(b)) return true;
    if (!a.defined() || !b.defined()) return false;
    if (a->type_index() != b->type_index()) return false;
    if (a.dtype() != b.dtype()) return false;
    if (const IntImmNode* op = a.as<IntImmNode>()) {
      return op->value == b.as<IntImmNode>()->value;
    }
    return false;
  }

  std::string attr_key_;
  ObjectRef attr_node_;
  PrimExpr attr_value_;
};

Stmt LiftAttrScope(Stmt stmt, std::string attr_key) {
  return AttrScopeLifter(attr_key).Lift(std::move(stmt));
}


namespace transform {

Pass LiftAttrScope(std::string attr_key) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = AttrScopeLifter(attr_key).Lift(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LiftAttrScope", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LiftAttrScope")
.set_body_typed(LiftAttrScope);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
