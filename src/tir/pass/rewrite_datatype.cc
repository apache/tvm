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
 * \file rewrite_datatype.cc
 */

#include <tvm/tir/ir_pass.h>
#include <tvm/tir/op.h>
#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tir {

using arith::Analyzer;
using arith::IRMutatorWithAnalyzer;
using arith::ConstIntBound;

class DataTypeRewriter;

class DataTypeVisitor final : public StmtExprVisitor {
 public:
  void VisitExpr(const PrimExpr& e) {
    if (e.dtype().is_int()) {
      int bits = 64;
      if (e.dtype().bits() <= 32 ||
          analyzer_.CanProve(e <= max_value(DataType::Int(32)) &&
                             e >= min_value(DataType::Int(32)))) {
        bits = 32;
      }
      int tmp = bits_;
      bits_ = bits > bits_ ? bits :  bits_;
      StmtExprVisitor::VisitExpr(e);
      bits_ = tmp;
    } else {
      StmtExprVisitor::VisitExpr(e);
    }
  }

  void VisitStmt_(const ForNode* op) {
    analyzer_.Bind(op->loop_var,
                   Range::make_by_min_extent(op->min, op->extent));
    vset_.insert(op->loop_var.as<Object>());
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      IterVar iv = Downcast<IterVar>(op->node);
      CHECK_NE(iv->thread_tag.length(), 0U);
      analyzer_.Bind(iv->var,
                      Range::make_by_min_extent(0, op->value));
      vset_.insert(iv->var.as<Object>());
      StmtExprVisitor::VisitStmt_(op);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitExpr_(const ReduceNode* op) {
    // Setup the domain information before simplification.
    for (const IterVar& iv : op->axis) {
      analyzer_.Bind(iv->var, iv->dom);
      vset_.insert(iv->var.as<Object>());
    }
    // Recursively call simplification when necessary.
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode* op) {
    if (vset_.find(op) != vset_.end()) {
      if (vmap.find(op) == vmap.end()) {
        vmap[op] = op->dtype.with_bits(bits_);
      } else {
        vmap[op] = op->dtype.with_bits(std::max(vmap[op].bits(), bits_));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IntImmNode* op) {
    if (op->dtype.is_int()) {
      int bits = std::min(op->dtype.bits(), bits_);
      if (vmap.find(op) == vmap.end()) {
        vmap[op] = op->dtype.with_bits(bits);
      } else {
        vmap[op] = op->dtype.with_bits(std::max(vmap[op].bits(), bits));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CastNode* op) {
    if (op->dtype.is_int()) {
      int bits = std::min(op->dtype.bits(), bits_);
      if (vmap.find(op) == vmap.end()) {
        vmap[op] = op->dtype.with_bits(bits);
      } else {
        vmap[op] = op->dtype.with_bits(std::max(vmap[op].bits(), bits));
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  // the narrowed datatype of Var, IntImm, and Cast
  std::unordered_map<const Object*, DataType> vmap;

 protected:
  // internal analyzer
  arith::Analyzer analyzer_;

 private:
  // the maximum bits of all containing expressions
  int bits_;
  // the vars to be rewritten
  std::unordered_set<const Object*> vset_;
  friend class DataTypeRewriter;
};

class DataTypeRewriter : public StmtExprMutator {
 public:
  Stmt operator()(Stmt s) {
    visitor_(s);
    return VisitStmt(s);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt s = StmtExprMutator::VisitStmt_(op);
    op = s.as<ForNode>();
    PrimExpr e = VisitExpr(op->loop_var);
    Var var = Downcast<Var, PrimExpr>(e);
    return ForNode::make(var, Cast(op->min, var.dtype()), Cast(op->extent, var.dtype()),
                         op->for_type, op->device_api, op->body);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent ||
        op->attr_key == attr::virtual_thread) {
      Stmt s = StmtExprMutator::VisitStmt_(op);
      op = s.as<AttrStmtNode>();
      IterVar iv = Downcast<IterVar>(op->node);
      PrimExpr e = VisitExpr(iv->var);
      Var var = Downcast<Var, PrimExpr>(e);
      return AttrStmtNode::make(
        IterVarNode::make(iv->dom, var, iv->iter_type, iv->thread_tag),
        op->attr_key,
        Cast(op->value, var.dtype()),
        op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      if (vmap_.find(op) == vmap_.end()) {
        vmap_[op] = Var(op->name_hint, visitor_.vmap[op]);
      }
      return vmap_[op];
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const SizeVarNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      if (vmap_.find(op) == vmap_.end()) {
        vmap_[op] = SizeVar(op->name_hint, visitor_.vmap[op]);
      }
      return vmap_[op];
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const IntImmNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      return IntImm(visitor_.vmap[op], op->value);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    if (visitor_.vmap.find(op) != visitor_.vmap.end()) {
      PrimExpr e = StmtExprMutator::VisitExpr_(op);
      const CastNode* new_op = e.as<CastNode>();
      return CastNode::make(visitor_.vmap[op], new_op->value);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const DivNode* op) final;
  PrimExpr VisitExpr_(const ModNode* op) final;
  PrimExpr VisitExpr_(const FloorDivNode* op) final;
  PrimExpr VisitExpr_(const FloorModNode* op) final;
  PrimExpr VisitExpr_(const MinNode* op) final;
  PrimExpr VisitExpr_(const MaxNode* op) final;
  PrimExpr VisitExpr_(const EQNode* op) final;
  PrimExpr VisitExpr_(const NENode* op) final;
  PrimExpr VisitExpr_(const LTNode* op) final;
  PrimExpr VisitExpr_(const LENode* op) final;
  PrimExpr VisitExpr_(const GTNode* op) final;
  PrimExpr VisitExpr_(const GENode* op) final;
  PrimExpr VisitExpr_(const CallNode* op) final;

 private:
  // the internal visitor to deduce the narrowed dtype
  DataTypeVisitor visitor_;
  // a map from Var before rewrite to Var after rewrite,
  // ensures one old Var maps to exactly one new Var
  std::unordered_map<const VarNode*, Var> vmap_;
  PrimExpr Cast(PrimExpr e, DataType dtype) {
    if (e.dtype() != dtype) {
      return CastNode::make(dtype, e);
    } else {
      return e;
    }
  }
};

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)               \
  PrimExpr DataTypeRewriter::VisitExpr_(const OP* op) {                 \
    PrimExpr a = this->VisitExpr(op->a);                                \
    PrimExpr b = this->VisitExpr(op->b);                                \
    if (a.same_as(op->a) &&                                             \
        b.same_as(op->b)) {                                             \
      return GetRef<PrimExpr>(op);                                      \
    } else {                                                            \
      return FUNC(a, b);                                                \
    }                                                                   \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(ModNode, truncmod)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorDivNode, floordiv)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorModNode, floormod)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator <)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator >)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=)

PrimExpr DataTypeRewriter::VisitExpr_(const CallNode* op) {
  PrimExpr e = StmtExprMutator::VisitExpr_(op);
  op = e.as<CallNode>();
  if (op->call_type == CallNode::PureIntrinsic) {
    if (op->name == intrinsic::tvm_if_then_else) {
      return if_then_else(op->args[0], op->args[1], op->args[2]);
    } else if (op->name == CallNode::shift_right) {
      return op->args[0] >> op->args[1];
    } else if (op->name == CallNode::shift_left) {
      return op->args[0] << op->args[1];
    } else if (op->name == CallNode::bitwise_and) {
      return op->args[0] & op->args[1];
    } else if (op->name == CallNode::bitwise_or) {
      return op->args[0] | op->args[1];
    } else if (op->name == CallNode::bitwise_xor) {
      return op->args[0] ^ op->args[1];
    } else if (op->name == "pow") {
      return pow(op->args[0], op->args[1]);
    }
  }
  return e;
}

Stmt DataTypeRewrite(Stmt stmt) {
  return DataTypeRewriter()(stmt);
}

}  // namespace tir
}  // namespace tvm
