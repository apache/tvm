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
 * \file data_type_rewriter.cc
 * \brief Rewrite the data type of expressions.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "./functor_common.h"

namespace tvm {
namespace tir {

Stmt DataTypeLegalizer::VisitStmt_(const ForNode* op) {
  Stmt s = StmtExprMutator::VisitStmt_(op);
  op = s.as<ForNode>();
  ICHECK(op != nullptr) << "Expected type to be ForNode, but get " << s->GetTypeKey();
  PrimExpr e = VisitExpr(op->loop_var);
  Var var = Downcast<Var>(e);
  return For(var, cast(var.dtype(), op->min), cast(var.dtype(), op->extent), op->kind, op->body,
             op->thread_binding, op->annotations);
}

Stmt DataTypeLegalizer::VisitStmt_(const BlockRealizeNode* op) {
  BlockRealize realize = Downcast<BlockRealize>(StmtExprMutator::VisitStmt_(op));
  Array<PrimExpr> new_iter_values;
  bool changed = false;
  for (int i = 0; i < static_cast<int>(op->iter_values.size()); ++i) {
    auto dtype = realize->block->iter_vars[i]->var->dtype;
    if (op->iter_values[i]->dtype != dtype) {
      new_iter_values.push_back(cast(dtype, realize->iter_values[i]));
      changed = true;
    } else {
      new_iter_values.push_back(realize->iter_values[i]);
    }
  }
  if (changed) {
    realize.CopyOnWrite()->iter_values = std::move(new_iter_values);
  }
  return std::move(realize);
}

Stmt DataTypeLegalizer::VisitStmt_(const BlockNode* op) {
  Block new_block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
  Array<IterVar> new_iter_vars = MutateArray(new_block->iter_vars, [this](const IterVar& iter) {
    auto dtype = iter->var.dtype();
    if (iter->dom->min->dtype != dtype || iter->dom->extent->dtype != dtype) {
      IterVar new_iter = iter;
      new_iter.CopyOnWrite()->dom =
          Range(cast(dtype, iter->dom->min), cast(dtype, iter->dom->extent));
      return new_iter;
    } else {
      return iter;
    }
  });
  if (!op->iter_vars.same_as(new_iter_vars)) {
    new_block.CopyOnWrite()->iter_vars = std::move(new_iter_vars);
  }
  return std::move(new_block);
}

Stmt DataTypeLegalizer::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
    Stmt s = StmtExprMutator::VisitStmt_(op);
    op = s.as<AttrStmtNode>();
    ICHECK(op != nullptr) << "Expected type to be AttrStmtNode"
                          << ", but get " << s->GetTypeKey();
    const IterVarNode* iv = op->node.as<IterVarNode>();
    ICHECK(iv != nullptr) << "Expected type to be IterVarNode"
                          << ", but get " << op->node->GetTypeKey();
    PrimExpr e = VisitExpr(iv->var);
    Var var = Downcast<Var>(e);
    if (ivmap_.find(iv) == ivmap_.end()) {
      Range dom = iv->dom;
      if (dom.defined()) {
        PrimExpr extend = dom->extent;
        ICHECK(extend.dtype().is_int() && var.dtype().is_int());
        if (var.dtype().bits() != extend.dtype().bits()) {
          DataType dtype = var.dtype();
          dom = Range(cast(dtype, dom->min), cast(dtype, extend), dom->span);
        }
      }
      ivmap_[iv] = IterVar(dom, var, iv->iter_type, iv->thread_tag);
    }
    return AttrStmt(ivmap_[iv], op->attr_key, cast(var.dtype(), op->value), op->body);
  }
  return StmtExprMutator::VisitStmt_(op);
}

PrimExpr DataTypeLegalizer::VisitExpr_(const SelectNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr true_value = this->VisitExpr(op->true_value);
  PrimExpr false_value = this->VisitExpr(op->false_value);
  if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value) && true_value.dtype() == false_value.dtype()) {
    return GetRef<PrimExpr>(op);
  } else {
    int bits = std::max(true_value.dtype().bits(), false_value.dtype().bits());
    DataType dtype = true_value.dtype().with_bits(bits);
    if (true_value.dtype() != dtype) true_value = cast(dtype, true_value);
    if (false_value.dtype() != dtype) false_value = cast(dtype, false_value);
    return Select(condition, true_value, false_value);
  }
}

PrimExpr DataTypeLegalizer::VisitExpr_(const RampNode* op) {
  PrimExpr base = VisitExpr(op->base);
  PrimExpr stride = VisitExpr(op->stride);
  if (base.same_as(op->base) && stride.same_as(op->stride) && base.dtype() == stride.dtype()) {
    return GetRef<PrimExpr>(op);
  } else {
    ICHECK(base.dtype().is_int() && stride.dtype().is_int());
    int bits = std::max(base.dtype().bits(), stride.dtype().bits());
    DataType dtype = base.dtype().with_bits(bits);
    if (base.dtype() != dtype) base = cast(dtype, base);
    if (stride.dtype() != dtype) stride = cast(dtype, stride);
    return Ramp(base, stride, op->lanes);
  }
}

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                 \
  PrimExpr DataTypeLegalizer::VisitExpr_(const OP* op) {                  \
    PrimExpr a = this->VisitExpr(op->a);                                  \
    PrimExpr b = this->VisitExpr(op->b);                                  \
    if (op->a.same_as(a) && op->b.same_as(b) && a.dtype() == b.dtype()) { \
      return GetRef<PrimExpr>(op);                                        \
    } else {                                                              \
      return FUNC(a, b);                                                  \
    }                                                                     \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(ModNode, truncmod);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorDivNode, floordiv);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(FloorModNode, floormod);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(EQNode, operator==);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(NENode, operator!=);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=);
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<);  // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>);  // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=);

#undef DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH

PrimExpr DataTypeLegalizer::VisitExpr_(const CallNode* op) {
  PrimExpr e = StmtExprMutator::VisitExpr_(op);
  op = e.as<CallNode>();
  static const Op& builtin_pow_ = Op::Get("tir.pow");
  ICHECK(op != nullptr) << "Expected type to be CallNode"
                        << ", but get " << e->GetTypeKey();
  if (op->op.same_as(builtin::shift_right())) {
    return op->args[0] >> op->args[1];
  } else if (op->op.same_as(builtin::shift_left())) {
    return op->args[0] << op->args[1];
  } else if (op->op.same_as(builtin::bitwise_and())) {
    return op->args[0] & op->args[1];
  } else if (op->op.same_as(builtin::bitwise_or())) {
    return op->args[0] | op->args[1];
  } else if (op->op.same_as(builtin::bitwise_xor())) {
    return op->args[0] ^ op->args[1];
  } else if (op->op.same_as(builtin_pow_)) {
    return pow(op->args[0], op->args[1]);
  } else if (op->op.same_as(builtin::if_then_else())) {
    return if_then_else(op->args[0], op->args[1], op->args[2]);
  }
  return e;
}

}  // namespace tir
}  // namespace tvm
