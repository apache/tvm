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
 * \file expr_functor.cc
 */
#include <tvm/ffi/cast.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>

#include "functor_common.h"

namespace tvm {
namespace tirx {

void ExprVisitor::VisitExpr_(const VarNode* op) {}

void ExprVisitor::VisitExpr_(const TensorLoadNode* op) {
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const OpaqueExprNode* op) {}

void ExprVisitor::VisitExpr_(const BufferRegionNode* op) {
  VisitArray(op->region, [this](const Range& range) {
    this->VisitExpr(range->min);
    this->VisitExpr(range->extent);
  });
}

void ExprVisitor::VisitExpr_(const TupleNode* op) {
  VisitArray(op->fields, [this](const Expr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const TupleGetItemNode* op) { this->VisitExpr(op->tuple); }

void ExprVisitor::VisitExpr_(const prim::LetNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  if (op->op.as<OpaqueExprNode>()) {
    this->VisitExpr(op->op);
  }
  VisitArray(op->args, [this](const Expr& e) { this->VisitExpr(e); });
}

#define DEFINE_BINOP_VISIT_(OP)                \
  void ExprVisitor::VisitExpr_(const OP* op) { \
    this->VisitExpr(op->a);                    \
    this->VisitExpr(op->b);                    \
  }

DEFINE_BINOP_VISIT_(prim::AddNode);
DEFINE_BINOP_VISIT_(prim::SubNode);
DEFINE_BINOP_VISIT_(prim::MulNode);
DEFINE_BINOP_VISIT_(prim::DivNode);
DEFINE_BINOP_VISIT_(prim::ModNode);
DEFINE_BINOP_VISIT_(prim::FloorDivNode);
DEFINE_BINOP_VISIT_(prim::FloorModNode);
DEFINE_BINOP_VISIT_(prim::MinNode);
DEFINE_BINOP_VISIT_(prim::MaxNode);
DEFINE_BINOP_VISIT_(prim::EQNode);
DEFINE_BINOP_VISIT_(prim::NENode);
DEFINE_BINOP_VISIT_(prim::LTNode);
DEFINE_BINOP_VISIT_(prim::LENode);
DEFINE_BINOP_VISIT_(prim::GTNode);
DEFINE_BINOP_VISIT_(prim::GENode);
DEFINE_BINOP_VISIT_(prim::AndNode);
DEFINE_BINOP_VISIT_(prim::OrNode);

void ExprVisitor::VisitExpr_(const IntImmNode* op) {}
void ExprVisitor::VisitExpr_(const FloatImmNode* op) {}
void ExprVisitor::VisitExpr_(const prim::StringImmNode* op) {}

void ExprVisitor::VisitExpr_(const prim::CastNode* op) { this->VisitExpr(op->value); }

void ExprVisitor::VisitExpr_(const prim::NotNode* op) { this->VisitExpr(op->a); }

void ExprVisitor::VisitExpr_(const prim::SelectNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->true_value);
  this->VisitExpr(op->false_value);
}

void ExprVisitor::VisitExpr_(const prim::RampNode* op) {
  this->VisitExpr(op->base);
  this->VisitExpr(op->stride);
}

void ExprVisitor::VisitExpr_(const prim::ShuffleNode* op) {
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
  VisitArray(op->vectors, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const prim::BroadcastNode* op) { this->VisitExpr(op->value); }

Expr ExprMutator::VisitExpr_(const VarNode* op) { return ffi::GetRef<Var>(op); }

Expr ExprMutator::VisitExpr_(const TensorLoadNode* op) {
  auto fmutate = [this](const PrimExpr& e) { return this->VisitPrimExpr(e); };
  ffi::Array<PrimExpr> indices = op->indices.Map(fmutate);
  if (indices.same_as(op->indices)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return BufferLoad(op->source.as_or_throw<tvm::tirx::BufferVar>(), indices, op->span);
  }
}

Expr ExprMutator::VisitExpr_(const OpaqueExprNode* op) { return ffi::GetRef<OpaqueExpr>(op); }

Expr ExprMutator::VisitExpr_(const BufferRegionNode* op) {
  ffi::Array<Range> region = op->region.Map([this](const Range& range) {
    PrimExpr min = this->VisitPrimExpr(range->min);
    PrimExpr extent = this->VisitPrimExpr(range->extent);
    return min.same_as(range->min) && extent.same_as(range->extent)
               ? range
               : Range::FromMinExtent(std::move(min), std::move(extent));
  });
  return region.same_as(op->region) ? ffi::GetRef<BufferRegion>(op)
                                    : BufferRegion(op->buffer, std::move(region), op->span);
}

Expr ExprMutator::VisitExpr_(const TupleNode* op) {
  ffi::Array<Expr> fields =
      op->fields.Map([this](const Expr& field) { return this->VisitExpr(field); });
  return fields.same_as(op->fields) ? ffi::GetRef<tvm::Tuple>(op) : tvm::Tuple(fields, op->span);
}

Expr ExprMutator::VisitExpr_(const TupleGetItemNode* op) {
  Expr tuple_value = this->VisitExpr(op->tuple);
  return tuple_value.same_as(op->tuple) ? ffi::GetRef<TupleGetItem>(op)
                                        : TupleGetItem(std::move(tuple_value), op->index, op->span);
}

Expr ExprMutator::VisitExpr_(const prim::LetNode* op) {
  PrimExpr value = this->VisitPrimExpr(op->value);
  PrimExpr body = this->VisitPrimExpr(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return prim::Let(op->var, value, body);
  }
}

Expr ExprMutator::VisitExpr_(const CallNode* op) {
  Expr call_op = op->op;
  if (op->op.as<OpaqueExprNode>()) {
    call_op = this->VisitExpr(op->op);
  }
  ffi::Array<Expr> args =
      op->args.Map([this](const Expr& arg) -> Expr { return this->VisitExpr(arg); });

  if (call_op.same_as(op->op) && args.same_as(op->args)) {
    return ffi::GetRef<Call>(op);
  } else {
    Type result_type = op->ExprNode::ty;
    if (op->op.same_as(builtin::buffer_data())) {
      TVM_FFI_ICHECK_EQ(args.size(), 1);
      const auto* buffer_var = args[0].as<VarNode>();
      TVM_FFI_ICHECK(buffer_var);
      const auto* buffer_type = buffer_var->ty.as<BufferTypeNode>();
      TVM_FFI_ICHECK(buffer_type);
      result_type = buffer_type->DataPointerType();
    }
    return Call(result_type, call_op, args, op->attrs, op->ty_args, op->span);
  }
}

#define DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(OP) \
  Expr ExprMutator::VisitExpr_(const OP* op) { return ffi::GetRef<PrimExpr>(op); }

DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(IntImmNode)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(FloatImmNode)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(prim::StringImmNode)

#define DEFINE_BIOP_EXPR_MUTATE_(OP)                 \
  Expr ExprMutator::VisitExpr_(const OP##Node* op) { \
    PrimExpr a = this->VisitPrimExpr(op->a);         \
    PrimExpr b = this->VisitPrimExpr(op->b);         \
    if (a.same_as(op->a) && b.same_as(op->b)) {      \
      return ffi::GetRef<PrimExpr>(op);              \
    } else {                                         \
      return OP(a, b);                               \
    }                                                \
  }

DEFINE_BIOP_EXPR_MUTATE_(prim::Add);
DEFINE_BIOP_EXPR_MUTATE_(prim::Sub);
DEFINE_BIOP_EXPR_MUTATE_(prim::Mul);
DEFINE_BIOP_EXPR_MUTATE_(prim::Div);
DEFINE_BIOP_EXPR_MUTATE_(prim::Mod);
DEFINE_BIOP_EXPR_MUTATE_(prim::FloorDiv);
DEFINE_BIOP_EXPR_MUTATE_(prim::FloorMod);
DEFINE_BIOP_EXPR_MUTATE_(prim::Min);
DEFINE_BIOP_EXPR_MUTATE_(prim::Max);
DEFINE_BIOP_EXPR_MUTATE_(prim::EQ);
DEFINE_BIOP_EXPR_MUTATE_(prim::NE);
DEFINE_BIOP_EXPR_MUTATE_(prim::LT);
DEFINE_BIOP_EXPR_MUTATE_(prim::LE);
DEFINE_BIOP_EXPR_MUTATE_(prim::GT);
DEFINE_BIOP_EXPR_MUTATE_(prim::GE);
DEFINE_BIOP_EXPR_MUTATE_(prim::And);
DEFINE_BIOP_EXPR_MUTATE_(prim::Or);

Expr ExprMutator::VisitExpr_(const prim::CastNode* op) {
  PrimExpr value = this->VisitPrimExpr(op->value);
  if (value.same_as(op->value)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return prim::Cast(op->ExprNode::ty.as_or_throw<PrimType>(), value);
  }
}

Expr ExprMutator::VisitExpr_(const prim::NotNode* op) {
  PrimExpr a = this->VisitPrimExpr(op->a);
  if (a.same_as(op->a)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return prim::Not(a);
  }
}

Expr ExprMutator::VisitExpr_(const prim::SelectNode* op) {
  PrimExpr condition = this->VisitPrimExpr(op->condition);
  PrimExpr true_value = this->VisitPrimExpr(op->true_value);
  PrimExpr false_value = this->VisitPrimExpr(op->false_value);
  if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return prim::Select(condition, true_value, false_value);
  }
}

Expr ExprMutator::VisitExpr_(const prim::RampNode* op) {
  PrimExpr base = this->VisitPrimExpr(op->base);
  PrimExpr stride = this->VisitPrimExpr(op->stride);
  PrimExpr lanes = this->VisitPrimExpr(op->lanes);
  if (base.same_as(op->base) && stride.same_as(op->stride) && lanes.same_as(op->lanes)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return prim::Ramp(base, stride, lanes);
  }
}

Expr ExprMutator::VisitExpr_(const prim::BroadcastNode* op) {
  PrimExpr value = this->VisitPrimExpr(op->value);
  PrimExpr lanes = this->VisitPrimExpr(op->lanes);
  if (value.same_as(op->value) && lanes.same_as(op->lanes)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return prim::Broadcast(value, lanes);
  }
}

Expr ExprMutator::VisitExpr_(const prim::ShuffleNode* op) {
  auto fexpr = [this](const PrimExpr& e) { return this->VisitPrimExpr(e); };
  auto vectors = op->vectors.Map(fexpr);
  auto indices = op->indices.Map(fexpr);
  if (vectors.same_as(op->vectors) && indices.same_as(op->indices)) {
    return ffi::GetRef<PrimExpr>(op);
  } else {
    return prim::Shuffle(vectors, indices);
  }
}

}  // namespace tirx
}  // namespace tvm
