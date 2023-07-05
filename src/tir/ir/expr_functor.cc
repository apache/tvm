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
#include <tvm/tir/expr_functor.h>

#include "functor_common.h"

namespace tvm {
namespace tir {

void ExprVisitor::VisitExpr_(const VarNode* op) {}

void ExprVisitor::VisitExpr_(const SizeVarNode* op) {
  this->VisitExpr_(static_cast<const VarNode*>(op));
}

void ExprVisitor::VisitExpr_(const AnyNode* op) {}

void ExprVisitor::VisitExpr_(const BufferLoadNode* op) {
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const ProducerLoadNode* op) {
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const LetNode* op) {
  this->VisitExpr(op->value);
  this->VisitExpr(op->body);
}

void ExprVisitor::VisitExpr_(const CallNode* op) {
  VisitArray(op->args, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

#define DEFINE_BINOP_VISIT_(OP)                \
  void ExprVisitor::VisitExpr_(const OP* op) { \
    this->VisitExpr(op->a);                    \
    this->VisitExpr(op->b);                    \
  }

DEFINE_BINOP_VISIT_(AddNode);
DEFINE_BINOP_VISIT_(SubNode);
DEFINE_BINOP_VISIT_(MulNode);
DEFINE_BINOP_VISIT_(DivNode);
DEFINE_BINOP_VISIT_(ModNode);
DEFINE_BINOP_VISIT_(FloorDivNode);
DEFINE_BINOP_VISIT_(FloorModNode);
DEFINE_BINOP_VISIT_(MinNode);
DEFINE_BINOP_VISIT_(MaxNode);
DEFINE_BINOP_VISIT_(EQNode);
DEFINE_BINOP_VISIT_(NENode);
DEFINE_BINOP_VISIT_(LTNode);
DEFINE_BINOP_VISIT_(LENode);
DEFINE_BINOP_VISIT_(GTNode);
DEFINE_BINOP_VISIT_(GENode);
DEFINE_BINOP_VISIT_(AndNode);
DEFINE_BINOP_VISIT_(OrNode);

void ExprVisitor::VisitExpr_(const IntImmNode* op) {}
void ExprVisitor::VisitExpr_(const FloatImmNode* op) {}
void ExprVisitor::VisitExpr_(const StringImmNode* op) {}

void ExprVisitor::VisitExpr_(const ReduceNode* op) {
  VisitArray(op->axis, [this](const IterVar& r) {
    this->VisitExpr(r->dom->min);
    this->VisitExpr(r->dom->extent);
  });
  VisitArray(op->source, [this](const PrimExpr& e) { this->VisitExpr(e); });
  if (!op->init.empty()) {
    VisitArray(op->init, [this](const PrimExpr& e) { this->VisitExpr(e); });
  }
  this->VisitExpr(op->condition);
}

void ExprVisitor::VisitExpr_(const CastNode* op) { this->VisitExpr(op->value); }

void ExprVisitor::VisitExpr_(const NotNode* op) { this->VisitExpr(op->a); }

void ExprVisitor::VisitExpr_(const SelectNode* op) {
  this->VisitExpr(op->condition);
  this->VisitExpr(op->true_value);
  this->VisitExpr(op->false_value);
}

void ExprVisitor::VisitExpr_(const RampNode* op) {
  this->VisitExpr(op->base);
  this->VisitExpr(op->stride);
}

void ExprVisitor::VisitExpr_(const ShuffleNode* op) {
  VisitArray(op->indices, [this](const PrimExpr& e) { this->VisitExpr(e); });
  VisitArray(op->vectors, [this](const PrimExpr& e) { this->VisitExpr(e); });
}

void ExprVisitor::VisitExpr_(const BroadcastNode* op) { this->VisitExpr(op->value); }

PrimExpr ExprMutator::VisitExpr_(const VarNode* op) { return GetRef<PrimExpr>(op); }

PrimExpr ExprMutator::VisitExpr_(const SizeVarNode* op) {
  return this->VisitExpr_(static_cast<const VarNode*>(op));
}

PrimExpr ExprMutator::VisitExpr_(const AnyNode* op) { return GetRef<PrimExpr>(op); }

PrimExpr ExprMutator::VisitExpr_(const BufferLoadNode* op) {
  auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  Array<PrimExpr> indices = op->indices.Map(fmutate);
  if (indices.same_as(op->indices)) {
    return GetRef<PrimExpr>(op);
  } else {
    return BufferLoad(op->buffer, indices);
  }
}

PrimExpr ExprMutator::VisitExpr_(const ProducerLoadNode* op) {
  auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  Array<PrimExpr> indices = op->indices.Map(fmutate);
  if (indices.same_as(op->indices)) {
    return GetRef<PrimExpr>(op);
  } else {
    return ProducerLoad(op->producer, indices);
  }
}

PrimExpr ExprMutator::VisitExpr_(const LetNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  PrimExpr body = this->VisitExpr(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Let(op->var, value, body);
  }
}

PrimExpr ExprMutator::VisitExpr_(const CallNode* op) {
  auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  Array<PrimExpr> args = op->args.Map(fmutate);

  if (args.same_as(op->args)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Call(op->dtype, op->op, args);
  }
}

#define DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(OP) \
  PrimExpr ExprMutator::VisitExpr_(const OP* op) { return GetRef<PrimExpr>(op); }

DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(IntImmNode)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(FloatImmNode)
DEFINE_OP_RETURN_SELF_EXPR_MUTATE_(StringImmNode)

#define DEFINE_BIOP_EXPR_MUTATE_(OP)                     \
  PrimExpr ExprMutator::VisitExpr_(const OP##Node* op) { \
    PrimExpr a = this->VisitExpr(op->a);                 \
    PrimExpr b = this->VisitExpr(op->b);                 \
    if (a.same_as(op->a) && b.same_as(op->b)) {          \
      return GetRef<PrimExpr>(op);                       \
    } else {                                             \
      return OP(a, b);                                   \
    }                                                    \
  }

DEFINE_BIOP_EXPR_MUTATE_(Add);
DEFINE_BIOP_EXPR_MUTATE_(Sub);
DEFINE_BIOP_EXPR_MUTATE_(Mul);
DEFINE_BIOP_EXPR_MUTATE_(Div);
DEFINE_BIOP_EXPR_MUTATE_(Mod);
DEFINE_BIOP_EXPR_MUTATE_(FloorDiv);
DEFINE_BIOP_EXPR_MUTATE_(FloorMod);
DEFINE_BIOP_EXPR_MUTATE_(Min);
DEFINE_BIOP_EXPR_MUTATE_(Max);
DEFINE_BIOP_EXPR_MUTATE_(EQ);
DEFINE_BIOP_EXPR_MUTATE_(NE);
DEFINE_BIOP_EXPR_MUTATE_(LT);
DEFINE_BIOP_EXPR_MUTATE_(LE);
DEFINE_BIOP_EXPR_MUTATE_(GT);
DEFINE_BIOP_EXPR_MUTATE_(GE);
DEFINE_BIOP_EXPR_MUTATE_(And);
DEFINE_BIOP_EXPR_MUTATE_(Or);

PrimExpr ExprMutator::VisitExpr_(const ReduceNode* op) {
  auto fitervar = [this](const IterVar& v) {
    Range r = v->dom;
    PrimExpr min = this->VisitExpr(r->min);
    PrimExpr extent = this->VisitExpr(r->extent);
    if (min.same_as(r->min) && extent.same_as(r->extent)) {
      return v;
    } else {
      return IterVar(Range::FromMinExtent(min, extent), v->var, v->iter_type, v->thread_tag);
    }
  };
  Array<IterVar> axis = op->axis.Map(fitervar);

  auto fexpr = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  Array<PrimExpr> source = op->source.Map(fexpr);
  Array<PrimExpr> init = op->init.Map(fexpr);

  PrimExpr condition = this->VisitExpr(op->condition);

  if (axis.same_as(op->axis) && source.same_as(op->source) && condition.same_as(op->condition) &&
      init.same_as(op->init)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Reduce(op->combiner, source, axis, condition, op->value_index, init);
  }
}

PrimExpr ExprMutator::VisitExpr_(const CastNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Cast(op->dtype, value);
  }
}

PrimExpr ExprMutator::VisitExpr_(const NotNode* op) {
  PrimExpr a = this->VisitExpr(op->a);
  if (a.same_as(op->a)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Not(a);
  }
}

PrimExpr ExprMutator::VisitExpr_(const SelectNode* op) {
  PrimExpr condition = this->VisitExpr(op->condition);
  PrimExpr true_value = this->VisitExpr(op->true_value);
  PrimExpr false_value = this->VisitExpr(op->false_value);
  if (condition.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Select(condition, true_value, false_value);
  }
}

PrimExpr ExprMutator::VisitExpr_(const RampNode* op) {
  PrimExpr base = this->VisitExpr(op->base);
  PrimExpr stride = this->VisitExpr(op->stride);
  if (base.same_as(op->base) && stride.same_as(op->stride)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Ramp(base, stride, op->lanes);
  }
}

PrimExpr ExprMutator::VisitExpr_(const BroadcastNode* op) {
  PrimExpr value = this->VisitExpr(op->value);
  if (value.same_as(op->value)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Broadcast(value, op->lanes);
  }
}

PrimExpr ExprMutator::VisitExpr_(const ShuffleNode* op) {
  auto fexpr = [this](const PrimExpr& e) { return this->VisitExpr(e); };
  auto vectors = op->vectors.Map(fexpr);
  if (vectors.same_as(op->vectors)) {
    return GetRef<PrimExpr>(op);
  } else {
    return Shuffle(vectors, op->indices);
  }
}

}  // namespace tir
}  // namespace tvm
