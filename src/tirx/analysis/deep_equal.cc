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
 * \file tirx/analysis/deep_equal.cc
 * \brief Deep equality checking.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr_functor.h>

namespace tvm {
namespace tirx {

#define DEFINE_DEEP_EQUAL_BIN_EXPR(OpNode)                                         \
  bool VisitExpr_(const OpNode* plhs, const PrimExpr& rhs) final {                 \
    const auto* prhs = rhs.as<OpNode>();                                           \
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() && \
           VisitExpr(plhs->a, prhs->a) && VisitExpr(plhs->b, prhs->b);             \
  }

#define DEFINE_DEEP_EQUAL_IMM_EXPR(OpNode)                                         \
  bool VisitExpr_(const OpNode* plhs, const PrimExpr& rhs) final {                 \
    const auto* prhs = rhs.as<OpNode>();                                           \
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() && \
           plhs->value == prhs->value;                                             \
  }

class ExprDeepEqualChecker : private ExprFunctor<bool(const Expr&, const PrimExpr&)> {
 public:
  static bool Check(const PrimExpr& lhs, const PrimExpr& rhs) {
    // quick path without constructing the object
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    if (auto* plhs = lhs.as<IntImmNode>()) {
      auto* prhs = rhs.as<IntImmNode>();
      return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
             plhs->value == prhs->value;
    }
    return ExprDeepEqualChecker().VisitExpr(lhs, rhs);
  }

  bool VisitExpr(const Expr& expr, const PrimExpr& rhs) final {
    PrimExpr lhs = expr.as_or_throw<PrimExpr>();
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() && rhs.defined()) return false;
    if (!rhs.defined() && lhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    return ExprFunctor::VisitExpr(lhs, rhs);
  }

  bool VisitExpr(const Expr& lhs, const Expr& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.defined() || !rhs.defined()) return false;
    if (lhs->type_index() != rhs->type_index()) return false;
    if (auto lhs_prim = lhs.as<PrimExpr>()) {
      auto rhs_prim = rhs.as<PrimExpr>();
      return rhs_prim && VisitExpr(lhs_prim.value(), rhs_prim.value());
    }
    if (lhs.as<VarNode>()) {
      return false;
    }
    if (auto* lhs_call = lhs.as<CallNode>()) {
      auto* rhs_call = rhs.as<CallNode>();
      return ffi::StructuralEqual()(lhs_call->ty, rhs_call->ty) &&
             lhs_call->op.same_as(rhs_call->op) && ArrayDeepEqual(lhs_call->args, rhs_call->args) &&
             ffi::StructuralEqual()(lhs_call->attrs, rhs_call->attrs);
    }
    return false;
  }

 private:
  bool ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!VisitExpr(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  bool ArrayDeepEqual(const ffi::Array<Expr>& lhs, const ffi::Array<Expr>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!VisitExpr(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  bool ArrayDeepEqual(const ffi::Array<IterVar>& lhs, const ffi::Array<IterVar>& rhs) {
    // for iter var, we require pointer equality
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!lhs[i].same_as(rhs[i])) return true;
    }
    return true;
  }

  bool OptionalDeepEqual(const ffi::Optional<PrimExpr>& lhs, const ffi::Optional<PrimExpr>& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (!lhs.has_value() && rhs.has_value()) return false;
    if (lhs.has_value() && !rhs.has_value()) return false;
    return VisitExpr(*lhs, *rhs);
  }

  bool VisitExpr_(const VarNode* plhs, const PrimExpr& rhs) final {
    // for var, we require pointer equality
    return plhs == rhs.get();
  }

  bool VisitExpr_(const BufferLoadNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<BufferLoadNode>();
    // we run pointer comparison of the buffer
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           plhs->buffer.same_as(prhs->buffer) && ArrayDeepEqual(plhs->indices, prhs->indices) &&
           OptionalDeepEqual(plhs->predicate, prhs->predicate);
  }

  bool VisitExpr_(const ProducerLoadNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<ProducerLoadNode>();
    // run shallow pointer comparison of the producer
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           plhs->producer.same_as(prhs->producer) && ArrayDeepEqual(plhs->indices, prhs->indices);
  }

  bool VisitExpr_(const LetNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<LetNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           VisitExpr(plhs->var, prhs->var) && VisitExpr(plhs->value, prhs->value) &&
           VisitExpr(plhs->body, prhs->body);
  }

  bool VisitExpr_(const CallNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<CallNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           plhs->op.same_as(prhs->op) && ArrayDeepEqual(plhs->args, prhs->args) &&
           ffi::StructuralEqual()(plhs->attrs, prhs->attrs);
  }

  bool VisitExpr_(const ReduceNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<ReduceNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           plhs->combiner.same_as(prhs->combiner) && ArrayDeepEqual(plhs->source, prhs->source) &&
           ArrayDeepEqual(plhs->init, prhs->init) && ArrayDeepEqual(plhs->axis, prhs->axis) &&
           VisitExpr(plhs->condition, prhs->condition) && plhs->value_index == prhs->value_index;
  }

  bool VisitExpr_(const CastNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<CastNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           VisitExpr(plhs->value, prhs->value);
  }

  bool VisitExpr_(const NotNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<NotNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           VisitExpr(plhs->a, prhs->a);
  }

  bool VisitExpr_(const SelectNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<SelectNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           VisitExpr(plhs->condition, prhs->condition) &&
           VisitExpr(plhs->true_value, prhs->true_value) &&
           VisitExpr(plhs->false_value, prhs->false_value);
  }

  bool VisitExpr_(const RampNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<RampNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           VisitExpr(plhs->base, prhs->base) && VisitExpr(plhs->stride, prhs->stride) &&
           VisitExpr(plhs->lanes, prhs->lanes);
  }

  bool VisitExpr_(const ShuffleNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<ShuffleNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           ArrayDeepEqual(plhs->vectors, prhs->vectors) &&
           ArrayDeepEqual(plhs->indices, prhs->indices);
  }

  bool VisitExpr_(const BroadcastNode* plhs, const PrimExpr& rhs) final {
    const auto* prhs = rhs.as<BroadcastNode>();
    return plhs->ty.as_or_throw<PrimType>() == prhs->ty.as_or_throw<PrimType>() &&
           VisitExpr(plhs->value, prhs->value) && VisitExpr(plhs->lanes, prhs->lanes);
  }

  DEFINE_DEEP_EQUAL_BIN_EXPR(AddNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(SubNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(MulNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(DivNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(ModNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(FloorDivNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(FloorModNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(MinNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(MaxNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(EQNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(NENode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(LTNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(LENode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(GTNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(GENode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(AndNode)
  DEFINE_DEEP_EQUAL_BIN_EXPR(OrNode)
  DEFINE_DEEP_EQUAL_IMM_EXPR(IntImmNode)
  DEFINE_DEEP_EQUAL_IMM_EXPR(FloatImmNode)
  DEFINE_DEEP_EQUAL_IMM_EXPR(StringImmNode)
};

bool ExprDeepEqual::operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
  return ExprDeepEqualChecker::Check(lhs, rhs);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.analysis.expr_deep_equal",
      [](const PrimExpr& lhs, const PrimExpr& rhs) { return ExprDeepEqual()(lhs, rhs); });
}

}  // namespace tirx
}  // namespace tvm
