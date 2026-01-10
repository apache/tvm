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

#include "symbolic_matcher.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

void SymbolicMatcher::Match(const ffi::Array<PrimExpr>& params, const ffi::Array<PrimExpr>& args) {
  CHECK_EQ(params.size(), args.size());
  for (size_t i = 0; i < params.size(); ++i) {
    Match(params[i], args[i]);
  }
}

void SymbolicMatcher::Match(const PrimExpr& param, const PrimExpr& arg) {
  VisitExpr(param, arg);
  must_prove_ = analyzer_->Simplify(Substitute(must_prove_, *var_remap_));
  CHECK(!is_zero(must_prove_));
}

void SymbolicMatcher::VisitExpr(const PrimExpr& node, const PrimExpr& other) {
  if (node.same_as(other)) {
    return;
  } else if (node.dtype().code() != other.dtype().code()) {
    LOG(FATAL) << "Parameter expression " << node << " with dtype " << node.dtype()
               << " cannot match to argument " << other << " with dtype " << other.dtype();
  } else {
    ExprFunctor::VisitExpr(node, other);
  }
}

#define TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(OpName)                            \
  void SymbolicMatcher::VisitExpr_(const OpName* op, const PrimExpr& other) { \
    const auto* rhs = other.as<OpName>();                                     \
    if (rhs) {                                                                \
      VisitExpr(op->a, rhs->a);                                               \
      VisitExpr(op->b, rhs->b);                                               \
    } else {                                                                  \
      must_prove_ = must_prove_ && (ffi::GetRef<PrimExpr>(op) == other);      \
    }                                                                         \
  }

TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(AddNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(SubNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(MulNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(DivNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(ModNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(EQNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(NENode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(LTNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(LENode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(GTNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(GENode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(AndNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(OrNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(MinNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(MaxNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(FloorDivNode);
TVM_DECLARE_SYMBOLIC_MATCHER_BINOP(FloorModNode);

void SymbolicMatcher::VisitExpr_(const IntImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<IntImmNode>();
  if (!rhs || (op->value != rhs->value)) {
    LOG(FATAL) << "Parameter expression " << ffi::GetRef<PrimExpr>(op)
               << " expected an integer argument with value " << op->value << ", "
               << "but was provided with the argument " << other;
  }
}

void SymbolicMatcher::VisitExpr_(const FloatImmNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<FloatImmNode>();
  if (!rhs || (op->value != rhs->value)) {
    LOG(FATAL) << "Parameter expression " << ffi::GetRef<PrimExpr>(op)
               << " expected an float argument with value " << op->value << ", "
               << "but was provided with the argument " << other;
  }
}

void SymbolicMatcher::VisitExpr_(const CastNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<CastNode>();
  if (!rhs) {
    LOG(FATAL) << "Parameter expression " << ffi::GetRef<PrimExpr>(op) << " expected an cast to "
               << op->dtype << " as the argument, "
               << "but was provided with the argument " << other;
  }
  VisitExpr(op->value, rhs->value);
}

void SymbolicMatcher::VisitExpr_(const VarNode* op, const PrimExpr& rhs) {
  auto lhs = ffi::GetRef<Var>(op);

  if (lhs.same_as(rhs)) {
    // Reference identity, no further checks needed.
  } else if (op->dtype.code() != rhs->dtype.code()) {
    LOG(FATAL) << "Parameter expression " << ffi::GetRef<PrimExpr>(op) << " with dtype "
               << op->dtype << " cannot match to argument " << rhs << " with dtype " << rhs.dtype();
  } else if (auto it = var_remap_->find(lhs); it != var_remap_->end()) {
    VisitExpr((*it).second, rhs);
  } else {
    var_remap_->Set(lhs, rhs);
  }
}

void SymbolicMatcher::VisitExpr_(const SelectNode* op, const PrimExpr& other) {
  const auto* rhs = other.as<SelectNode>();
  if (rhs) {
    VisitExpr(op->condition, rhs->condition);
    VisitExpr(op->true_value, rhs->true_value);
    VisitExpr(op->false_value, rhs->false_value);
  } else {
    must_prove_ = must_prove_ && (ffi::GetRef<PrimExpr>(op) == other);
  }
}

}  // namespace tir
}  // namespace tvm
