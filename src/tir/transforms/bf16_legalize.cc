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
 * \file narrow_datatype.cc
 * \brief narrow the datatype of indexing vars
 */

#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>
#include <tvm/runtime/registry.h>
#include <tuple>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/ir_visitor_with_analyzer.h"

namespace tvm {
namespace tir {

using arith::Analyzer;
using arith::IRMutatorWithAnalyzer;

class BF16PromoteRewriter : public StmtExprMutator {
 public:
  BF16PromoteRewriter() {}

  Stmt operator()(Stmt s) {
    return VisitStmt(s);
  }

  std::tuple<PrimExpr, PrimExpr> DoCast(PrimExpr orig_a,
        PrimExpr orig_b, bool* is_bf16) {
    auto a = this->VisitExpr(orig_a);
    auto b = this->VisitExpr(orig_b);
    *is_bf16 = false;
    if (a->dtype.is_bf16()) {
        CHECK(b->dtype.is_bf16());
        *is_bf16 = true;
    } else if (b->dtype.is_bf16()) {
        CHECK(a->dtype.is_bf16());
        *is_bf16 = true;
    }

    if (is_bf16) {
        DataType fp32ty(kDLFloat, 32, 1);
        a = CastNode::make(fp32ty, a);
        b = CastNode::make(fp32ty, b);
    }
    return std::make_tuple(a, b);
  }

  PrimExpr VisitExpr_(const AddNode* op) final;
  PrimExpr VisitExpr_(const SubNode* op) final;
  PrimExpr VisitExpr_(const MulNode* op) final;
  PrimExpr VisitExpr_(const DivNode* op) final;
  PrimExpr VisitExpr_(const MinNode* op) final;
  PrimExpr VisitExpr_(const MaxNode* op) final;
  PrimExpr VisitExpr_(const LTNode* op) final;
  PrimExpr VisitExpr_(const LENode* op) final;
  PrimExpr VisitExpr_(const GTNode* op) final;
  PrimExpr VisitExpr_(const GENode* op) final;
};


#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)               \
  PrimExpr BF16PromoteRewriter::VisitExpr_(const OP* op) {              \
    PrimExpr a, b;                                                      \
    bool is_bf16;                                                       \
    std::tie(a, b) = DoCast(op->a, op->b, &is_bf16);                     \
    if (a.same_as(op->a) &&                                             \
        b.same_as(op->b)) {                                             \
        return GetRef<PrimExpr>(op);                                    \
    } else {                                                            \
        auto ret = FUNC(a, b);                                          \
        if (!is_bf16)                                                   \
            return ret;                                                 \
        else                                                            \
            return CastNode::make(DataType(kTVMBFloat, 16, 1), ret);    \
    }                                                                   \
  }

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(OP, FUNC)       \
  PrimExpr BF16PromoteRewriter::VisitExpr_(const OP* op) {              \
    PrimExpr a, b;                                                      \
    bool is_bf16;                                                       \
    std::tie(a, b) = DoCast(op->a, op->b, &is_bf16);                     \
    if (a.same_as(op->a) &&                                             \
        b.same_as(op->b)) {                                             \
        return GetRef<PrimExpr>(op);                                    \
    } else {                                                            \
        auto ret = FUNC(a, b);                                          \
        return ret;                                                     \
    }                                                                   \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(LTNode, operator <)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(LENode, operator<=)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(GTNode, operator >)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(GENode, operator>=)

/*
 * Eliminate verbose casting between fp32 and bf16
 * Checks if the AST has the pattern:
 *     castto32(castto16(some_fp32_op(...)))
 * The verbose casting is generated by BF16Promote for multiple
 * bf16 Ops in a row. e.g.:
 *  X[i] + Y[i] + T[i] =>
 *  bf16((float32(bf16((float32(X[i]) + float32(Y[i])))) + float32(T[i])))
 * After this pass:
 *  bf16(float32(X[i]) + float32(Y[i]) + float32(T[i]))
*/
class BF16CastEliminationRewriter : public StmtExprMutator {
 public:
  BF16CastEliminationRewriter() {}

  Stmt operator()(Stmt s) {
    return VisitStmt(s);
  }

  PrimExpr VisitExpr_(const CastNode* op) {
    auto op_val = StmtExprMutator::VisitExpr(op->value);
    if (op->dtype.is_float() && op->dtype.bits() == 32) {
        // if is cast_to_fp32, check if op->value is cast_to_fp16
        // and op->value->value is a float32
        if (auto innercast = op_val.as<CastNode>()) {
            if (innercast->dtype.is_bf16()
            && innercast->value->dtype.is_float()
            && innercast->value->dtype.bits() == 32) {
                return innercast->value;
            }
        }
    }
    if (op->value.same_as(op_val))
        return GetRef<PrimExpr>(op);
    return CastNode::make(op->dtype, op_val);
  }
};


namespace transform {

Pass BF16Promote() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BF16PromoteRewriter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(
      pass_func, 0, "tir.BF16Promote", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BF16Promote")
.set_body_typed(BF16Promote);

Pass BF16CastElimination() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BF16CastEliminationRewriter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(
      pass_func, 0, "tir.BF16CastElimination", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BF16CastElimination")
.set_body_typed(BF16CastElimination);

Pass BF16Legalize() {
  return Sequential({BF16Promote(), BF16CastElimination()},
      "tir.BF16Legalize");
}

TVM_REGISTER_GLOBAL("tir.transform.BF16Legalize")
.set_body_typed(BF16Legalize);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
