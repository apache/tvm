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
 * \file bf16_legalize.cc
 * \brief legalize bf16 type by adding cast_to_fp32
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include <cmath>
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

  Stmt operator()(Stmt s) { return VisitStmt(s); }

  std::tuple<PrimExpr, PrimExpr> DoCast(PrimExpr orig_a, PrimExpr orig_b, bool* is_bfloat16) {
    auto a = this->VisitExpr(orig_a);
    auto b = this->VisitExpr(orig_b);
    *is_bfloat16 = false;
    if (a->dtype.is_bfloat16()) {
      CHECK(b->dtype.is_bfloat16());
      *is_bfloat16 = true;
    } else if (b->dtype.is_bfloat16()) {
      CHECK(a->dtype.is_bfloat16());
      *is_bfloat16 = true;
    }

    if (*is_bfloat16) {
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

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)        \
  PrimExpr BF16PromoteRewriter::VisitExpr_(const OP* op) {       \
    PrimExpr a, b;                                               \
    bool is_bfloat16;                                            \
    std::tie(a, b) = DoCast(op->a, op->b, &is_bfloat16);         \
    if (a.same_as(op->a) && b.same_as(op->b)) {                  \
      return GetRef<PrimExpr>(op);                               \
    } else {                                                     \
      auto ret = FUNC(a, b);                                     \
      if (!is_bfloat16)                                          \
        return ret;                                              \
      else                                                       \
        return CastNode::make(DataType(kTVMBFloat, 16, 1), ret); \
    }                                                            \
  }

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(OP, FUNC) \
  PrimExpr BF16PromoteRewriter::VisitExpr_(const OP* op) {        \
    PrimExpr a, b;                                                \
    bool is_bfloat16;                                             \
    std::tie(a, b) = DoCast(op->a, op->b, &is_bfloat16);          \
    if (a.same_as(op->a) && b.same_as(op->b)) {                   \
      return GetRef<PrimExpr>(op);                                \
    } else {                                                      \
      auto ret = FUNC(a, b);                                      \
      return ret;                                                 \
    }                                                             \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(LTNode, operator<)   // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(LENode, operator<=)  // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(GTNode, operator>)   // NOLINT(*)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH_NO_CAST(GENode, operator>=)  // NOLINT(*)

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

  Stmt operator()(Stmt s) { return VisitStmt(s); }

  PrimExpr VisitExpr_(const CastNode* op) {
    auto op_val = StmtExprMutator::VisitExpr(op->value);
    if (op->dtype.is_float() && op->dtype.bits() == 32) {
      // if is cast_to_fp32, check if op->value is cast_to_fp16
      // and op->value->value is a float32
      if (auto innercast = op_val.as<CastNode>()) {
        if (innercast->dtype.is_bfloat16() && innercast->value->dtype.is_float() &&
            innercast->value->dtype.bits() == 32) {
          return innercast->value;
        }
      }
    }
    if (op->value.same_as(op_val)) return GetRef<PrimExpr>(op);
    return CastNode::make(op->dtype, op_val);
  }
};

// implementation from
// https://github.com/pytorch/pytorch/blob/master/c10/util/BFloat16.h
inline uint16_t round_to_nearest_even(float src) {
#if defined(_MSC_VER)
  if (isnan(src)) {
#else
  if (std::isnan(src)) {
#endif
    return UINT16_C(0x7FC0);
  } else {
    union {
      uint32_t U32;
      float F32;
    };

    F32 = src;
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}

/*
 * Lower the bf16 type to int16
 * Lower cast between bf16 and fp32
 * Lower bf16 FloatImm to int16
 */
class BF16LowerRewriter : StmtExprMutator {
 public:
  BF16LowerRewriter() {}

  std::unordered_map<const BufferNode*, Buffer> buffer_remap;
  std::unordered_map<const VarNode*, Var> var_remap;

  Stmt operator()(Stmt s) { return VisitStmt(s); }

  PrimExpr VisitExpr_(const CastNode* op) {
    auto op_val = StmtExprMutator::VisitExpr(op->value);
    if (op->value->dtype.is_bfloat16()) {
      // if is cast_from_bf16, check if is to fp32
      CHECK(op->dtype.is_float() && op->dtype.bits() == 32);
      auto uint32_dtype = DataType(kDLUInt, 32, op_val->dtype.lanes());
      auto uint32_v = CastNode::make(uint32_dtype, op_val);
      return CallNode::make(op->dtype, CallNode::reinterpret, {uint32_v << 16},
                            CallNode::PureIntrinsic);

    } else if (op->dtype.is_bfloat16()) {
      // if is cast_to_bf16, check if op->value is fp32
      CHECK(op->value->dtype.is_float() && op->value->dtype.bits() == 32);
      auto uint32_dtype = DataType(kDLUInt, 32, op_val->dtype.lanes());
      auto uint32_v =
          CallNode::make(uint32_dtype, CallNode::reinterpret, {op_val}, CallNode::PureIntrinsic);
      auto uint16_dtype = DataType(kDLUInt, 16, op_val->dtype.lanes());
      // uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
      // return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
      auto rounding_bias = ((uint32_v >> 16) & 1) + make_const(uint16_dtype, 0x7FFF);
      return CastNode::make(uint16_dtype, {(uint32_v + rounding_bias) >> 16});
    }
    if (op->value.same_as(op_val)) return GetRef<PrimExpr>(op);
    return CastNode::make(op->dtype, op_val);
  }

  PrimExpr VisitExpr_(const VarNode* op) {
    auto itr = var_remap.find(op);
    if (itr != var_remap.end()) {
      return itr->second;
    }
    if (op->dtype.is_bfloat16()) {
      CHECK(!op->type_annotation.defined());
      auto ret = Var(op->name_hint, op->dtype);
      var_remap[op] = ret;
      return ret;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const AllocateNode* op) {
    Stmt node_holder;
    const AllocateNode* newop;
    if (op->dtype.is_bfloat16()) {
      auto v = AllocateNode::make(op->buffer_var, DataType::UInt(16, op->dtype.lanes()),
                                  op->extents, op->condition, op->body);
      node_holder = v;
      newop = static_cast<const AllocateNode*>(v.operator->());
    } else {
      newop = op;
    }
    return StmtExprMutator::VisitStmt_(newop);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) {
    auto itr = buffer_remap.find(op->buffer.operator->());
    const BufferStoreNode* newop;
    BufferStore newop_holder;
    if (itr != buffer_remap.end()) {
      newop_holder = BufferStore(itr->second, op->value, op->indices);
      newop = newop_holder.operator->();
    } else {
      newop = op;
    }
    return StmtExprMutator::VisitStmt_(newop);
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) {
    auto itr = buffer_remap.find(op->buffer.operator->());
    const BufferRealizeNode* newop;
    Stmt newop_holder;
    if (itr != buffer_remap.end()) {
      auto v = BufferRealize(itr->second, op->bounds, op->condition, op->body);
      newop_holder = v;
      newop = v.operator->();
    } else {
      newop = op;
    }
    return StmtExprMutator::VisitStmt_(newop);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    auto itr = buffer_remap.find(op->buffer.operator->());
    const BufferLoadNode* newop;
    BufferLoad newop_holder;
    if (itr != buffer_remap.end()) {
      newop_holder = BufferLoad(itr->second, op->indices);
      newop = newop_holder.operator->();
    } else {
      newop = op;
    }
    return StmtExprMutator::VisitExpr_(newop);
  }

  PrimExpr VisitExpr_(const LoadNode* op) override {
    bool is_bf16 = false;
    if (op->dtype.is_bfloat16()) {
      is_bf16 = true;
    }
    PrimExpr index = this->VisitExpr(op->index);
    PrimExpr predicate = this->VisitExpr(op->predicate);
    if (index.same_as(op->index) && predicate.same_as(op->predicate) && !is_bf16) {
      return GetRef<PrimExpr>(op);
    } else {
      return LoadNode::make(is_bf16 ? DataType::UInt(16, op->dtype.lanes()) : op->dtype,
                            op->buffer_var, index, predicate);
    }
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) override {
    if (op->dtype.is_bfloat16()) {
      return IntImm(DataType::UInt(16, op->dtype.lanes()),
                    round_to_nearest_even(static_cast<float>(op->value)));
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  void alter_buffers(PrimFuncNode* op) {
    std::vector<std::pair<Var, Buffer>> changes;
    for (auto& itr : op->buffer_map) {
      auto oldbuf = itr.second;
      if (oldbuf->dtype.is_bfloat16()) {
        auto newbuf =
            BufferNode::make(oldbuf->data, DataType::UInt(16, oldbuf->dtype.lanes()), oldbuf->shape,
                             oldbuf->strides, oldbuf->elem_offset, oldbuf->name, oldbuf->scope,
                             oldbuf->data_alignment, oldbuf->offset_factor, oldbuf->buffer_type);
        buffer_remap[oldbuf.operator->()] = newbuf;
        changes.emplace_back(itr.first, newbuf);
      }
    }
    if (buffer_remap.size() != 0) {
      op->buffer_map.assign(changes.begin(), changes.end());
    }
  }
};

namespace transform {

Pass BF16Promote() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BF16PromoteRewriter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BF16Promote", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BF16Promote").set_body_typed(BF16Promote);

Pass BF16CastElimination() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = BF16CastEliminationRewriter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BF16CastElimination", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BF16CastElimination").set_body_typed(BF16CastElimination);

Pass BF16TypeLowering() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    BF16LowerRewriter lowerer;
    lowerer.alter_buffers(n);
    n->body = lowerer(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BF16TypeLowering", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BF16TypeLowering").set_body_typed(BF16TypeLowering);

Pass BF16Legalize() {
  return Sequential({BF16Promote(), BF16CastElimination(), BF16TypeLowering()}, "tir.BF16Legalize");
}

TVM_REGISTER_GLOBAL("tir.transform.BF16Legalize").set_body_typed(BF16Legalize);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
