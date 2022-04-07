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
#include <tvm/tir/builtin.h>
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

#define DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC, NEEDCAST)                \
  PrimExpr BF16PromoteRewriter::VisitExpr_(const OP* op) {                         \
    PrimExpr origin_a = this->VisitExpr(op->a);                                    \
    PrimExpr origin_b = this->VisitExpr(op->b);                                    \
    bool a_is_bfloat16 = origin_a->dtype.is_bfloat16();                            \
    bool b_is_bfloat16 = origin_b->dtype.is_bfloat16();                            \
    bool both_bfloat16 = a_is_bfloat16 && b_is_bfloat16;                           \
    bool none_bfloat16 = !(a_is_bfloat16 || b_is_bfloat16);                        \
    if (none_bfloat16) {                                                           \
      return GetRef<PrimExpr>(op);                                                 \
    }                                                                              \
    DataType float32_dtype(kDLFloat, 32, 1);                                       \
    PrimExpr float32_a = a_is_bfloat16 ? Cast(float32_dtype, origin_a) : origin_a; \
    PrimExpr float32_b = b_is_bfloat16 ? Cast(float32_dtype, origin_b) : origin_b; \
    PrimExpr result = FUNC(float32_a, float32_b);                                  \
    DataType bfloat16_dtype(kDLBfloat, 16, 1);                                     \
    bool do_cast = both_bfloat16 && NEEDCAST;                                      \
    return do_cast ? Cast(bfloat16_dtype, result) : result;                        \
  }

DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(AddNode, operator+, true)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(SubNode, operator-, true)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MulNode, operator*, true)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(DivNode, div, true)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MinNode, min, true)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(MaxNode, max, true)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LTNode, operator<, false)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(LENode, operator<=, false)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GTNode, operator>, false)
DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(GENode, operator>=, false)

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

  PrimExpr VisitExpr_(const CastNode* op) final {
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
    return Cast(op->dtype, op_val);
  }
};

union FloatCaster {
  uint32_t u32;
  float f32;
};

uint16_t RoundToNearestEven(float src) {
  if (std::isnan(src)) {
    return UINT16_C(0x7FC0);
  } else {
    FloatCaster caster;
    caster.f32 = src;
    uint32_t rounding_bias = ((caster.u32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((caster.u32 + rounding_bias) >> 16);
  }
}

/*
 * Lower the bf16 type to int16
 * Lower cast between bf16 and fp32
 * Lower bf16 FloatImm to int16
 */
class BF16LowerRewriter : public StmtExprMutator {
 public:
  BF16LowerRewriter() {}

  using StmtExprMutator::operator();

  PrimExpr VisitExpr_(const CastNode* op) final {
    PrimExpr op_val = StmtExprMutator::VisitExpr(op->value);
    DataType uint32_dtype(kDLUInt, 32, op_val->dtype.lanes());
    DataType float32_dtype(kDLFloat, 32, op_val->dtype.lanes());
    if (op->value->dtype.is_bfloat16()) {  // cast from bf16
      PrimExpr uint32_v = Cast(uint32_dtype, op_val);
      PrimExpr float32_v = Call(float32_dtype, builtin::reinterpret(), {uint32_v << 16});
      bool is_to_float32 = op->dtype.is_float() && op->dtype.bits() == 32;
      return is_to_float32 ? float32_v : Cast(op->dtype, float32_v);
    } else if (op->dtype.is_bfloat16()) {  // cast to bf16
      bool is_from_float32 = op->value->dtype.is_float() && op->value->dtype.bits() == 32;
      PrimExpr float32_v = is_from_float32 ? op_val : Cast(float32_dtype, op_val);
      PrimExpr uint32_v = Call(uint32_dtype, builtin::reinterpret(), {float32_v});
      DataType uint16_dtype(kDLUInt, 16, op_val->dtype.lanes());
      /* the following TIR is equivalent to the C++ code below:
      uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
      return static_cast<uint16_t>((U32 + rounding_bias) >> 16);*/
      PrimExpr rounding_bias = ((uint32_v >> 16) & 1) + make_const(uint16_dtype, 0x7FFF);
      return Cast(uint16_dtype, {(uint32_v + rounding_bias) >> 16});
    }
    if (op->value.same_as(op_val)) return GetRef<PrimExpr>(op);
    return Cast(op->dtype, op_val);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    auto itr = var_remap_.find(var);
    if (itr != var_remap_.end()) {
      return itr->second;
    } else {
      return std::move(var);
    }
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    if (op->dtype.is_bfloat16()) {
      DataType dtype = DataType::UInt(16, op->dtype.lanes());
      Var buffer_var = Var(op->buffer_var->name_hint, PointerType(PrimType(dtype)));
      var_remap_[op->buffer_var] = buffer_var;
      return VisitStmt(Allocate(buffer_var, dtype, op->extents, op->condition, op->body));
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<BufferStoreNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return BufferStore(new_buf, op->value, op->indices);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();

    if (auto* buffer = op->node.as<BufferNode>()) {
      auto it = buffer_remap_.find(GetRef<Buffer>(buffer));
      if (it != buffer_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    } else if (auto* var = op->node.as<VarNode>()) {
      auto it = var_remap_.find(GetRef<Var>(var));
      if (it != var_remap_.end()) {
        return AttrStmt(it->second, op->attr_key, op->value, op->body);
      }
    }
    return ret;
  }

  Stmt VisitStmt_(const BufferRealizeNode* op) final {
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<BufferRealizeNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return BufferRealize(new_buf, op->bounds, op->condition, op->body);
    }
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated StoreNode.  Please use BufferStoreNode instead.";
    return Stmt();
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<BufferLoadNode>();

    Buffer new_buf = GetRemappedBuffer(op->buffer);
    if (new_buf.same_as(op->buffer)) {
      return ret;
    } else {
      return BufferLoad(new_buf, op->indices);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    LOG(FATAL) << "Unexpected use of deprecated LoadNode.  Please use BufferLoadNode instead.";
    return PrimExpr();
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) final {
    if (op->dtype.is_bfloat16()) {
      return IntImm(DataType::UInt(16, op->dtype.lanes()),
                    RoundToNearestEven(static_cast<float>(op->value)));
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  void AlterBuffers(PrimFuncNode* op) {
    Map<Var, Buffer> new_buffer_map;

    for (auto& itr : op->buffer_map) {
      auto param_var = itr.first;
      auto oldbuf = itr.second;
      if (oldbuf->dtype.is_bfloat16()) {
        DataType dtype = DataType::UInt(16, oldbuf->dtype.lanes());
        Var buffer_var = Var(oldbuf->data->name_hint, PointerType(PrimType(dtype)));
        auto newbuf = Buffer(buffer_var, dtype, oldbuf->shape, oldbuf->strides, oldbuf->elem_offset,
                             oldbuf->name, oldbuf->data_alignment, oldbuf->offset_factor,
                             oldbuf->buffer_type);
        buffer_remap_[oldbuf] = newbuf;
        var_remap_[oldbuf->data] = buffer_var;
        new_buffer_map.Set(param_var, newbuf);
      } else {
        new_buffer_map.Set(param_var, oldbuf);
      }
    }

    if (buffer_remap_.size() != 0) {
      op->buffer_map = new_buffer_map;
    }
  }

 private:
  Buffer GetRemappedBuffer(Buffer buf) {
    auto buf_it = buffer_remap_.find(buf);
    if (buf_it != buffer_remap_.end()) {
      return buf_it->second;
    }

    Buffer new_buf = buf;

    auto var_it = var_remap_.find(buf->data);
    if (var_it != var_remap_.end()) {
      DataType dtype =
          buf->dtype.is_bfloat16() ? DataType::UInt(16, buf->dtype.lanes()) : buf->dtype;
      new_buf = Buffer(var_it->second, dtype, buf->shape, buf->strides, buf->elem_offset, buf->name,
                       buf->data_alignment, buf->offset_factor, buf->buffer_type,
                       buf->axis_separators, buf->span);
    }

    buffer_remap_[buf] = new_buf;

    return new_buf;
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> var_remap_;
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
    lowerer.AlterBuffers(n);
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
