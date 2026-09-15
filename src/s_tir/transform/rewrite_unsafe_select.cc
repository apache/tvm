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
 * \file unsafe_select_rewrite.cc
 * \brief Rewrite uinsafe select expression.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/s_tir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

// For now, rewrite unsafe select expression to if_then_else
// TODO(tqchen) pattern matching to support masked load
class UnsafeExprDetector : public tirx::ExprFunctor<bool(const Expr& n)> {
 public:
  // select itself is always considered safe if condition is safe
  // Because we will issue guard to make sure it is.
  bool Dispatch_(const SelectNode* op) { return Dispatch(op->condition); }
  bool Dispatch_(const CallNode* op) {
    if (op->op.same_as(prim::builtin::if_then_else())) {
      return Dispatch(op->args[0].as_or_throw<PrimExpr>());
    } else if (op->op.same_as(tirx::builtin::address_of())) {
      if (const auto* load = op->args[0].as<TensorLoadNode>()) {
        for (const auto& index : load->indices) {
          if (Dispatch(index)) {
            return true;
          }
        }
        return false;
      }
      return Dispatch(op->args[0]);
    } else if (auto opt = op->op.as<Op>()) {
      auto effect_kind = static_cast<CallEffectKind>(op_call_effect_[opt.value()]);
      if (effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation) {
        for (const Expr& arg : op->args) {
          if (Dispatch(arg)) return true;
        }
        return false;
      } else {
        return true;
      }
    } else {
      return true;
    }
  }
  bool Dispatch_(const TensorLoadNode* op) {
    // Load is considered unsafe.
    return true;
  }
  bool Dispatch_(const AddNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const SubNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const MulNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const DivNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const ModNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const FloorDivNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const FloorModNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const MinNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const MaxNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const EQNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const NENode* op) final { return BinaryOp(op); }
  bool Dispatch_(const LTNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const LENode* op) final { return BinaryOp(op); }
  bool Dispatch_(const GTNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const GENode* op) final { return BinaryOp(op); }
  bool Dispatch_(const AndNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const OrNode* op) final { return BinaryOp(op); }
  bool Dispatch_(const NotNode* op) final { return Dispatch(op->a); }
  bool Dispatch_(const LetNode* op) final { return Dispatch(op->body) || Dispatch(op->value); }
  bool Dispatch_(const CastNode* op) final { return Dispatch(op->value); }
  bool Dispatch_(const BroadcastNode* op) final { return Dispatch(op->value); }
  bool Dispatch_(const RampNode* op) final { return Dispatch(op->base) && Dispatch(op->stride); }
  bool Dispatch_(const ShuffleNode* op) final {
    for (PrimExpr e : op->vectors) {
      if (Dispatch(e)) return true;
    }
    return false;
  }
  bool Dispatch_(const VarNode* op) final { return false; }
  bool Dispatch_(const IntImmNode* op) final { return false; }
  bool Dispatch_(const FloatImmNode* op) final { return false; }
  bool Dispatch_(const StringImmNode* op) final { return false; }

 private:
  template <typename T>
  bool BinaryOp(const T* op) {
    return Dispatch(op->a) || Dispatch(op->b);
  }

  OpAttrMap<TCallEffectKind> op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
};

class UnsafeSelectRewriter : public StmtExprMutator {
 public:
  UnchangedOr<PrimExpr> Mutate_(const SelectNode* op, InplaceMode inplace_mode) {
    PrimExpr expr = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op)).as_or_throw<PrimExpr>();
    op = expr.as<SelectNode>();
    UnsafeExprDetector unsafe;
    PrimType cond_ty = op->condition.ty();
    bool cond_is_scalar_bool = cond_ty.MatchesCode(DLDataTypeCode::kDLBool) && cond_ty.IsScalar();
    if ((unsafe.Dispatch(op->true_value) || unsafe.Dispatch(op->false_value)) &&
        cond_is_scalar_bool) {
      return Call(op->ty.as_or_throw<PrimType>(), prim::builtin::if_then_else(),
                  {op->condition, op->true_value, op->false_value})
          .as_or_throw<PrimExpr>();
    } else {
      return expr;
    }
  }
};

Stmt RewriteUnsafeSelect(Stmt stmt) { return UnsafeSelectRewriter()(std::move(stmt)); }

namespace transform {

Pass RewriteUnsafeSelect() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = UnsafeSelectRewriter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.RewriteUnsafeSelect", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.RewriteUnsafeSelect", RewriteUnsafeSelect);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
