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
 * \file unwrap_vector_expr.cc
 * \brief Utility for tracking currently active constraints
 */

#include "unwrap_vector_expr.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/cast.h>
#include <tvm/ir/expr_functor.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace arith {

using namespace tirx;

class Scalarizer : public tvm::ExprMutator {
 public:
  explicit Scalarizer(PrimExpr lane) : lane_(lane) {}

#define TVM_SCALARIZER_BINARY_MUTATE_(Name)                                                        \
  UnchangedOr<PrimExpr> Mutate_(const prim::Name##Node* op, InplaceMode inplace_mode) final {      \
    return Rebuild(op, [](const prim::Name##Node* node) { return prim::Name(node->a, node->b); }); \
  }

  TVM_SCALARIZER_BINARY_MUTATE_(Add);
  TVM_SCALARIZER_BINARY_MUTATE_(Sub);
  TVM_SCALARIZER_BINARY_MUTATE_(Mul);
  TVM_SCALARIZER_BINARY_MUTATE_(Div);
  TVM_SCALARIZER_BINARY_MUTATE_(Mod);
  TVM_SCALARIZER_BINARY_MUTATE_(FloorDiv);
  TVM_SCALARIZER_BINARY_MUTATE_(FloorMod);
  TVM_SCALARIZER_BINARY_MUTATE_(Min);
  TVM_SCALARIZER_BINARY_MUTATE_(Max);
  TVM_SCALARIZER_BINARY_MUTATE_(EQ);
  TVM_SCALARIZER_BINARY_MUTATE_(NE);
  TVM_SCALARIZER_BINARY_MUTATE_(LT);
  TVM_SCALARIZER_BINARY_MUTATE_(LE);
  TVM_SCALARIZER_BINARY_MUTATE_(GT);
  TVM_SCALARIZER_BINARY_MUTATE_(GE);
  TVM_SCALARIZER_BINARY_MUTATE_(And);
  TVM_SCALARIZER_BINARY_MUTATE_(Or);

#undef TVM_SCALARIZER_BINARY_MUTATE_

  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [](const prim::CastNode* node) {
      return prim::Cast(node->ExprNode::ty.as_or_throw<PrimType>(), node->value);
    });
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NotNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [](const prim::NotNode* node) { return prim::Not(node->a); });
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::SelectNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [](const prim::SelectNode* node) {
      return prim::Select(node->condition, node->true_value, node->false_value);
    });
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::ShuffleNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [](const prim::ShuffleNode* node) {
      return prim::Shuffle(node->vectors, node->indices);
    });
  }

  UnchangedOr<Expr> Mutate_(const TupleNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [](const TupleNode* node) { return tvm::Tuple(node->fields, node->span); });
  }

  UnchangedOr<Expr> Mutate_(const TupleGetItemNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [](const TupleGetItemNode* node) {
      return TupleGetItem(node->tuple, node->index, node->span);
    });
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [](const TensorLoadNode* node) {
      return tirx::BufferLoad(node->source.as_or_throw<tirx::BufferVar>(), node->indices,
                              node->span);
    });
  }

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    return Rebuild(op, [op](const CallNode* node) -> Expr {
      if (!op->op.same_as(tirx::builtin::buffer_data()) || node->args.same_as(op->args)) {
        return ffi::GetRef<Call>(node);
      }
      TVM_FFI_ICHECK_EQ(node->args.size(), 1);
      const auto* buffer_var = node->args[0].as<VarNode>();
      TVM_FFI_ICHECK(buffer_var);
      const auto* buffer_type = buffer_var->ty.as<tirx::BufferTypeNode>();
      TVM_FFI_ICHECK(buffer_type);
      return Call(buffer_type->DataPointerType(), node->op, node->args, node->attrs, node->ty_args,
                  node->span);
    });
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::RampNode* op, InplaceMode inplace_mode) final {
    return op->base + lane_ * op->stride;
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::BroadcastNode* op, InplaceMode inplace_mode) final {
    return op->value;
  }

  UnchangedOr<Expr> Mutate_(const VarNode* op, InplaceMode inplace_mode) final {
    auto it = let_var_remap_.find(op);
    if (it != let_var_remap_.end()) {
      return it->second;
    } else {
      return tvm::ExprMutator::Mutate_(op, inplace_mode);
    }
  }
  UnchangedOr<PrimExpr> Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) final {
    PrimType value_ty = op->value.ty();
    if (value_ty.lanes() == 1) {
      return Rebuild(op, [](const prim::LetNode* node) {
        return prim::Let(node->var, node->value, node->body);
      });
    }

    auto it = let_var_remap_.find(op->var.get());
    TVM_FFI_ICHECK(it == let_var_remap_.end()) << "Duplicate binding of variable " << op->var;

    PrimType var_ty = op->var.as_or_throw<PrimVar>().ty();
    PrimVar new_var(op->var->name + "_scalar", var_ty.WithLanes(1));
    let_var_remap_[op->var.get()] = new_var;
    struct RemapGuard {
      std::unordered_map<const VarNode*, PrimVar>& remap;
      const VarNode* var;
      ~RemapGuard() { remap.erase(var); }
    } remap_guard{let_var_remap_, op->var.get()};

    PrimExpr value = Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value);
    PrimExpr body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);

    return prim::Let(op->var, value, body);
  }

 private:
  // Scalarized children can change a node's result type. Reuse native child mutation,
  // then the existing constructors to derive the type and check their invariants.
  // Disable in-place mutation so changed children cannot be reported as Unchanged.
  template <typename Node, typename F>
  auto Rebuild(const Node* op, F rebuild)
      -> decltype(tvm::ExprMutator::Mutate_(op, InplaceMode::kDisallow)) {
    auto rewritten_u = tvm::ExprMutator::Mutate_(op, InplaceMode::kDisallow);
    if (rewritten_u.IsUnchanged()) return ffi::Unchanged();
    auto rewritten = std::move(rewritten_u).ValueUnchecked();
    return rebuild(static_cast<const Node*>(rewritten.get()));
  }

  // The lane to extract
  PrimExpr lane_;

  // Let binding
  std::unordered_map<const VarNode*, PrimVar> let_var_remap_;
};

PrimExpr UnwrapVectorExpr(const PrimExpr& vector_expr, const PrimExpr& lane) {
  return ffi::make_object<Scalarizer>(lane)->Mutate(vector_expr).ValueOrUnchanged(vector_expr);
}

}  // namespace arith
}  // namespace tvm
