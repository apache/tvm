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

#include "simplify_base.h"

#include <tvm/ir/expr.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/op.h>

#include "constraint_helpers.h"

namespace tvm {
namespace arith {

using detail::EnterConstraintFacts;

UnchangedOr<Expr> SimplifierBase::Mutate_(const TupleNode* op, InplaceMode inplace_mode) {
  auto fields_u = Mutate(op->fields, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  if (fields_u.UnchangedOrSameAs(op->fields)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<TupleNode*>(op)->fields = std::move(fields_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TupleNode>(*op);
  copy->fields = std::move(fields_u).ValueUnchecked();
  return Tuple(std::move(copy));
}

UnchangedOr<Expr> SimplifierBase::Mutate_(const TupleGetItemNode* op, InplaceMode inplace_mode) {
  auto tuple_u = Mutate(op->tuple, inplace_mode);
  if (tuple_u.UnchangedOrSameAs(op->tuple)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<TupleGetItemNode*>(op)->tuple = std::move(tuple_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TupleGetItemNode>(*op);
  copy->tuple = std::move(tuple_u).ValueUnchecked();
  return TupleGetItem(std::move(copy));
}

UnchangedOr<PrimExpr> SimplifierBase::Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) {
  auto source_u = Mutate(op->source, inplace_mode);
  auto indices_u =
      Mutate(op->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  if (source_u.UnchangedOrSameAs(op->source) && indices_u.UnchangedOrSameAs(op->indices)) {
    return ffi::Unchanged();
  }
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TensorLoadNode*>(op);
    if (!source_u.IsUnchanged()) writable->source = std::move(source_u).ValueUnchecked();
    if (!indices_u.IsUnchanged()) writable->indices = std::move(indices_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TensorLoadNode>(*op);
  if (!source_u.IsUnchanged()) copy->source = std::move(source_u).ValueUnchecked();
  if (!indices_u.IsUnchanged()) copy->indices = std::move(indices_u).ValueUnchecked();
  return TensorLoad(std::move(copy));
}

UnchangedOr<Expr> SimplifierBase::Mutate_(const CallNode* op, InplaceMode inplace_mode) {
  if (op->op.same_as(prim::builtin::if_then_else())) {
    InplaceMode inplace_mode_args = inplace_mode;
    // Ensure uniqueness along op -> args -> args[i].
    // op was already checked; check args here, and Mutate checks args[i].
    if (!op->args.unique()) inplace_mode_args = InplaceMode::kDisallow;
    // Borrow stored elements: owning typed handles would suppress in-place mutation.
    const auto* args = op->args.GetArrayObj();
    PrimExpr cond =
        Mutate((*args)[0], inplace_mode_args).ValueOrUnchanged((*args)[0]).as_or_throw<PrimExpr>();
    Expr true_value = constraint_scope_
                          .WithNewScope([&]() {
                            EnterConstraintFacts(&constraint_scope_.Current(), analyzer_, cond);
                            return Mutate((*args)[1], inplace_mode_args);
                          })
                          .ValueOrUnchanged((*args)[1])
                          .as_or_throw<Expr>();
    Expr false_value;
    {
      PrimExpr not_cond = prim::Not(cond);
      false_value = constraint_scope_
                        .WithNewScope([&]() {
                          constraint_scope_.Current().Emplace(analyzer_, not_cond);
                          return Mutate((*args)[2], inplace_mode_args);
                        })
                        .ValueOrUnchanged((*args)[2])
                        .as_or_throw<Expr>();
    }
    if (prim::is_zero(cond)) return false_value;
    if (prim::is_one(cond)) return true_value;
    if (cond.same_as(op->args[0]) && true_value.same_as(op->args[1]) &&
        false_value.same_as(op->args[2])) {
      // Reuse the original node identity; there is no replacement to process.
      return ffi::Unchanged();
    }
    return Call(op->ty, op->op, {cond, true_value, false_value}, op->attrs, op->ty_args, op->span);
  }
  auto args_u = Mutate(op->args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  if (args_u.UnchangedOrSameAs(op->args)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<CallNode*>(op)->args = std::move(args_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<CallNode>(*op);
  copy->args = std::move(args_u).ValueUnchecked();
  return Call(std::move(copy));
}

UnchangedOr<PrimExpr> SimplifierBase::Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) {
  PrimExpr value = Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value);
  if (SideEffect(value) <= CallEffectKind::kPure) {
    analyzer_->Bind(op->var, value);
  }
  PrimExpr body = Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
  if (value.same_as(op->value) && body.same_as(op->body)) {
    // Reuse the original node identity; there is no replacement to process.
    return ffi::Unchanged();
  }
  return prim::Let(op->var, value, body, op->span);
}

UnchangedOr<PrimExpr> SimplifierBase::Mutate_(const prim::SelectNode* op,
                                              InplaceMode inplace_mode) {
  PrimExpr cond = Mutate(op->condition, inplace_mode).ValueOrUnchanged(op->condition);
  PrimExpr true_value = constraint_scope_
                            .WithNewScope([&]() {
                              EnterConstraintFacts(&constraint_scope_.Current(), analyzer_, cond);
                              return Mutate(op->true_value, inplace_mode);
                            })
                            .ValueOrUnchanged(op->true_value);
  PrimExpr false_value;
  {
    PrimExpr neg_cond = analyzer_->rewrite_simplify(prim::Not(cond));
    false_value = constraint_scope_
                      .WithNewScope([&]() {
                        constraint_scope_.Current().Emplace(analyzer_, neg_cond);
                        return Mutate(op->false_value, inplace_mode);
                      })
                      .ValueOrUnchanged(op->false_value);
  }
  if (prim::is_zero(cond)) return false_value;
  if (prim::is_one(cond)) return true_value;
  if (cond.same_as(op->condition) && true_value.same_as(op->true_value) &&
      false_value.same_as(op->false_value)) {
    // Reuse the original node identity; there is no replacement to process.
    return ffi::Unchanged();
  }
  return prim::Select(cond, true_value, false_value, op->span);
}

}  // namespace arith
}  // namespace tvm
