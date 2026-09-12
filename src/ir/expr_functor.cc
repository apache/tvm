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
#include <tvm/ir/expr_functor.h>

namespace tvm {

void ExprVisitor::InitVTable(VTable* vtable) {
  ObjectVisitor::InitVTable(vtable);
  SetDispatch<ExprVisitor, OpaqueExprNode>(vtable);
  SetDispatch<ExprVisitor, TupleNode>(vtable);
  SetDispatch<ExprVisitor, TupleGetItemNode>(vtable);
  SetDispatch<ExprVisitor, TensorLoadNode>(vtable);
  SetDispatch<ExprVisitor, VarNode>(vtable);
  SetDispatch<ExprVisitor, GlobalVarNode>(vtable);
  SetDispatch<ExprVisitor, CallNode>(vtable);
  SetDispatch<ExprVisitor, IntImmNode>(vtable);
  SetDispatch<ExprVisitor, FloatImmNode>(vtable);
  SetDispatch<ExprVisitor, OpNode>(vtable);
  SetDispatch<ExprVisitor, prim::StringImmNode>(vtable);
  SetDispatch<ExprVisitor, prim::CastNode>(vtable);
  SetDispatch<ExprVisitor, prim::AddNode>(vtable);
  SetDispatch<ExprVisitor, prim::SubNode>(vtable);
  SetDispatch<ExprVisitor, prim::MulNode>(vtable);
  SetDispatch<ExprVisitor, prim::DivNode>(vtable);
  SetDispatch<ExprVisitor, prim::ModNode>(vtable);
  SetDispatch<ExprVisitor, prim::FloorDivNode>(vtable);
  SetDispatch<ExprVisitor, prim::FloorModNode>(vtable);
  SetDispatch<ExprVisitor, prim::MinNode>(vtable);
  SetDispatch<ExprVisitor, prim::MaxNode>(vtable);
  SetDispatch<ExprVisitor, prim::EQNode>(vtable);
  SetDispatch<ExprVisitor, prim::NENode>(vtable);
  SetDispatch<ExprVisitor, prim::LTNode>(vtable);
  SetDispatch<ExprVisitor, prim::LENode>(vtable);
  SetDispatch<ExprVisitor, prim::GTNode>(vtable);
  SetDispatch<ExprVisitor, prim::GENode>(vtable);
  SetDispatch<ExprVisitor, prim::AndNode>(vtable);
  SetDispatch<ExprVisitor, prim::OrNode>(vtable);
  SetDispatch<ExprVisitor, prim::NotNode>(vtable);
  SetDispatch<ExprVisitor, prim::SelectNode>(vtable);
  SetDispatch<ExprVisitor, prim::LetNode>(vtable);
  SetDispatch<ExprVisitor, prim::RampNode>(vtable);
  SetDispatch<ExprVisitor, prim::BroadcastNode>(vtable);
  SetDispatch<ExprVisitor, prim::ShuffleNode>(vtable);
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const OpaqueExprNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->ty));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const TupleNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->fields));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const TupleGetItemNode* node) {
  // skips: index
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->tuple));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const TensorLoadNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->indices));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const VarNode* node) {
  // Primitive types have no children; dynamic type fields are visited.
  if (!node->ty.as<PrimTypeNode>()) {
    // Clamp Simple for dynamic type fields; Pattern continues through them.
    if (this->def_region_kind() == kTVMFFIDefRegionKindSimple) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
          kTVMFFIDefRegionKindNone, [&]() { return this->VisitExpected(node->ty); }));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->ty));
    }
  }
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const GlobalVarNode* node) {
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const CallNode* node) {
  // Skip constant attrs and primitive result types.
  if (!node->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->ty));
  }
  // Interned operators have no children; function-valued operators are visited.
  if (!node->op.as<OpNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->op));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->args));
  // Empty type arguments are skipped, including the container callback.
  if (!node->ty_args.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->ty_args));
  }
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const IntImmNode* node) {
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const FloatImmNode* node) {
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const OpNode* node) {
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::StringImmNode* node) {
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::CastNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->value));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::AddNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::SubNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::MulNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::DivNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::ModNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::FloorDivNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::FloorModNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::MinNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::MaxNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::EQNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::NENode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::LTNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::LENode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::GTNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::GENode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::AndNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::OrNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->b));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::NotNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->a));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::SelectNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->true_value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->false_value));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::LetNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->VisitExpected(node->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->body));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::RampNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->base));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->stride));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->lanes));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::BroadcastNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->lanes));
  return std::nullopt;
}

Expected<ffi::Optional<VisitInterrupt>> ExprVisitor::Visit_(const prim::ShuffleNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->vectors));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->VisitExpected(node->indices));
  return std::nullopt;
}

void ExprMutator::InitVTable(VTable* vtable) {
  ObjectMutator::InitVTable(vtable);
  SetDispatch<ExprMutator, OpaqueExprNode>(vtable);
  SetDispatch<ExprMutator, TupleNode>(vtable);
  SetDispatch<ExprMutator, TupleGetItemNode>(vtable);
  SetDispatch<ExprMutator, TensorLoadNode>(vtable);
  SetDispatch<ExprMutator, VarNode>(vtable);
  SetDispatch<ExprMutator, GlobalVarNode>(vtable);
  SetDispatch<ExprMutator, CallNode>(vtable);
  SetDispatch<ExprMutator, IntImmNode>(vtable);
  SetDispatch<ExprMutator, FloatImmNode>(vtable);
  SetDispatch<ExprMutator, OpNode>(vtable);
  SetDispatch<ExprMutator, prim::StringImmNode>(vtable);
  SetDispatch<ExprMutator, prim::CastNode>(vtable);
  SetDispatch<ExprMutator, prim::AddNode>(vtable);
  SetDispatch<ExprMutator, prim::SubNode>(vtable);
  SetDispatch<ExprMutator, prim::MulNode>(vtable);
  SetDispatch<ExprMutator, prim::DivNode>(vtable);
  SetDispatch<ExprMutator, prim::ModNode>(vtable);
  SetDispatch<ExprMutator, prim::FloorDivNode>(vtable);
  SetDispatch<ExprMutator, prim::FloorModNode>(vtable);
  SetDispatch<ExprMutator, prim::MinNode>(vtable);
  SetDispatch<ExprMutator, prim::MaxNode>(vtable);
  SetDispatch<ExprMutator, prim::EQNode>(vtable);
  SetDispatch<ExprMutator, prim::NENode>(vtable);
  SetDispatch<ExprMutator, prim::LTNode>(vtable);
  SetDispatch<ExprMutator, prim::LENode>(vtable);
  SetDispatch<ExprMutator, prim::GTNode>(vtable);
  SetDispatch<ExprMutator, prim::GENode>(vtable);
  SetDispatch<ExprMutator, prim::AndNode>(vtable);
  SetDispatch<ExprMutator, prim::OrNode>(vtable);
  SetDispatch<ExprMutator, prim::NotNode>(vtable);
  SetDispatch<ExprMutator, prim::SelectNode>(vtable);
  SetDispatch<ExprMutator, prim::LetNode>(vtable);
  SetDispatch<ExprMutator, prim::RampNode>(vtable);
  SetDispatch<ExprMutator, prim::BroadcastNode>(vtable);
  SetDispatch<ExprMutator, prim::ShuffleNode>(vtable);
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const OpaqueExprNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<Type>, ty, this->MaybeInplaceMutateIfUniqueExpected(node->ty, allow_inplace));
  if (ty.UnchangedOrSameAs(node->ty)) return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<OpaqueExprNode*>(node);
    if (!ty.IsUnchanged()) writable->ty = std::move(ty).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<OpaqueExprNode>(*node);
    copy->ty = std::move(ty).ValueOrUnchanged(std::move(copy->ty));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const TupleNode* node, bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<Type>, ty, this->MaybeInplaceMutateIfUniqueExpected(node->ty, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<ffi::Array<Expr>>, fields,
      this->MaybeInplaceMutateIfUniqueExpected(node->fields, allow_inplace));
  if (ty.UnchangedOrSameAs(node->ty) && fields.UnchangedOrSameAs(node->fields))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<TupleNode*>(node);
    if (!ty.IsUnchanged()) writable->ty = std::move(ty).ValueUnchecked();
    if (!fields.IsUnchanged()) writable->fields = std::move(fields).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<TupleNode>(*node);
    copy->ty = std::move(ty).ValueOrUnchanged(std::move(copy->ty));
    copy->fields = std::move(fields).ValueOrUnchanged(std::move(copy->fields));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const TupleGetItemNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<Type>, ty, this->MaybeInplaceMutateIfUniqueExpected(node->ty, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Expr>, tuple,
                                    MaybeInplaceMutateIfUniqueExpected(node->tuple, allow_inplace));
  if (ty.UnchangedOrSameAs(node->ty) && tuple.UnchangedOrSameAs(node->tuple))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<TupleGetItemNode*>(node);
    if (!ty.IsUnchanged()) writable->ty = std::move(ty).ValueUnchecked();
    if (!tuple.IsUnchanged()) writable->tuple = std::move(tuple).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<TupleGetItemNode>(*node);
    copy->ty = std::move(ty).ValueOrUnchanged(std::move(copy->ty));
    copy->tuple = std::move(tuple).ValueOrUnchanged(std::move(copy->tuple));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const TensorLoadNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<Type>, ty, this->MaybeInplaceMutateIfUniqueExpected(node->ty, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<Expr>, source, MaybeInplaceMutateIfUniqueExpected(node->source, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<ffi::Array<PrimExpr>>, indices,
      this->MaybeInplaceMutateIfUniqueExpected(node->indices, allow_inplace));
  if (ty.UnchangedOrSameAs(node->ty) && source.UnchangedOrSameAs(node->source) &&
      indices.UnchangedOrSameAs(node->indices))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<TensorLoadNode*>(node);
    if (!ty.IsUnchanged()) writable->ty = std::move(ty).ValueUnchecked();
    if (!source.IsUnchanged()) writable->source = std::move(source).ValueUnchecked();
    if (!indices.IsUnchanged()) writable->indices = std::move(indices).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<TensorLoadNode>(*node);
    copy->ty = std::move(ty).ValueOrUnchanged(std::move(copy->ty));
    copy->source = std::move(source).ValueOrUnchanged(std::move(copy->source));
    copy->indices = std::move(indices).ValueOrUnchanged(std::move(copy->indices));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const GlobalVarNode* node,
                                                     bool allow_inplace) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const CallNode* node, bool allow_inplace) {
  UnchangedOr<Type> ty = ffi::Unchanged();
  if (!node->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        UnchangedOr<Type>, mapped_ty,
        this->MaybeInplaceMutateIfUniqueExpected(node->ty, allow_inplace));
    ty = std::move(mapped_ty);
  }
  UnchangedOr<Expr> op = ffi::Unchanged();
  if (!node->op.as<OpNode>()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Expr>, mapped_op,
                                      MaybeInplaceMutateIfUniqueExpected(node->op, allow_inplace));
    op = std::move(mapped_op);
  }
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<ffi::Array<Expr>>, args,
      this->MaybeInplaceMutateIfUniqueExpected(node->args, allow_inplace));
  UnchangedOr<ffi::Array<Type>> ty_args = ffi::Unchanged();
  if (!node->ty_args.empty()) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
        UnchangedOr<ffi::Array<Type>>, mapped_ty_args,
        this->MaybeInplaceMutateIfUniqueExpected(node->ty_args, allow_inplace));
    ty_args = std::move(mapped_ty_args);
  }
  if (ty.UnchangedOrSameAs(node->ty) && op.UnchangedOrSameAs(node->op) &&
      args.UnchangedOrSameAs(node->args) && ty_args.UnchangedOrSameAs(node->ty_args))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<CallNode*>(node);
    if (!ty.IsUnchanged()) writable->ty = std::move(ty).ValueUnchecked();
    if (!op.IsUnchanged()) writable->op = std::move(op).ValueUnchecked();
    if (!args.IsUnchanged()) writable->args = std::move(args).ValueUnchecked();
    if (!ty_args.IsUnchanged()) writable->ty_args = std::move(ty_args).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<CallNode>(*node);
    copy->ty = std::move(ty).ValueOrUnchanged(std::move(copy->ty));
    copy->op = std::move(op).ValueOrUnchanged(std::move(copy->op));
    copy->args = std::move(args).ValueOrUnchanged(std::move(copy->args));
    copy->ty_args = std::move(ty_args).ValueOrUnchanged(std::move(copy->ty_args));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const IntImmNode* node, bool allow_inplace) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const FloatImmNode* node, bool allow_inplace) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const OpNode* node, bool allow_inplace) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::StringImmNode* node,
                                                     bool allow_inplace) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::CastNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, value,
                                    MaybeInplaceMutateIfUniqueExpected(node->value, allow_inplace));
  if (value.UnchangedOrSameAs(node->value)) return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<prim::CastNode*>(node);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<prim::CastNode>(*node);
    copy->value = std::move(value).ValueOrUnchanged(std::move(copy->value));
    return ffi::Any(Expr(std::move(copy)));
  }
}

#define TVM_IR_BINARY_MUTATE_IMPL(Name)                                                            \
  Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::Name##Node* node,               \
                                                       bool allow_inplace) {                       \
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, a,                                    \
                                      MaybeInplaceMutateIfUniqueExpected(node->a, allow_inplace)); \
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, b,                                    \
                                      MaybeInplaceMutateIfUniqueExpected(node->b, allow_inplace)); \
    if (a.UnchangedOrSameAs(node->a) && b.UnchangedOrSameAs(node->b)) return ffi::Unchanged();     \
    if (allow_inplace) {                                                                           \
      auto* writable = const_cast<prim::Name##Node*>(node);                                        \
      if (!a.IsUnchanged()) writable->a = std::move(a).ValueUnchecked();                           \
      if (!b.IsUnchanged()) writable->b = std::move(b).ValueUnchecked();                           \
      return ffi::Unchanged();                                                                     \
    } else {                                                                                       \
      auto copy = ffi::make_object<prim::Name##Node>(*node);                                       \
      copy->a = std::move(a).ValueOrUnchanged(std::move(copy->a));                                 \
      copy->b = std::move(b).ValueOrUnchanged(std::move(copy->b));                                 \
      return ffi::Any(Expr(std::move(copy)));                                                      \
    }                                                                                              \
  }
TVM_IR_BINARY_MUTATE_IMPL(Add)
TVM_IR_BINARY_MUTATE_IMPL(Sub)
TVM_IR_BINARY_MUTATE_IMPL(Mul)
TVM_IR_BINARY_MUTATE_IMPL(Div)
TVM_IR_BINARY_MUTATE_IMPL(Mod)
TVM_IR_BINARY_MUTATE_IMPL(FloorDiv)
TVM_IR_BINARY_MUTATE_IMPL(FloorMod)
TVM_IR_BINARY_MUTATE_IMPL(Min)
TVM_IR_BINARY_MUTATE_IMPL(Max)
TVM_IR_BINARY_MUTATE_IMPL(EQ)
TVM_IR_BINARY_MUTATE_IMPL(NE)
TVM_IR_BINARY_MUTATE_IMPL(LT)
TVM_IR_BINARY_MUTATE_IMPL(LE)
TVM_IR_BINARY_MUTATE_IMPL(GT)
TVM_IR_BINARY_MUTATE_IMPL(GE)
TVM_IR_BINARY_MUTATE_IMPL(And)
TVM_IR_BINARY_MUTATE_IMPL(Or)
#undef TVM_IR_BINARY_MUTATE_IMPL

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::NotNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, a,
                                    MaybeInplaceMutateIfUniqueExpected(node->a, allow_inplace));
  if (a.UnchangedOrSameAs(node->a)) return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<prim::NotNode*>(node);
    if (!a.IsUnchanged()) writable->a = std::move(a).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<prim::NotNode>(*node);
    copy->a = std::move(a).ValueOrUnchanged(std::move(copy->a));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::SelectNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<PrimExpr>, condition,
      MaybeInplaceMutateIfUniqueExpected(node->condition, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<PrimExpr>, true_value,
      MaybeInplaceMutateIfUniqueExpected(node->true_value, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<PrimExpr>, false_value,
      MaybeInplaceMutateIfUniqueExpected(node->false_value, allow_inplace));
  if (condition.UnchangedOrSameAs(node->condition) &&
      true_value.UnchangedOrSameAs(node->true_value) &&
      false_value.UnchangedOrSameAs(node->false_value))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<prim::SelectNode*>(node);
    if (!condition.IsUnchanged()) writable->condition = std::move(condition).ValueUnchecked();
    if (!true_value.IsUnchanged()) writable->true_value = std::move(true_value).ValueUnchecked();
    if (!false_value.IsUnchanged()) writable->false_value = std::move(false_value).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<prim::SelectNode>(*node);
    copy->condition = std::move(condition).ValueOrUnchanged(std::move(copy->condition));
    copy->true_value = std::move(true_value).ValueOrUnchanged(std::move(copy->true_value));
    copy->false_value = std::move(false_value).ValueOrUnchanged(std::move(copy->false_value));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::LetNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<Var>, var, WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
        return MaybeInplaceMutateIfUniqueExpected(node->var, allow_inplace);
      }));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, value,
                                    MaybeInplaceMutateIfUniqueExpected(node->value, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, body,
                                    MaybeInplaceMutateIfUniqueExpected(node->body, allow_inplace));
  if (var.UnchangedOrSameAs(node->var) && value.UnchangedOrSameAs(node->value) &&
      body.UnchangedOrSameAs(node->body))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<prim::LetNode*>(node);
    if (!var.IsUnchanged()) writable->var = std::move(var).ValueUnchecked();
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    if (!body.IsUnchanged()) writable->body = std::move(body).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<prim::LetNode>(*node);
    copy->var = std::move(var).ValueOrUnchanged(std::move(copy->var));
    copy->value = std::move(value).ValueOrUnchanged(std::move(copy->value));
    copy->body = std::move(body).ValueOrUnchanged(std::move(copy->body));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::RampNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, base,
                                    MaybeInplaceMutateIfUniqueExpected(node->base, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<PrimExpr>, stride,
      MaybeInplaceMutateIfUniqueExpected(node->stride, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, lanes,
                                    MaybeInplaceMutateIfUniqueExpected(node->lanes, allow_inplace));
  if (base.UnchangedOrSameAs(node->base) && stride.UnchangedOrSameAs(node->stride) &&
      lanes.UnchangedOrSameAs(node->lanes))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<prim::RampNode*>(node);
    if (!base.IsUnchanged()) writable->base = std::move(base).ValueUnchecked();
    if (!stride.IsUnchanged()) writable->stride = std::move(stride).ValueUnchecked();
    if (!lanes.IsUnchanged()) writable->lanes = std::move(lanes).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<prim::RampNode>(*node);
    copy->base = std::move(base).ValueOrUnchanged(std::move(copy->base));
    copy->stride = std::move(stride).ValueOrUnchanged(std::move(copy->stride));
    copy->lanes = std::move(lanes).ValueOrUnchanged(std::move(copy->lanes));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::BroadcastNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, value,
                                    MaybeInplaceMutateIfUniqueExpected(node->value, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<PrimExpr>, lanes,
                                    MaybeInplaceMutateIfUniqueExpected(node->lanes, allow_inplace));
  if (value.UnchangedOrSameAs(node->value) && lanes.UnchangedOrSameAs(node->lanes))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<prim::BroadcastNode*>(node);
    if (!value.IsUnchanged()) writable->value = std::move(value).ValueUnchecked();
    if (!lanes.IsUnchanged()) writable->lanes = std::move(lanes).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<prim::BroadcastNode>(*node);
    copy->value = std::move(value).ValueOrUnchanged(std::move(copy->value));
    copy->lanes = std::move(lanes).ValueOrUnchanged(std::move(copy->lanes));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const prim::ShuffleNode* node,
                                                     bool allow_inplace) {
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<ffi::Array<PrimExpr>>, vectors,
      this->MaybeInplaceMutateIfUniqueExpected(node->vectors, allow_inplace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      UnchangedOr<ffi::Array<PrimExpr>>, indices,
      this->MaybeInplaceMutateIfUniqueExpected(node->indices, allow_inplace));
  if (vectors.UnchangedOrSameAs(node->vectors) && indices.UnchangedOrSameAs(node->indices))
    return ffi::Unchanged();
  if (allow_inplace) {
    auto* writable = const_cast<prim::ShuffleNode*>(node);
    if (!vectors.IsUnchanged()) writable->vectors = std::move(vectors).ValueUnchecked();
    if (!indices.IsUnchanged()) writable->indices = std::move(indices).ValueUnchecked();
    return ffi::Unchanged();
  } else {
    auto copy = ffi::make_object<prim::ShuffleNode>(*node);
    copy->vectors = std::move(vectors).ValueOrUnchanged(std::move(copy->vectors));
    copy->indices = std::move(indices).ValueOrUnchanged(std::move(copy->indices));
    return ffi::Any(Expr(std::move(copy)));
  }
}

Expected<UnchangedOr<ffi::Any>> ExprMutator::Mutate_(const VarNode* node, bool allow_inplace) {
  if (node->ty.as<PrimTypeNode>()) return ffi::Unchanged();
  if (TVM_FFI_PREDICT_TRUE(var_remap_.empty() && def_region_kind() == kTVMFFIDefRegionKindNone)) {
    return ffi::Unchanged();
  }
  Expected<ffi::Any> remap_result = VarRemapGetExpected(ffi::AnyView(node));
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(remap_result);
  if (ffi::details::ExpectedUnsafe::GetData(remap_result).type_index() !=
      ffi::TypeIndex::kTVMFFINone) {
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Expr>, mapped, std::move(remap_result));
    return mapped;
  }
  if (def_region_kind() == kTVMFFIDefRegionKindNone) return ffi::Unchanged();
  UnchangedOr<ffi::Any> result = ffi::Unchanged();
  ffi::Any mapped_value = ffi::Unchanged();
  // PrimType has no children; dynamic type fields inherit Pattern but are visited outside Simple.
  if (!node->ty.as<PrimTypeNode>()) {
    Expected<UnchangedOr<ffi::Any>> mapped_ty_result =
        def_region_kind() == kTVMFFIDefRegionKindSimple
            ? WithDefRegionKind(
                  kTVMFFIDefRegionKindNone,
                  [&] { return this->MaybeInplaceMutateIfUniqueExpected(node->ty, allow_inplace); })
            : this->MaybeInplaceMutateIfUniqueExpected(node->ty, allow_inplace);
    TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(UnchangedOr<Type>, mapped_ty, std::move(mapped_ty_result));
    if (!mapped_ty.UnchangedOrSameAs(node->ty)) {
      if (allow_inplace) {
        const_cast<VarNode*>(node)->ty = std::move(mapped_ty).ValueUnchecked();
        mapped_value = ffi::Any(node);
      } else {
        auto copy = ffi::make_object<VarNode>(*node);
        copy->ty = std::move(mapped_ty).ValueUnchecked();
        mapped_value = ffi::Any(std::move(copy));
      }
      result = mapped_value;
    }
  }
  if (!result.IsUnchanged() || def_region_kind() == kTVMFFIDefRegionKindPattern) {
    auto set_result = VarRemapSetExpected(ffi::AnyView(node), mapped_value);
    TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(set_result);
  }
  return result;
}

}  // namespace tvm
