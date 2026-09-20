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
  SetDispatch<ExprVisitor, TensorRegionNode>(vtable);
  SetDispatch<ExprVisitor, VarNode>(vtable);
  SetDispatch<ExprVisitor, GlobalVarNode>(vtable);
  SetDispatch<ExprVisitor, CallNode>(vtable);
  SetDispatch<ExprVisitor, GenericConstNode>(vtable);
  SetDispatch<ExprVisitor, IntImmNode>(vtable);
  SetDispatch<ExprVisitor, FloatImmNode>(vtable);
  SetDispatch<ExprVisitor, OpNode>(vtable);
  SetDispatch<ExprVisitor, StringImmNode>(vtable);
  SetDispatch<ExprVisitor, prim::CastNode>(vtable);
  SetDispatch<ExprVisitor, prim::AddNode>(vtable);
  SetDispatch<ExprVisitor, prim::LShiftNode>(vtable);
  SetDispatch<ExprVisitor, prim::RShiftNode>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseAndNode>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseOrNode>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseXorNode>(vtable);
  SetDispatch<ExprVisitor, prim::BitwiseNotNode>(vtable);
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

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const OpaqueExprNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TupleNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->fields));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TupleGetItemNode* node) {
  // skips: index
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->tuple));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TensorLoadNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->indices));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const TensorRegionNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->source));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->region));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const VarNode* node) {
  // Primitive types have no children; dynamic type fields are visited.
  if (!node->ty.as<PrimTypeNode>()) {
    // Clamp Simple for dynamic type fields; Pattern continues through them.
    if (this->def_region_kind() == kTVMFFIDefRegionKindSimple) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
          kTVMFFIDefRegionKindNone, [&]() { return this->Visit(node->ty); }));
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
    }
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const GlobalVarNode* node) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const CallNode* node) {
  // Skip constant attrs and primitive result types.
  if (!node->ty.as<PrimTypeNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty));
  }
  // Interned operators have no children; function-valued operators are visited.
  if (!node->op.as<OpNode>()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->op));
  }
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->args));
  // Empty type arguments are skipped, including the container callback.
  if (!node->ty_args.empty()) {
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->ty_args));
  }
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const GenericConstNode* node) {
  return this->Visit(node->ty);
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const IntImmNode* node) { return std::nullopt; }

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const FloatImmNode* node) { return std::nullopt; }

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const OpNode* node) { return std::nullopt; }

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const StringImmNode* node) {
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::CastNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->value));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::AddNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LShiftNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::RShiftNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseAndNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseOrNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseXorNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::SubNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::MulNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::DivNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::ModNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::FloorDivNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::FloorModNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::MinNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::MaxNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::EQNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::NENode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LTNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LENode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::GTNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::GENode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::AndNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::OrNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->b));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::NotNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BitwiseNotNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->a));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::SelectNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->condition));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->true_value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->false_value));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::LetNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->WithDefRegionKind(
      kTVMFFIDefRegionKindSimple, [&]() { return this->Visit(node->var); }));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->body));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::RampNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->base));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->stride));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->lanes));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::BroadcastNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->value));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->lanes));
  return std::nullopt;
}

ffi::Optional<VisitInterrupt> ExprVisitor::Visit_(const prim::ShuffleNode* node) {
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->vectors));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(node->indices));
  return std::nullopt;
}

void ExprMutator::InitVTable(VTable* vtable) {
  ObjectMutator::InitVTable(vtable);
  SetDispatch<ExprMutator, OpaqueExprNode>(vtable);
  SetDispatch<ExprMutator, TupleNode>(vtable);
  SetDispatch<ExprMutator, TupleGetItemNode>(vtable);
  SetDispatch<ExprMutator, TensorLoadNode>(vtable);
  SetDispatch<ExprMutator, TensorRegionNode>(vtable);
  SetDispatch<ExprMutator, VarNode>(vtable);
  SetDispatch<ExprMutator, GlobalVarNode>(vtable);
  SetDispatch<ExprMutator, CallNode>(vtable);
  SetDispatch<ExprMutator, GenericConstNode>(vtable);
  SetDispatch<ExprMutator, IntImmNode>(vtable);
  SetDispatch<ExprMutator, FloatImmNode>(vtable);
  SetDispatch<ExprMutator, OpNode>(vtable);
  SetDispatch<ExprMutator, StringImmNode>(vtable);
  SetDispatch<ExprMutator, prim::CastNode>(vtable);
  SetDispatch<ExprMutator, prim::AddNode>(vtable);
  SetDispatch<ExprMutator, prim::LShiftNode>(vtable);
  SetDispatch<ExprMutator, prim::RShiftNode>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseAndNode>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseOrNode>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseXorNode>(vtable);
  SetDispatch<ExprMutator, prim::BitwiseNotNode>(vtable);
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

UnchangedOr<Expr> ExprMutator::Mutate_(const OpaqueExprNode* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  if (ty_u.UnchangedOrSameAs(node->ty)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<OpaqueExprNode*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<OpaqueExprNode>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::Mutate_(const TupleNode* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  auto fields_u = Mutate(node->fields, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  if (ty_u.UnchangedOrSameAs(node->ty) && fields_u.UnchangedOrSameAs(node->fields))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TupleNode*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!fields_u.IsUnchanged()) writable->fields = std::move(fields_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TupleNode>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!fields_u.IsUnchanged()) copy->fields = std::move(fields_u).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::Mutate_(const TupleGetItemNode* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  auto tuple_u = Mutate(node->tuple, inplace_mode);
  if (ty_u.UnchangedOrSameAs(node->ty) && tuple_u.UnchangedOrSameAs(node->tuple))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TupleGetItemNode*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!tuple_u.IsUnchanged()) writable->tuple = std::move(tuple_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TupleGetItemNode>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!tuple_u.IsUnchanged()) copy->tuple = std::move(tuple_u).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const TensorLoadNode* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  auto source_u = Mutate(node->source, inplace_mode);
  auto indices_u =
      Mutate(node->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  if (ty_u.UnchangedOrSameAs(node->ty) && source_u.UnchangedOrSameAs(node->source) &&
      indices_u.UnchangedOrSameAs(node->indices))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TensorLoadNode*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!source_u.IsUnchanged()) writable->source = std::move(source_u).ValueUnchecked();
    if (!indices_u.IsUnchanged()) writable->indices = std::move(indices_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TensorLoadNode>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!source_u.IsUnchanged()) copy->source = std::move(source_u).ValueUnchecked();
  if (!indices_u.IsUnchanged()) copy->indices = std::move(indices_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::Mutate_(const TensorRegionNode* node, InplaceMode inplace_mode) {
  auto ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  auto source_u = Mutate(node->source, inplace_mode);
  auto region_u = Mutate(node->region, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Range>>>();
  if (ty_u.UnchangedOrSameAs(node->ty) && source_u.UnchangedOrSameAs(node->source) &&
      region_u.UnchangedOrSameAs(node->region))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<TensorRegionNode*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!source_u.IsUnchanged()) writable->source = std::move(source_u).ValueUnchecked();
    if (!region_u.IsUnchanged()) writable->region = std::move(region_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<TensorRegionNode>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!source_u.IsUnchanged()) copy->source = std::move(source_u).ValueUnchecked();
  if (!region_u.IsUnchanged()) copy->region = std::move(region_u).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::Mutate_(const GlobalVarNode* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<Expr> ExprMutator::Mutate_(const CallNode* node, InplaceMode inplace_mode) {
  UnchangedOr<Type> ty_u = ffi::Unchanged();
  if (!node->ty.as<PrimTypeNode>()) {
    ty_u = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  }
  UnchangedOr<Expr> op_u = ffi::Unchanged();
  if (!node->op.as<OpNode>()) {
    op_u = Mutate(node->op, inplace_mode);
  }
  auto args_u = Mutate(node->args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Expr>>>();
  UnchangedOr<ffi::Array<Type>> ty_args_u = ffi::Unchanged();
  if (!node->ty_args.empty()) {
    ty_args_u = Mutate(node->ty_args, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<Type>>>();
  }
  if (ty_u.UnchangedOrSameAs(node->ty) && op_u.UnchangedOrSameAs(node->op) &&
      args_u.UnchangedOrSameAs(node->args) && ty_args_u.UnchangedOrSameAs(node->ty_args))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<CallNode*>(node);
    if (!ty_u.IsUnchanged()) writable->ty = std::move(ty_u).ValueUnchecked();
    if (!op_u.IsUnchanged()) writable->op = std::move(op_u).ValueUnchecked();
    if (!args_u.IsUnchanged()) writable->args = std::move(args_u).ValueUnchecked();
    if (!ty_args_u.IsUnchanged()) writable->ty_args = std::move(ty_args_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<CallNode>(*node);
  if (!ty_u.IsUnchanged()) copy->ty = std::move(ty_u).ValueUnchecked();
  if (!op_u.IsUnchanged()) copy->op = std::move(op_u).ValueUnchecked();
  if (!args_u.IsUnchanged()) copy->args = std::move(args_u).ValueUnchecked();
  if (!ty_args_u.IsUnchanged()) copy->ty_args = std::move(ty_args_u).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::Mutate_(const GenericConstNode* node, InplaceMode inplace_mode) {
  auto ty = Mutate(node->ty, inplace_mode).as_or_throw<UnchangedOr<Type>>();
  if (ty.UnchangedOrSameAs(node->ty)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    const_cast<GenericConstNode*>(node)->ty = std::move(ty).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<GenericConstNode>(*node);
  copy->ty = std::move(ty).ValueUnchecked();
  return Expr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const IntImmNode* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const FloatImmNode* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<Expr> ExprMutator::Mutate_(const OpNode* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<Expr> ExprMutator::Mutate_(const StringImmNode* node, InplaceMode inplace_mode) {
  // Registry atoms and constant leaves do not descend into metadata or types.
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::CastNode* node, InplaceMode inplace_mode) {
  auto value_u = Mutate(node->value, inplace_mode);
  if (value_u.UnchangedOrSameAs(node->value)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::CastNode*>(node);
    if (!value_u.IsUnchanged()) writable->value = std::move(value_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::CastNode>(*node);
  if (!value_u.IsUnchanged()) copy->value = std::move(value_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

#define TVM_IR_BINARY_MUTATE_IMPL(Name)                                                            \
  UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::Name##Node* node,                         \
                                             InplaceMode inplace_mode) {                           \
    auto a_u = Mutate(node->a, inplace_mode);                                                      \
    auto b_u = Mutate(node->b, inplace_mode);                                                      \
    if (a_u.UnchangedOrSameAs(node->a) && b_u.UnchangedOrSameAs(node->b)) return ffi::Unchanged(); \
    if (inplace_mode == InplaceMode::kAllow) {                                                     \
      auto* writable = const_cast<prim::Name##Node*>(node);                                        \
      if (!a_u.IsUnchanged()) writable->a = std::move(a_u).ValueUnchecked();                       \
      if (!b_u.IsUnchanged()) writable->b = std::move(b_u).ValueUnchecked();                       \
      return ffi::Unchanged();                                                                     \
    }                                                                                              \
    auto copy = ffi::make_object<prim::Name##Node>(*node);                                         \
    if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();                             \
    if (!b_u.IsUnchanged()) copy->b = std::move(b_u).ValueUnchecked();                             \
    return PrimExpr(std::move(copy));                                                              \
  }
TVM_IR_BINARY_MUTATE_IMPL(Add)
TVM_IR_BINARY_MUTATE_IMPL(LShift)
TVM_IR_BINARY_MUTATE_IMPL(RShift)
TVM_IR_BINARY_MUTATE_IMPL(BitwiseAnd)
TVM_IR_BINARY_MUTATE_IMPL(BitwiseOr)
TVM_IR_BINARY_MUTATE_IMPL(BitwiseXor)
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

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::NotNode* node, InplaceMode inplace_mode) {
  auto a_u = Mutate(node->a, inplace_mode);
  if (a_u.UnchangedOrSameAs(node->a)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::NotNode*>(node);
    if (!a_u.IsUnchanged()) writable->a = std::move(a_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::NotNode>(*node);
  if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::BitwiseNotNode* node,
                                           InplaceMode inplace_mode) {
  auto a_u = Mutate(node->a, inplace_mode);
  if (a_u.UnchangedOrSameAs(node->a)) return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::BitwiseNotNode*>(node);
    if (!a_u.IsUnchanged()) writable->a = std::move(a_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::BitwiseNotNode>(*node);
  if (!a_u.IsUnchanged()) copy->a = std::move(a_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::SelectNode* node, InplaceMode inplace_mode) {
  auto condition_u = Mutate(node->condition, inplace_mode);
  auto true_value_u = Mutate(node->true_value, inplace_mode);
  auto false_value_u = Mutate(node->false_value, inplace_mode);
  if (condition_u.UnchangedOrSameAs(node->condition) &&
      true_value_u.UnchangedOrSameAs(node->true_value) &&
      false_value_u.UnchangedOrSameAs(node->false_value))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::SelectNode*>(node);
    if (!condition_u.IsUnchanged()) writable->condition = std::move(condition_u).ValueUnchecked();
    if (!true_value_u.IsUnchanged())
      writable->true_value = std::move(true_value_u).ValueUnchecked();
    if (!false_value_u.IsUnchanged())
      writable->false_value = std::move(false_value_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::SelectNode>(*node);
  if (!condition_u.IsUnchanged()) copy->condition = std::move(condition_u).ValueUnchecked();
  if (!true_value_u.IsUnchanged()) copy->true_value = std::move(true_value_u).ValueUnchecked();
  if (!false_value_u.IsUnchanged()) copy->false_value = std::move(false_value_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::LetNode* node, InplaceMode inplace_mode) {
  auto var_u = WithDefRegionKind(kTVMFFIDefRegionKindSimple, [&] {
                 return Mutate(node->var, inplace_mode);
               }).as_or_throw<UnchangedOr<Var>>();
  auto value_u = Mutate(node->value, inplace_mode);
  auto body_u = Mutate(node->body, inplace_mode);
  if (var_u.UnchangedOrSameAs(node->var) && value_u.UnchangedOrSameAs(node->value) &&
      body_u.UnchangedOrSameAs(node->body))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::LetNode*>(node);
    if (!var_u.IsUnchanged()) writable->var = std::move(var_u).ValueUnchecked();
    if (!value_u.IsUnchanged()) writable->value = std::move(value_u).ValueUnchecked();
    if (!body_u.IsUnchanged()) writable->body = std::move(body_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::LetNode>(*node);
  if (!var_u.IsUnchanged()) copy->var = std::move(var_u).ValueUnchecked();
  if (!value_u.IsUnchanged()) copy->value = std::move(value_u).ValueUnchecked();
  if (!body_u.IsUnchanged()) copy->body = std::move(body_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::RampNode* node, InplaceMode inplace_mode) {
  auto base_u = Mutate(node->base, inplace_mode);
  auto stride_u = Mutate(node->stride, inplace_mode);
  auto lanes_u = Mutate(node->lanes, inplace_mode);
  if (base_u.UnchangedOrSameAs(node->base) && stride_u.UnchangedOrSameAs(node->stride) &&
      lanes_u.UnchangedOrSameAs(node->lanes))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::RampNode*>(node);
    if (!base_u.IsUnchanged()) writable->base = std::move(base_u).ValueUnchecked();
    if (!stride_u.IsUnchanged()) writable->stride = std::move(stride_u).ValueUnchecked();
    if (!lanes_u.IsUnchanged()) writable->lanes = std::move(lanes_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::RampNode>(*node);
  if (!base_u.IsUnchanged()) copy->base = std::move(base_u).ValueUnchecked();
  if (!stride_u.IsUnchanged()) copy->stride = std::move(stride_u).ValueUnchecked();
  if (!lanes_u.IsUnchanged()) copy->lanes = std::move(lanes_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::BroadcastNode* node,
                                           InplaceMode inplace_mode) {
  auto value_u = Mutate(node->value, inplace_mode);
  auto lanes_u = Mutate(node->lanes, inplace_mode);
  if (value_u.UnchangedOrSameAs(node->value) && lanes_u.UnchangedOrSameAs(node->lanes))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::BroadcastNode*>(node);
    if (!value_u.IsUnchanged()) writable->value = std::move(value_u).ValueUnchecked();
    if (!lanes_u.IsUnchanged()) writable->lanes = std::move(lanes_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::BroadcastNode>(*node);
  if (!value_u.IsUnchanged()) copy->value = std::move(value_u).ValueUnchecked();
  if (!lanes_u.IsUnchanged()) copy->lanes = std::move(lanes_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<PrimExpr> ExprMutator::Mutate_(const prim::ShuffleNode* node,
                                           InplaceMode inplace_mode) {
  auto vectors_u =
      Mutate(node->vectors, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  auto indices_u =
      Mutate(node->indices, inplace_mode).as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
  if (vectors_u.UnchangedOrSameAs(node->vectors) && indices_u.UnchangedOrSameAs(node->indices))
    return ffi::Unchanged();
  if (inplace_mode == InplaceMode::kAllow) {
    auto* writable = const_cast<prim::ShuffleNode*>(node);
    if (!vectors_u.IsUnchanged()) writable->vectors = std::move(vectors_u).ValueUnchecked();
    if (!indices_u.IsUnchanged()) writable->indices = std::move(indices_u).ValueUnchecked();
    return ffi::Unchanged();
  }
  auto copy = ffi::make_object<prim::ShuffleNode>(*node);
  if (!vectors_u.IsUnchanged()) copy->vectors = std::move(vectors_u).ValueUnchecked();
  if (!indices_u.IsUnchanged()) copy->indices = std::move(indices_u).ValueUnchecked();
  return PrimExpr(std::move(copy));
}

UnchangedOr<Expr> ExprMutator::Mutate_(const VarNode* node, InplaceMode inplace_mode) {
  if (TVM_FFI_PREDICT_TRUE(var_remap_.empty() && def_region_kind() == kTVMFFIDefRegionKindNone)) {
    return ffi::Unchanged();
  }
  ffi::Any remap_result = VarRemapGet(ffi::AnyView(node));
  if (remap_result.type_index() != ffi::TypeIndex::kTVMFFINone) {
    return std::move(remap_result).as_or_throw<UnchangedOr<Expr>>();
  }
  if (def_region_kind() == kTVMFFIDefRegionKindNone) return ffi::Unchanged();
  UnchangedOr<Expr> result_u = ffi::Unchanged();
  ffi::Any mapped_value = ffi::Unchanged();
  // PrimType has no children; dynamic type fields inherit Pattern but are visited outside Simple.
  if (!node->ty.as<PrimTypeNode>()) {
    UnchangedOr<ffi::Any> mapped_ty_result_u =
        def_region_kind() == kTVMFFIDefRegionKindSimple
            ? WithDefRegionKind(kTVMFFIDefRegionKindNone,
                                [&] { return Mutate(node->ty, inplace_mode); })
            : Mutate(node->ty, inplace_mode);
    auto mapped_ty_u = std::move(mapped_ty_result_u).as_or_throw<UnchangedOr<Type>>();
    if (!mapped_ty_u.UnchangedOrSameAs(node->ty)) {
      Expr mapped_expr;
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<VarNode*>(node)->ty = std::move(mapped_ty_u).ValueUnchecked();
        mapped_expr = ffi::GetRef<Expr>(node);
      } else {
        auto copy = ffi::make_object<VarNode>(*node);
        copy->ty = std::move(mapped_ty_u).ValueUnchecked();
        mapped_expr = Expr(std::move(copy));
      }
      result_u = mapped_expr;
      mapped_value = std::move(mapped_expr);
    }
  }
  if (!result_u.IsUnchanged() || def_region_kind() == kTVMFFIDefRegionKindPattern) {
    VarRemapSet(ffi::AnyView(node), mapped_value);
  }
  return result_u;
}

}  // namespace tvm
