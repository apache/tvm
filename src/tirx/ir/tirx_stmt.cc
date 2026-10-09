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
 * \file tir/tirx_stmt.cc
 * TIRX statement nodes.
 */

#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <utility>

namespace tvm {
namespace tirx {

// TileOpCall
TileOpCall::TileOpCall(tvm::Op op, ffi::Array<Expr> args,
                       ffi::Map<ffi::String, TensorVar> workspace,
                       ffi::Map<ffi::String, Expr> config, ffi::Optional<ffi::String> dispatch,
                       ExecScope scope)
    : Stmt(ffi::UnsafeInit{}) {
  TVM_FFI_CHECK(op.defined(), ValueError) << "TileOpCall expects a defined operator";
  static const auto& category_map = Op::GetAttrMap<TIRxOpCategory>("TIRxOpCategory");
  TVM_FFI_ICHECK(category_map.get(op, ffi::String("")) == "tile_primitive")
      << "Only tile primitive ops can be used in tirx::TileOpCall";
  TVM_FFI_CHECK_GE(args.size(), op->args_info.size(), ValueError)
      << op->name << " requires " << op->args_info.size() << " operands";
  if (!op->var_args_info.has_value()) {
    TVM_FFI_CHECK_EQ(args.size(), op->args_info.size(), ValueError)
        << op->name << " expects " << op->args_info.size() << " operands";
  }
  if (auto gather = config.Get("gather4")) {
    auto tuple = gather.value().as<tvm::Tuple>();
    TVM_FFI_CHECK(tuple.has_value() && tuple.value()->fields.size() == 4, ValueError)
        << "gather4 must contain exactly four row coordinates";
  }
  ffi::StructuralVisit(args,
                       [](const TensorRegionNode* region,
                          ffi::StructuralVisitorObj*) -> ffi::Optional<ffi::VisitInterrupt> {
                         const auto buffer = region->source.as_or_throw<TensorVar>();
                         TVM_FFI_ICHECK_EQ(buffer->shape.size(), region->region.size())
                             << "Tile region must match its buffer rank";
                         return std::nullopt;
                       });
  ffi::ObjectPtr<TileOpCallNode> n =
      ffi::make_object<TileOpCallNode>(std::move(op), std::move(args), std::move(workspace),
                                       std::move(config), std::move(dispatch), std::move(scope));
  data_ = std::move(n);
}

namespace {

TVM_FFI_INLINE ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> TileOpCallVisit(
    ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  const TileOpCallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TileOpCallNode>(value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->workspace));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->config));
  return std::nullopt;
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TileOpCallMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  const TileOpCallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TileOpCallNode>(value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
                                    mutator->MutateExpected(self->args));

  using WorkspaceMap = ffi::Map<ffi::String, TensorVar>;
  using ConfigMap = ffi::Map<ffi::String, Expr>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<WorkspaceMap>, mapped_workspace,
                                    mutator->MutateExpected(self->workspace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ConfigMap>, mapped_config,
                                    mutator->MutateExpected(self->config));

  if (mapped_args.UnchangedOrSameAs(self->args) &&
      mapped_workspace.UnchangedOrSameAs(self->workspace) &&
      mapped_config.UnchangedOrSameAs(self->config)) {
    return ffi::Unchanged();
  }
  ffi::ObjectPtr<TileOpCallNode> copy = ffi::make_object<TileOpCallNode>(*self);
  copy->args = std::move(mapped_args).ValueOrUnchanged(std::move(copy->args));
  copy->workspace = std::move(mapped_workspace).ValueOrUnchanged(std::move(copy->workspace));
  copy->config = std::move(mapped_config).ValueOrUnchanged(std::move(copy->config));
  return ffi::Any(std::move(copy));
}

TVM_FFI_INLINE ffi::Expected<ffi::UnchangedOr<ffi::Any>> TileOpCallMaybeInplaceMutate(
    ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  TileOpCallNode* self = const_cast<TileOpCallNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TileOpCallNode>(value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<Expr>>, mapped_args,
                                    mutator->MutateExpected(self->args, ffi::InplaceMode::kAllow));

  using WorkspaceMap = ffi::Map<ffi::String, TensorVar>;
  using ConfigMap = ffi::Map<ffi::String, Expr>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<WorkspaceMap>, mapped_workspace,
      mutator->MutateExpected(self->workspace, ffi::InplaceMode::kAllow));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(
      ffi::UnchangedOr<ConfigMap>, mapped_config,
      mutator->MutateExpected(self->config, ffi::InplaceMode::kAllow));

  if (mapped_args.UnchangedOrSameAs(self->args) &&
      mapped_workspace.UnchangedOrSameAs(self->workspace) &&
      mapped_config.UnchangedOrSameAs(self->config)) {
    return ffi::Unchanged();
  }
  self->args = std::move(mapped_args).ValueOrUnchanged(std::move(self->args));
  self->workspace = std::move(mapped_workspace).ValueOrUnchanged(std::move(self->workspace));
  self->config = std::move(mapped_config).ValueOrUnchanged(std::move(self->config));
  return ffi::Unchanged();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TileOpCallNode::RegisterReflection();
  refl::GlobalDef().def(
      "tirx.TileOpCall",
      [](tvm::Op op, ffi::Array<Expr> args, ffi::Map<ffi::String, TensorVar> workspace,
         ffi::Map<ffi::String, Expr> config, ffi::Optional<ffi::String> dispatch,
         ExecScope scope) { return TileOpCall(op, args, workspace, config, dispatch, scope); });
  refl::TypeAttrDef<TileOpCallNode>()
      .attr(refl::type_attr::kStructuralVisit,
            ffi::FStructuralVisit::FromNative<&TileOpCallVisit>())
      .attr(refl::type_attr::kStructuralMutate,
            ffi::FStructuralMutate::FromNative<&TileOpCallMutate>())
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            ffi::FStructuralMutate::FromNative<&TileOpCallMaybeInplaceMutate>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.TileOpCallCopyHandle",
                        [](const TileOpCall& op) { return TileOpCall(op); });
}

}  // namespace tirx
}  // namespace tvm
