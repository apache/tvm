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
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tile_primitive.h>

#include <utility>

namespace tvm {
namespace tirx {

// TilePrimitiveCall
TilePrimitiveCall::TilePrimitiveCall(tvm::Op op, ffi::Array<ffi::Any> args,
                                     ffi::Map<ffi::String, BufferVar> workspace,
                                     ffi::Map<ffi::String, ffi::Any> config,
                                     ffi::Optional<ffi::String> dispatch, ExecScope scope) {
  TVM_FFI_CHECK(op.defined(), ValueError) << "TilePrimitiveCall expects a defined operator";
  static const auto& category_map = Op::GetAttrMap<TIRxOpCategory>("TIRxOpCategory");
  TVM_FFI_ICHECK(category_map.get(op, ffi::String("")) == "tile_primitive")
      << "Only tile primitive ops can be used in tirx::TilePrimitiveCall";
  ffi::ObjectPtr<TilePrimitiveCallNode> n = ffi::make_object<TilePrimitiveCallNode>(
      std::move(op), std::move(args), std::move(workspace), std::move(config), std::move(dispatch),
      std::move(scope));
  data_ = std::move(n);
}

namespace {

TVMFFIAny TilePrimitiveCallVisit(ffi::StructuralVisitorObj* visitor, ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  const TilePrimitiveCallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TilePrimitiveCallNode>(
          value);
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->args));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->workspace));
  TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(visitor->VisitExpected(self->config));
  return ffi::AnyView(nullptr).CopyToTVMFFIAny();
}

TVMFFIAny TilePrimitiveCallMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  const TilePrimitiveCallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TilePrimitiveCallNode>(
          value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<ffi::Any>>, mapped_args,
                                    mutator->MutateExpected(self->args));

  using WorkspaceMap = ffi::Map<ffi::String, BufferVar>;
  using ConfigMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<WorkspaceMap>, mapped_workspace,
                                    mutator->MutateExpected(self->workspace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ConfigMap>, mapped_config,
                                    mutator->MutateExpected(self->config));

  if (mapped_args.UnchangedOrSameAs(self->args) &&
      mapped_workspace.UnchangedOrSameAs(self->workspace) &&
      mapped_config.UnchangedOrSameAs(self->config)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  ffi::ObjectPtr<TilePrimitiveCallNode> copy = ffi::make_object<TilePrimitiveCallNode>(*self);
  copy->args = std::move(mapped_args).ValueOrUnchanged(std::move(copy->args));
  copy->workspace = std::move(mapped_workspace).ValueOrUnchanged(std::move(copy->workspace));
  copy->config = std::move(mapped_config).ValueOrUnchanged(std::move(copy->config));
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TilePrimitiveCallMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                              ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  TilePrimitiveCallNode* self = const_cast<TilePrimitiveCallNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TilePrimitiveCallNode>(
          value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ffi::Array<ffi::Any>>, mapped_args,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->args));

  using WorkspaceMap = ffi::Map<ffi::String, BufferVar>;
  using ConfigMap = ffi::Map<ffi::String, ffi::Any>;
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<WorkspaceMap>, mapped_workspace,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->workspace));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::UnchangedOr<ConfigMap>, mapped_config,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->config));

  if (mapped_args.UnchangedOrSameAs(self->args) &&
      mapped_workspace.UnchangedOrSameAs(self->workspace) &&
      mapped_config.UnchangedOrSameAs(self->config)) {
    return ffi::Unchanged().CopyToTVMFFIAny();
  }
  self->args = std::move(mapped_args).ValueOrUnchanged(std::move(self->args));
  self->workspace = std::move(mapped_workspace).ValueOrUnchanged(std::move(self->workspace));
  self->config = std::move(mapped_config).ValueOrUnchanged(std::move(self->config));
  return ffi::Unchanged().CopyToTVMFFIAny();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  TilePrimitiveCallNode::RegisterReflection();
  refl::GlobalDef().def(
      "tirx.TilePrimitiveCall",
      [](tvm::Op op, ffi::Array<ffi::Any> args, ffi::Map<ffi::String, BufferVar> workspace,
         ffi::Map<ffi::String, ffi::Any> config, ffi::Optional<ffi::String> dispatch,
         ExecScope scope) {
        return TilePrimitiveCall(op, args, workspace, config, dispatch, scope);
      });
  refl::TypeAttrDef<TilePrimitiveCallNode>()
      .attr(refl::type_attr::kStructuralVisit, reinterpret_cast<void*>(&TilePrimitiveCallVisit))
      .attr(refl::type_attr::kStructuralMutate, reinterpret_cast<void*>(&TilePrimitiveCallMutate))
      .attr(refl::type_attr::kStructuralMaybeInplaceMutate,
            reinterpret_cast<void*>(&TilePrimitiveCallMaybeInplaceMutate));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.TilePrimitiveCallCopyHandle",
                        [](const TilePrimitiveCall& op) { return TilePrimitiveCall(op); });
}

}  // namespace tirx
}  // namespace tvm
