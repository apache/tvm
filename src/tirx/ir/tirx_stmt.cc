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
  TVM_FFI_S_VISIT_RETURN_NONE();
}

TVMFFIAny TilePrimitiveCallMutate(ffi::StructuralMutatorObj* mutator, ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  const TilePrimitiveCallNode* self =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TilePrimitiveCallNode>(
          value);
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<ffi::Any>, mapped_args,
                                    mutator->MutateExpected(self->args));

  auto mapped_workspace_result = mutator->MutateExpected(self->workspace);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(mapped_workspace_result);
  bool mapped_workspace_type_ok =
      ffi::details::AnyUnsafe::CheckAnyStrict<ffi::Map<ffi::String, BufferVar>>(
          ffi::details::ExpectedUnsafe::GetData(mapped_workspace_result));
  if (TVM_FFI_PREDICT_FALSE(!mapped_workspace_type_ok)) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::details::SMutateDeclaredTypeError());
  }
  ffi::Map<ffi::String, BufferVar> mapped_workspace =
      ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<ffi::Map<ffi::String, BufferVar>>(
          std::move(ffi::details::ExpectedUnsafe::GetData(mapped_workspace_result)));

  auto mapped_config_result = mutator->MutateExpected(self->config);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(mapped_config_result);
  bool mapped_config_type_ok =
      ffi::details::AnyUnsafe::CheckAnyStrict<ffi::Map<ffi::String, ffi::Any>>(
          ffi::details::ExpectedUnsafe::GetData(mapped_config_result));
  if (TVM_FFI_PREDICT_FALSE(!mapped_config_type_ok)) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::details::SMutateDeclaredTypeError());
  }
  ffi::Map<ffi::String, ffi::Any> mapped_config =
      ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<ffi::Map<ffi::String, ffi::Any>>(
          std::move(ffi::details::ExpectedUnsafe::GetData(mapped_config_result)));

  if (mapped_args.same_as(self->args) && mapped_workspace.same_as(self->workspace) &&
      mapped_config.same_as(self->config)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  ffi::ObjectPtr<TilePrimitiveCallNode> copy = ffi::make_object<TilePrimitiveCallNode>(*self);
  copy->args = std::move(mapped_args);
  copy->workspace = std::move(mapped_workspace);
  copy->config = std::move(mapped_config);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(std::move(copy)));
}

TVMFFIAny TilePrimitiveCallMaybeInplaceMutate(ffi::StructuralMutatorObj* mutator,
                                              ffi::AnyView value) noexcept {
  // skips: op, dispatch, scope
  TilePrimitiveCallNode* self = const_cast<TilePrimitiveCallNode*>(
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TilePrimitiveCallNode>(
          value));
  TVM_FFI_S_MUTATE_ASSIGN_OR_RETURN(ffi::Array<ffi::Any>, mapped_args,
                                    mutator->MaybeInplaceMutateIfUniqueExpected(self->args));

  auto mapped_workspace_result = mutator->MaybeInplaceMutateIfUniqueExpected(self->workspace);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(mapped_workspace_result);
  bool mapped_workspace_type_ok =
      ffi::details::AnyUnsafe::CheckAnyStrict<ffi::Map<ffi::String, BufferVar>>(
          ffi::details::ExpectedUnsafe::GetData(mapped_workspace_result));
  if (TVM_FFI_PREDICT_FALSE(!mapped_workspace_type_ok)) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::details::SMutateDeclaredTypeError());
  }
  ffi::Map<ffi::String, BufferVar> mapped_workspace =
      ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<ffi::Map<ffi::String, BufferVar>>(
          std::move(ffi::details::ExpectedUnsafe::GetData(mapped_workspace_result)));

  auto mapped_config_result = mutator->MaybeInplaceMutateIfUniqueExpected(self->config);
  TVM_FFI_S_MUTATE_MAYBE_EARLY_RETURN(mapped_config_result);
  bool mapped_config_type_ok =
      ffi::details::AnyUnsafe::CheckAnyStrict<ffi::Map<ffi::String, ffi::Any>>(
          ffi::details::ExpectedUnsafe::GetData(mapped_config_result));
  if (TVM_FFI_PREDICT_FALSE(!mapped_config_type_ok)) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::details::SMutateDeclaredTypeError());
  }
  ffi::Map<ffi::String, ffi::Any> mapped_config =
      ffi::details::AnyUnsafe::MoveFromAnyAfterCheck<ffi::Map<ffi::String, ffi::Any>>(
          std::move(ffi::details::ExpectedUnsafe::GetData(mapped_config_result)));

  if (mapped_args.same_as(self->args) && mapped_workspace.same_as(self->workspace) &&
      mapped_config.same_as(self->config)) {
    return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
  }
  self->args = std::move(mapped_args);
  self->workspace = std::move(mapped_workspace);
  self->config = std::move(mapped_config);
  return ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(ffi::Any(self));
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
