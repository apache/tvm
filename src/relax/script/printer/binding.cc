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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/op.h>
#include <tvm/relax/block_builder.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/tirx/type.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> EmitRelaxMatchCast(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object* destination) {
  const auto* binding =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::MatchCastNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ExprDoc rhs =
      NamespaceDoc("relax")
          ->Attr("match_cast")
          ->Call({d->Translate(binding->value).value(),
                  (binding->ty.as<tirx::BufferTypeNode>() ? TypeValue(d, binding->ty, false)
                                                          : d->Translate(binding->ty).value())});
  IdDoc lhs = VarDoc(d, binding->var);
  ffi::Optional<ExprDoc> annotation = std::nullopt;
  if (!binding->var->ty.IsMissing())
    annotation =
        (binding->var->ty.as<tirx::BufferTypeNode>() ? TypeValue(d, binding->var->ty, false)
                                                     : d->Translate(binding->var->ty).value());
  if (!d->GetExtraConfig<bool>("relax.show_all_ty", true)) annotation = std::nullopt;
  d->Emit(AssignDoc(lhs, rhs, annotation), ffi::GetRef<ffi::ObjectRef>(binding));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::MatchCastNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&EmitRelaxMatchCast>());
}

ffi::Optional<ExprDoc> EmitRelaxVarBinding(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object* destination) {
  const auto* binding =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::VarBindingNode>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  if (auto func = binding->value.as<relax::FunctionNode>()) {
    d->Emit(CommentDoc("from tvm.script import relax as R"), ffi::GetRef<ffi::ObjectRef>(binding));
    d->Emit(CommentDoc(""), ffi::GetRef<ffi::ObjectRef>(binding));
    IdDoc lhs = VarDoc(d, binding->var);
    d->Translate(binding->value);
    FunctionDoc function = d->CurrentScopeDocs().back().as_or_throw<FunctionDoc>();
    function->name = lhs;
    ExprDoc decorator = NamespaceDoc("relax")->Attr("function");
    if (!func->is_pure) {
      decorator = decorator->Call({}, {"pure"}, {LiteralDoc::Boolean(false, std::nullopt)});
    }
    function->decorators = {decorator};
    return std::nullopt;
  }
  if (relax::HasVoidType(binding->value) && relax::HasVoidType(binding->var)) {
    VarDoc(d, binding->var);
    ffi::Optional<ExprDoc> rhs = d->Translate(binding->value, binding->var);
    if (rhs.has_value()) d->Emit(ExprStmtDoc(rhs.value()), ffi::GetRef<ffi::ObjectRef>(binding));
    return std::nullopt;
  }
  IdDoc lhs = VarDoc(d, binding->var);
  ffi::Optional<ExprDoc> rhs = d->Translate(binding->value, binding->var);
  if (!rhs.has_value()) return std::nullopt;
  ffi::Optional<ExprDoc> annotation = std::nullopt;
  bool infer_vdevice = false;
  bool explicit_output_type = false;
  if (auto tensor = binding->var->ty.as<relax::TensorTypeNode>();
      tensor && tensor->vdevice.has_value()) {
    if (auto call = binding->value.as<CallNode>()) {
      if (auto op = call->op.as<Op>()) infer_vdevice = op.value()->name == "relax.to_vdevice";
    }
  }
  if (auto call = binding->value.as<CallNode>(); call && call->ty_args.size() == 1) {
    if (auto op = call->op.as<Op>()) {
      const auto& name = op.value()->name;
      explicit_output_type =
          (name == "relax.call_tir" || name == "relax.call_dps_packed" ||
           name == "relax.call_tir_with_grad" || name == "relax.call_tir_inplace" ||
           name == "relax.dist.call_tir_local_view") &&
          ffi::StructuralEqual()(binding->var->ty, call->ty_args[0]);
    }
  }
  if (!binding->var->ty.IsMissing() && !infer_vdevice && !explicit_output_type)
    annotation =
        (binding->var->ty.as<tirx::BufferTypeNode>() ? TypeValue(d, binding->var->ty, false)
                                                     : d->Translate(binding->var->ty).value());
  bool inferable = false;
  if (annotation.has_value()) {
    ffi::Optional<Type> inferred = binding->value->ty;
    if (auto call = binding->value.as<CallNode>()) {
      inferred = std::nullopt;
      if (auto op = call->op.as<Op>()) {
        static const OpAttrMap<FInferType> context_free = Op::GetAttrMap<FInferType>("FInferType");
        static const OpAttrMap<relax::FInferTypeWithBuilder> with_builder =
            Op::GetAttrMap<relax::FInferTypeWithBuilder>("relax.FInferTypeWithBuilder");
        try {
          if (context_free.count(op.value())) {
            inferred = Call::ReinferType(call);
          } else if (with_builder.count(op.value())) {
            auto builder = relax::BlockBuilder::Create(std::nullopt);
            inferred = with_builder[op.value()](ffi::GetRef<Call>(call), builder);
          }
        } catch (const ffi::Error&) {
          inferred = std::nullopt;
        }
      }
    }
    // Primitive bindings retain their annotation even when the expression's
    // type is known.  The Relax parser uses it to keep primitive aliases as
    // typed bindings rather than plain Python assignments.
    inferable = !binding->var->ty.as<PrimTypeNode>() && inferred.has_value() &&
                !inferred.value().IsMissing() &&
                ffi::StructuralEqual()(binding->var->ty, inferred.value());
  }
  if (inferable && !d->GetExtraConfig<bool>("relax.show_all_ty", true)) annotation = std::nullopt;
  d->Emit(AssignDoc(lhs, rhs.value(), annotation), ffi::GetRef<ffi::ObjectRef>(binding));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::VarBindingNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&EmitRelaxVarBinding>());
}

ffi::Optional<ExprDoc> EmitRelaxIf(DocTranslatorObj* d, ffi::AnyView input,
                                   const ffi::Object* destination) {
  const auto* branch =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::IfNode>(input);
  // A value context needs a usable expression after the conditional.  A
  // destination supplied by a binding is completed directly on both arms.
  ffi::Optional<Var> temporary = std::nullopt;
  if (!destination) {
    temporary = Var("if_result", branch->ty);
    VarDoc(d, temporary.value());
    destination = temporary.value().get();
  }
  TVM_FFI_CHECK(destination->IsInstance<VarNode>(), TypeError)
      << "printer Relax If destination must be a Var";
  Var var = ffi::GetRef<Var>(static_cast<const VarNode*>(destination));
  ffi::Optional<IdDoc> lhs = VarDoc(d, var);
  ffi::Optional<ExprDoc> annotation = std::nullopt;
  if (!var->ty.IsMissing())
    annotation = (var->ty.as<tirx::BufferTypeNode>() ? TypeValue(d, var->ty, false)
                                                     : d->Translate(var->ty).value());
  ExprDoc condition = d->Translate(branch->cond).value();
  d->Emit(IfDoc(condition, RelaxSeqBody(d, branch->true_branch.get(), lhs, annotation, destination),
                RelaxSeqBody(d, branch->false_branch.get(), lhs, annotation, destination)),
          ffi::GetRef<ffi::ObjectRef>(branch));
  return temporary.has_value() ? ffi::Optional<ExprDoc>(lhs.value()) : std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::IfNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&EmitRelaxIf>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
