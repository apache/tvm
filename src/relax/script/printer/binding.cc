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
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/op.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/type.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> MatchCastDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object* destination) {
  const auto* binding =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::MatchCastNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ExprDoc rhs =
      NamespaceDoc("relax")
          ->Attr("match_cast")
          ->Call({d->Translate(binding->value).value(), d->Translate(binding->ty).value()});
  IdDoc lhs = VarDoc(d, binding->var);
  ffi::Optional<ExprDoc> annotation = std::nullopt;
  if (!binding->var->ty.as<MissingType>().has_value())
    annotation = d->Translate(binding->var->ty).value();
  if (!d->GetExtraConfig<bool>("relax.show_all_ty", true)) annotation = std::nullopt;
  d->Emit(AssignDoc(lhs, rhs, annotation), ffi::GetRef<ffi::ObjectRef>(binding));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::MatchCastNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&MatchCastDocTranslate>());
}

ffi::Optional<ExprDoc> VarBindingDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* binding =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::VarBindingNode>(
          input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  if (auto func = binding->value.as<relax::FunctionNode>()) {
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
  if (auto tensor = binding->var->ty.as<relax::TensorTypeNode>();
      tensor && tensor->vdevice.has_value()) {
    if (auto call = binding->value.as<CallNode>()) {
      if (auto op = call->op.as<Op>()) infer_vdevice = op.value()->name == "relax.to_vdevice";
    }
  }
  bool show_all_ty = d->GetExtraConfig<bool>("relax.show_all_ty", true);
  const auto* call = binding->value.as<CallNode>();
  bool output_type_argument = call && call->ty_args.size() == 1 &&
                              ffi::StructuralEqual()(binding->var->ty, call->ty_args[0]);
  ffi::Optional<Type> inferred = std::nullopt;
  if (!show_all_ty || output_type_argument) {
    if (!call) {
      inferred = binding->value->ty;
    } else if (auto op = call->op.as<Op>()) {
      static const auto fixed = Op::GetAttrMap<TFixedReturnType>(tvm::op_attr::kFixedReturnType);
      static const auto context_free = Op::GetAttrMap<FInferType>(tvm::op_attr::kInferType);
      if (fixed.count(op.value()) || context_free.count(op.value())) {
        try {
          inferred = Call::ReinferType(call);
        } catch (const ffi::Error&) {
          // Contextual inference remains the parser's responsibility.
        }
      }
    }
  }
  bool inferable = inferred.has_value() && !inferred.value().as<MissingType>().has_value() &&
                   ffi::StructuralEqual()(binding->var->ty, inferred.value());
  bool explicit_output_type = output_type_argument && inferable;
  // Primitive aliases need their annotation. Without context-free inference,
  // keep the binding type and let the parser's deferred inference use it.
  bool elide_annotation = infer_vdevice || explicit_output_type ||
                          (!show_all_ty && !binding->var->ty.as<PrimTypeNode>() && inferable);
  if (!binding->var->ty.as<MissingType>().has_value() && !elide_annotation) {
    annotation = d->Translate(binding->var->ty).value();
  }
  d->Emit(AssignDoc(lhs, rhs.value(), annotation), ffi::GetRef<ffi::ObjectRef>(binding));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::VarBindingNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&VarBindingDocTranslate>());
}

ffi::Optional<ExprDoc> IfDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                      const ffi::Object* destination) {
  const auto* branch =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::IfExprNode>(input);
  // A value context needs a usable expression after the conditional.  A
  // destination supplied by a binding is completed directly on both arms.
  ffi::Optional<Var> temporary = std::nullopt;
  if (!destination) {
    temporary = Var("if_result", branch->ty);
    VarDoc(d, temporary.value());
    destination = temporary.value().get();
  }
  TVM_FFI_CHECK(destination->IsInstance<VarNode>(), TypeError)
      << "printer Relax IfExpr destination must be a Var";
  Var var = ffi::GetRef<Var>(static_cast<const VarNode*>(destination));
  ffi::Optional<IdDoc> lhs = VarDoc(d, var);
  ffi::Optional<ExprDoc> annotation = std::nullopt;
  if (!var->ty.as<MissingType>().has_value()) annotation = d->Translate(var->ty).value();
  ExprDoc condition = d->Translate(branch->cond).value();
  d->Emit(IfDoc(condition, RelaxSeqBody(d, branch->true_branch.get(), lhs, annotation, destination),
                RelaxSeqBody(d, branch->false_branch.get(), lhs, annotation, destination)),
          ffi::GetRef<ffi::ObjectRef>(branch));
  return temporary.has_value() ? ffi::Optional<ExprDoc>(lhs.value()) : std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::IfExprNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate, FDocTranslate::FromNative<&IfDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
