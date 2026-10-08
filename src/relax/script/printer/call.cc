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
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/op.h>
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/op/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/type.h>

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../../../script/printer/ir/utils.h"
#include "../../../tirx/script/printer/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

// Shared Call takes Expr values rather than a named wrapper's Python sugar.
ExprDoc MaterializeCallArgument(DocTranslatorObj* d, const Expr& arg, ExprDoc doc) {
  if (const auto* tuple = arg.as<TupleNode>()) {
    if (auto fields = doc.as<TupleDocNode>()) {
      ffi::Array<ExprDoc> values;
      for (size_t i = 0; i < tuple->fields.size(); ++i) {
        values.push_back(MaterializeCallArgument(d, tuple->fields[i], fields->elements[i]));
      }
      doc = NamespaceDoc("relax")->Attr("tuple")->Call(values);
    }
  }
  if (const auto* region = arg.as<TensorRegionNode>()) {
    doc = TensorRegionValue(d, region, true);
  }
  d->RecordOrigin(doc, arg);
  return doc;
}

namespace {

// Relax's named constructors defer result typing until a binding supplies it.
// A standalone named call therefore reconstructs only a missing stored result.
bool HasRelaxCallResult(const CallNode* call, const ffi::Object* destination) {
  if (call->ty.as<MissingType>().has_value()) return true;
  const auto* var = destination && destination->IsInstance<VarNode>()
                        ? static_cast<const VarNode*>(destination)
                        : nullptr;
  return var && ffi::StructuralEqual()(var->ty, call->ty);
}

ffi::Optional<ExprDoc> RelaxCallDocTranslate(DocTranslatorObj* d, const CallNode* call,
                                             const ffi::Array<ExprDoc>& args) {
  const Op& op = call->op.as_or_throw<Op>();
  auto relax_format = [&](size_t index) -> ExprDoc {
    if (const auto* format = call->args[index].as<StringImmNode>()) {
      return NamespaceDoc("relax")->Attr("str")->Call(
          {LiteralDoc::Str(format->value, std::nullopt)});
    }
    return args[index];
  };

  if (op->name == "relax.call_py_func" && call->args.size() == 2 && !call->attrs.defined() &&
      call->ty_args.size() == 1) {
    const auto* tuple = call->args[1].as<TupleNode>();
    if (!tuple || !call->args[0].as<StringImmNode>()) return std::nullopt;
    const Type& output = call->ty_args[0];
    const auto* output_tuple = output.as<TupleTypeNode>();
    if (output_tuple && output_tuple->fields.size() == 1) return std::nullopt;
    ffi::Array<Type> types = output_tuple ? output_tuple->fields : ffi::Array<Type>{output};
    ffi::Array<ExprDoc> outputs;
    for (const Type& type : types) {
      const auto* tensor = type.as<relax::TensorTypeNode>();
      if (!tensor || !tensor->shape.as<relax::ShapeExprNode>()) return std::nullopt;
      outputs.push_back(d->Translate(type).value());
    }
    ExprDoc output_doc = output_tuple ? ExprDoc(ListDoc(outputs)) : outputs[0];
    d->RecordOrigin(output_doc, output);
    ffi::Array<ExprDoc> packed_args = {args[0]};
    for (const Expr& arg : tuple->fields) packed_args.push_back(d->Translate(arg).value());
    return NamespaceDoc("relax")->Attr("call_py_func")->Call(packed_args, {"out_ty"}, {output_doc});
  }

  if (op->name == "relax.print" && !call->args.empty() && !call->attrs.defined() &&
      call->ty_args.empty()) {
    ffi::Array<ExprDoc> positional;
    for (size_t i = 1; i < args.size(); ++i) positional.push_back(args[i]);
    return NamespaceDoc("relax")->Attr("print")->Call(positional, {"format"}, {relax_format(0)});
  }

  if (op->name == "relax.assert_op" && call->args.size() >= 2 && !call->attrs.defined() &&
      call->ty_args.empty()) {
    ffi::Array<ExprDoc> positional = {args[0]};
    for (size_t i = 2; i < args.size(); ++i) positional.push_back(args[i]);
    return NamespaceDoc("relax")
        ->Attr("assert_op")
        ->Call(positional, {"format"}, {relax_format(1)});
  }

  if (op->name == "relax.hint_on_device") {
    const auto* attrs = call->attrs.as<relax::HintOnDeviceAttrs>();
    if (!attrs || args.size() != 1 || !call->ty_args.empty()) return std::nullopt;
    ExprDoc device = NamespaceDoc("relax")->Attr("device")->Call(
        {LiteralDoc::Int(attrs->device_type, std::nullopt),
         LiteralDoc::Int(attrs->index, std::nullopt)});
    return NamespaceDoc("relax")
        ->Attr("hint_on_device")
        ->Call({args[0], device, LiteralDoc::Str(attrs->memory_scope, std::nullopt)});
  }

  static const auto& names = Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
  if (!names.count(op) || names[op].find("relax.") != 0 || !call->ty_args.empty()) {
    return std::nullopt;
  }
  if (call->attrs.defined() ? op->attrs_type_key != call->attrs->GetTypeKey()
                            : !op->attrs_type_key.empty()) {
    return std::nullopt;
  }
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (call->attrs.defined()) {
    std::vector<std::pair<ffi::String, ffi::Any>> fields;
    if (auto attrs = call->attrs.as<DictAttrsNode>()) {
      for (const auto& [key, value] : attrs->dict) fields.emplace_back(key, value);
    } else {
      ffi::reflection::ForEachFieldInfo(
          TVMFFIGetTypeInfo(call->attrs->type_index()), [&](const TVMFFIFieldInfo* field) {
            fields.emplace_back(ffi::String(field->name),
                                ffi::reflection::FieldGetter(field)(call->attrs));
          });
    }
    std::sort(fields.begin(), fields.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [key, value] : fields) {
      if (key == "ty" || key == "ty_args" || key == "attrs") return std::nullopt;
      keys.push_back(key);
      values.push_back(AnyValue(d, value));
    }
  }
  ffi::Array<ExprDoc> positional = args;
  if (op->name == "relax.call_builtin_with_ctx" && !call->args.empty() &&
      call->args[0].as<StringImmNode>()) {
    // A Python string would be converted to an ExternFunc by this constructor.
    positional.Set(0, AnyValue(d, call->args[0]));
  }
  return NamedCallCallee(names[op])->Call(positional, keys, values);
}

ffi::Optional<ExprDoc> CallTIRDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (!HasRelaxCallResult(call, destination) || call->args.size() != 2 ||
      !call->args[1].as<TupleNode>() || call->ty_args.size() != 1) {
    return RawCall(d, call);
  }
  const Op& op = call->op.as_or_throw<Op>();
  const auto* inplace = call->attrs.as<relax::CallTIRInplaceAttrs>();
  const auto* grad = call->attrs.as<relax::CallTIRWithGradAttrs>();
  if (op->name == "relax.call_tir_inplace"     ? inplace == nullptr
      : op->name == "relax.call_tir_with_grad" ? grad == nullptr
                                               : call->attrs.defined()) {
    return RawCall(d, call);
  }

  const Type& output = call->ty_args[0];
  const auto* tuple = output.as<TupleTypeNode>();
  // The constructor unwraps a single-element output list into a tensor type.
  if (tuple && tuple->fields.size() == 1) return RawCall(d, call);
  ffi::Array<Type> output_types = tuple ? tuple->fields : ffi::Array<Type>{output};
  bool distributed =
      !output_types.empty() && output_types[0].as<relax::distributed::DTensorTypeNode>();
  if (distributed && op->name != "relax.call_tir") return RawCall(d, call);
  ffi::Array<ExprDoc> output_docs;
  for (const Type& type : output_types) {
    if (distributed) {
      const auto* tensor = type.as<relax::distributed::DTensorTypeNode>();
      if (!tensor || !tensor->tensor_ty->shape.as<relax::ShapeExprNode>()) {
        return RawCall(d, call);
      }
    } else {
      const auto* tensor = type.as<relax::TensorTypeNode>();
      if (!tensor || !tensor->shape.as<relax::ShapeExprNode>()) return RawCall(d, call);
    }
    output_docs.push_back(d->Translate(type).value());
  }
  ExprDoc output_doc = tuple ? ExprDoc(ListDoc(output_docs)) : output_docs[0];
  d->RecordOrigin(output_doc, output);
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) args.push_back(d->Translate(arg).value());
  if (op->name == "relax.call_dps_packed" || distributed) {
    if (const auto* packed = call->args[0].as<relax::ExternFuncNode>();
        packed &&
        ffi::StructuralEqual()(packed->ty, relax::ExternFunc(packed->global_symbol)->ty)) {
      ExprDoc symbol = LiteralDoc::Str(packed->global_symbol, std::nullopt);
      d->RecordOrigin(symbol, call->args[0]);
      args.Set(0, symbol);
    } else if (call->args[0].as<StringImmNode>()) {
      args.Set(0, AnyValue(d, call->args[0]));
    }
  }
  ffi::Array<ffi::String> keys = {"out_ty"};
  ffi::Array<ExprDoc> values = {output_doc};
  if (inplace) {
    keys.push_back("inplace_indices");
    values.push_back(AnyValue(d, inplace->inplace_indices));
  } else if (grad) {
    keys.push_back("te_grad_name");
    values.push_back(LiteralDoc::Str(grad->te_grad_name, std::nullopt));
    keys.push_back("te_grad_kwargs");
    values.push_back(AnyValue(d, grad->te_grad_kwargs));
  }
  return NamedCallCallee(distributed ? ffi::String("relax.dist.call_tir") : op->name)
      ->Call(args, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"relax.call_tir", "relax.call_tir_with_grad", "relax.call_tir_inplace",
                           "relax.call_dps_packed"}) {
    OpDef(name).set_attr<FDocTranslate>(kOpCallDocTranslate,
                                        FDocTranslate::FromNative<&CallTIRDocTranslate>());
  }
}

// The name is published only for a callable implementing the shared Op contract.
// Legacy/semantic wrappers retain their existing translators below.
ffi::Optional<ExprDoc> StandardCallDocTranslate(DocTranslatorObj* d, const CallNode* call) {
  auto maybe_op = call->op.as<Op>();
  if (!maybe_op || !Op::HasAttrMap("TScriptStandardCall")) return std::nullopt;
  const Op& op = maybe_op.value();
  if (op->attrs_type_key.empty()) return std::nullopt;
  static const auto& standard = Op::GetAttrMap<tirx::TScriptStandardCall>("TScriptStandardCall");
  static const auto& names = Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
  if (!standard.get(op, false) || !names.count(op) || names[op].empty()) return std::nullopt;
  if (call->attrs.defined() ? op->attrs_type_key != call->attrs->GetTypeKey()
                            : !op->attrs_type_key.empty()) {
    return RawCall(d, call);
  }
  try {
    op.Validate(call);
  } catch (const ffi::Error&) {
    return RawCall(d, call);
  }
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) {
    args.push_back(MaterializeCallArgument(d, arg, d->Translate(arg).value()));
  }
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (!call->ty_args.empty()) {
    ffi::Array<ExprDoc> types;
    for (const Type& type : call->ty_args) types.push_back(TypeValue(d, type, false));
    keys.push_back("ty_args");
    values.push_back(ListDoc(types));
  }
  if (call->attrs.defined()) {
    std::vector<std::pair<ffi::String, ffi::Any>> fields;
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(call->attrs->type_index()), [&](const TVMFFIFieldInfo* field) {
          ffi::Any value = ffi::reflection::FieldGetter(field)(call->attrs);
          // Factory defaults can have effects or change between constructions.
          // Only a reflected literal default is safe to omit here.
          if ((field->flags & kTVMFFIFieldFlagBitMaskHasDefault) &&
              !(field->flags & kTVMFFIFieldFlagBitMaskDefaultFromFactory) &&
              ffi::StructuralEqual()(
                  value, ffi::AnyView::CopyFromTVMFFIAny(field->default_value_or_factory))) {
            return;
          }
          fields.emplace_back(ffi::String(field->name), std::move(value));
        });
    // Without attribute keywords the named builder does not construct attrs.
    // Preserve even an empty or entirely default-valued schema explicitly.
    if (fields.empty()) return RawCall(d, call);
    std::sort(fields.begin(), fields.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [key, value] : fields) {
      if (key == "ty" || key == "ret_ty" || key == "attrs" || key == "ty_args" || key == "span" ||
          key == "type_key" ||
          std::any_of(op->args_info.begin(), op->args_info.end(),
                      [&](const ArgumentInfo& info) { return info->name == key; })) {
        return RawCall(d, call);
      }
      keys.push_back(key);
      values.push_back(AnyValue(d, value));
    }
  }
  bool omit_result = false;
  try {
    // Match construction without ty, rather than let inference read the stored result.
    Call provisional =
        Call::Unchecked(Type::Missing(), call->op, call->args, call->attrs, call->ty_args);
    omit_result = ffi::StructuralEqual()(Call::ReinferType(provisional.get()), call->ty);
  } catch (const ffi::Error&) {
    // Missing information, unavailable inference, or errors require the exact stored type.
  }
  if (!omit_result) {
    keys.push_back("ty");
    values.push_back(TypeValue(d, call->ty));
  }
  return NamedCallCallee(names[op])->Call(args, keys, values);
}

ffi::Optional<ExprDoc> CallDefaultDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (auto doc = StandardCallDocTranslate(d, call)) return doc;
  if (auto op = call->op.as<Op>(); op && op.value()->name.find("relax.") == 0) {
    if (HasRelaxCallResult(call, destination)) {
      ffi::Array<ExprDoc> args;
      for (const Expr& arg : call->args) args.push_back(d->Translate(arg).value());
      if (auto doc = RelaxCallDocTranslate(d, call, args)) return doc;
      return RawCall(d, call, args);
    }
    return RawCall(d, call);
  }
  if ((call->op.as<VarNode>() || call->op.as<GlobalVarNode>() ||
       call->op.as<relax::FunctionNode>()) &&
      call->op->ty.as<relax::FuncTypeNode>() && HasRelaxCallResult(call, destination) &&
      !call->attrs.defined() && call->ty_args.empty()) {
    ffi::Array<ExprDoc> args;
    for (const Expr& arg : call->args) {
      args.push_back(MaterializeCallArgument(d, arg, d->Translate(arg).value()));
    }
    return d->Translate(call->op).value()->Call(args);
  }
  ffi::Optional<Type> inferred = std::nullopt;
  if (call->op.as<Op>() && std::all_of(call->args.begin(), call->args.end(), [](const Expr& arg) {
        return !arg->ty.as<MissingType>().has_value();
      })) {
    try {
      inferred = Call::ReinferType(call);
    } catch (const ffi::Error&) {
      // A registered hook may require information absent from this Call.
    }
  }
  if (!inferred || !ffi::StructuralEqual()(inferred.value(), call->ty)) {
    return RawCall(d, call);
  }
  const Type& result_type = inferred.value();
  if (auto doc = TIRCallPrefixDocTranslate(d, call)) return doc;
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) args.push_back(d->Translate(arg).value());
  if (auto doc = FFIKernelDocTranslate(d, call, result_type, args)) return doc;
  if (auto doc = TIRCallDocTranslate(d, call, result_type, args)) return doc;
  return RawCall(d, call, args);
}

ffi::Optional<ExprDoc> CallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (auto op = call->op.as<Op>(); op && Op::HasAttrMap(kOpCallDocTranslate)) {
    static const auto& overrides = Op::GetAttrMap<ffi::Any>(kOpCallDocTranslate);
    if (overrides.count(op.value())) {
      ffi::Any hook = overrides[op.value()];
      Call object = ffi::GetRef<Call>(call);
      if (hook.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
        return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Optional<ExprDoc>>(
                   reinterpret_cast<decltype(DocTranslatorVTable::translate)>(hook.cast<void*>())(
                       d, object, destination))
            .value();
      }
      ffi::Any destination_arg = nullptr;
      if (destination) destination_arg = ffi::GetRef<ffi::ObjectRef>(destination);
      return hook.cast<ffi::Function>()
          .CallExpected<ffi::Optional<ExprDoc>>(d, object, destination_arg)
          .value();
    }
  }
  return CallDefaultDocTranslate(d, input, destination);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<CallNode>().attr(kDocTranslate,
                                                FDocTranslate::FromNative<&CallDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
