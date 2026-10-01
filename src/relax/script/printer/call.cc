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
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/op.h>
#include <tvm/relax/attrs/op.h>
#include <tvm/relax/distributed/type.h>
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

namespace {

ffi::Optional<ExprDoc> TranslateRelaxCall(DocTranslatorObj* d, const CallNode* call,
                                          ffi::Array<ExprDoc>* translated_args) {
  auto& args = *translated_args;
  auto relax_format = [&](size_t index) -> ExprDoc {
    if (const auto* format = call->args[index].as<StringImmNode>()) {
      return NamespaceDoc("relax")->Attr("str")->Call(
          {LiteralDoc::Str(format->value, std::nullopt)});
    }
    return args[index];
  };

  if (auto op = call->op.as<Op>(); op.has_value() && op.value()->name == "relax.call_py_func" &&
                                   call->args.size() == 2 && !call->attrs.defined() &&
                                   call->ty_args.size() == 1) {
    ffi::Array<ExprDoc> packed_args = {d->Translate(call->args[0]).value()};
    const auto* tuple = call->args[1].as<TupleNode>();
    if (!tuple) return RawCall(d, call, true, args);
    for (const Expr& arg : tuple->fields) packed_args.push_back(d->Translate(arg).value());
    return NamespaceDoc("relax")
        ->Attr("call_py_func")
        ->Call(packed_args, {"out_ty"}, {d->Translate(call->ty_args[0]).value()});
  }

  if (auto op = call->op.as<Op>(); op.has_value() && op.value()->name == "relax.print" &&
                                   !call->args.empty() && !call->attrs.defined() &&
                                   call->ty_args.empty()) {
    ffi::Array<ExprDoc> positional;
    for (size_t i = 1; i < args.size(); ++i) positional.push_back(args[i]);
    return NamespaceDoc("relax")->Attr("print")->Call(positional, {"format"}, {relax_format(0)});
  }

  if (auto op = call->op.as<Op>(); op.has_value() && op.value()->name == "relax.assert_op" &&
                                   call->args.size() >= 2 && !call->attrs.defined() &&
                                   call->ty_args.empty()) {
    ffi::Array<ExprDoc> positional = {args[0]};
    for (size_t i = 2; i < args.size(); ++i) positional.push_back(args[i]);
    return NamespaceDoc("relax")
        ->Attr("assert_op")
        ->Call(positional, {"format"}, {relax_format(1)});
  }

  if (auto op = call->op.as<Op>(); op.has_value() && op.value()->name == "relax.hint_on_device") {
    const auto* attrs = call->attrs.as<relax::HintOnDeviceAttrs>();
    if (!attrs || args.size() != 1 || !call->ty_args.empty()) {
      return RawCall(d, call, true, args);
    }
    ExprDoc device = NamespaceDoc("relax")->Attr("device")->Call(
        {LiteralDoc::Int(attrs->device_type, std::nullopt),
         LiteralDoc::Int(attrs->index, std::nullopt)});
    return NamespaceDoc("relax")
        ->Attr("hint_on_device")
        ->Call({args[0], device, LiteralDoc::Str(attrs->memory_scope, std::nullopt)});
  }

  ffi::Optional<Op> op = call->op.as<Op>();
  static const auto& names = Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
  ffi::Optional<ffi::String> canonical = std::nullopt;
  if (names.count(op.value())) canonical = names[op.value()];
  bool specialized_tir =
      op.value()->name == "relax.call_tir" || op.value()->name == "relax.call_tir_with_grad" ||
      op.value()->name == "relax.call_tir_inplace" || op.value()->name == "relax.call_dps_packed";
  bool attrs_match =
      !call->attrs.defined() || (op && op.value()->attrs_type_key == call->attrs->GetTypeKey());
  bool reserved_attrs = false;
  if (call->attrs.defined()) {
    auto check_key = [&](const ffi::String& key) {
      reserved_attrs |= key == "ty" || key == "ty_args" || key == "attrs";
    };
    if (auto attrs = call->attrs.as<DictAttrsNode>()) {
      for (const auto& [key, value] : attrs->dict) check_key(key);
    } else {
      ffi::reflection::ForEachFieldInfo(
          TVMFFIGetTypeInfo(call->attrs->type_index()),
          [&](const TVMFFIFieldInfo* field) { check_key(ffi::String(field->name)); });
    }
  }
  if (attrs_match && !reserved_attrs && (specialized_tir || call->ty_args.empty()) &&
      (specialized_tir || (canonical && canonical.value().find("relax.") == 0))) {
    std::string name = specialized_tir ? op.value()->name.substr(6) : canonical.value().substr(6);
    if (name == "call_tir" && call->ty_args.size() == 1) {
      const Type& output = call->ty_args[0];
      bool distributed = output.as<relax::distributed::DTensorTypeNode>() != nullptr;
      if (auto tuple = output.as<relax::TupleTypeNode>()) {
        for (const Type& field : tuple->fields) {
          distributed |= field.as<relax::distributed::DTensorTypeNode>() != nullptr;
        }
      }
      if (distributed) name = "dist.call_tir";
    }
    ExprDoc callee = NamespaceDoc("relax");
    size_t start = 0;
    while (start < name.size()) {
      size_t end = name.find('.', start);
      callee = callee->Attr(name.substr(start, end - start));
      if (end == std::string::npos) break;
      start = end + 1;
    }
    ffi::Array<ffi::String> keys;
    ffi::Array<ExprDoc> values;
    if (call->attrs.defined()) {
      std::vector<std::pair<ffi::String, ffi::Any>> fields;
      if (auto attrs = call->attrs.as<DictAttrsNode>()) {
        for (const auto& [key, value] : attrs->dict) fields.emplace_back(key, value);
      } else {
        const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(call->attrs->type_index());
        ffi::reflection::ForEachFieldInfo(info, [&](const TVMFFIFieldInfo* field) {
          fields.emplace_back(ffi::String(field->name),
                              ffi::reflection::FieldGetter(field)(call->attrs));
        });
      }
      std::sort(fields.begin(), fields.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
      for (const auto& [key, value] : fields) {
        keys.push_back(key);
        values.push_back(AnyValue(d, value));
      }
    }
    if (!call->ty_args.empty()) {
      if ((op.value()->name == "relax.call_tir" || op.value()->name == "relax.call_dps_packed" ||
           op.value()->name == "relax.call_tir_inplace" ||
           op.value()->name == "relax.call_tir_with_grad") &&
          call->ty_args.size() == 1) {
        keys.push_back("out_ty");
        values.push_back(d->Translate(call->ty_args[0]).value());
      } else {
        ffi::Array<ExprDoc> types;
        for (const Type& type : call->ty_args) types.push_back(d->Translate(type).value());
        keys.push_back("ty_args");
        values.push_back(TupleDoc(types));
      }
    }
    if ((op.value()->name == "relax.call_tir_with_grad" ||
         op.value()->name == "relax.call_tir_inplace") &&
        call->ty_args.size() == 1) {
      ffi::Array<ffi::String> ordered_keys = {"out_ty"};
      ffi::Array<ExprDoc> ordered_values;
      if (op.value()->name == "relax.call_tir_inplace") {
        if (auto tuple = call->ty_args[0].as<relax::TupleTypeNode>()) {
          ffi::Array<ExprDoc> fields;
          for (const Type& field : tuple->fields) fields.push_back(d->Translate(field).value());
          ordered_values.push_back(ListDoc(fields));
        } else {
          ordered_values.push_back(d->Translate(call->ty_args[0]).value());
        }
      } else {
        ordered_values.push_back(d->Translate(call->ty_args[0]).value());
      }
      auto append_key = [&](const ffi::String& key) {
        for (size_t i = 0; i < keys.size(); ++i) {
          if (keys[i] == key) {
            ordered_keys.push_back(keys[i]);
            ordered_values.push_back(values[i]);
            break;
          }
        }
      };
      if (op.value()->name == "relax.call_tir_with_grad") {
        append_key("te_grad_name");
        append_key("te_grad_kwargs");
      } else {
        append_key("inplace_indices");
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == "out_ty" ||
            (op.value()->name == "relax.call_tir_with_grad" &&
             (keys[i] == "te_grad_name" || keys[i] == "te_grad_kwargs")) ||
            (op.value()->name == "relax.call_tir_inplace" && keys[i] == "inplace_indices"))
          continue;
        ordered_keys.push_back(keys[i]);
        ordered_values.push_back(values[i]);
      }
      return callee->Call(args, ordered_keys, ordered_values);
    }
    if (op.value()->name == "relax.call_dps_packed" && !args.empty()) {
      if (const auto* packed = call->args[0].as<relax::ExternFuncNode>();
          packed && !packed->ty.IsMissing() &&
          ffi::StructuralEqual()(packed->ty, relax::ExternFunc(packed->global_symbol)->ty)) {
        ExprDoc symbol = LiteralDoc::Str(packed->global_symbol, std::nullopt);
        d->RecordOrigin(symbol, call->args[0]);
        args.Set(0, symbol);
      }
    }
    if ((op.value()->name == "relax.call_dps_packed" ||
         op.value()->name == "relax.call_builtin_with_ctx") &&
        !call->args.empty() && call->args[0].as<StringImmNode>()) {
      // These Python constructors interpret a string literal as an ExternFunc symbol.
      args.Set(0, AnyValue(d, call->args[0]));
    }
    return callee->Call(args, keys, values);
  }

  return std::nullopt;
}

ffi::Optional<ExprDoc> CallDefaultDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  // Eligibility depends on inputs and the registered inference contract, not
  // on equality to a potentially stale stored result type.
  ffi::Optional<Type> inferred = std::nullopt;
  if (auto op = call->op.as<Op>();
      op && std::all_of(call->args.begin(), call->args.end(),
                        [](const Expr& arg) { return !arg->ty.IsMissing(); })) {
    try {
      op.value().Validate(call);
      if (destination && destination->IsInstance<VarNode>() &&
          op.value()->name.find("relax.") == 0) {
        const auto* var = static_cast<const VarNode*>(destination);
        // A typed Relax binding carries the result type. Its deferred
        // constructor is resolved by the parser in that binding context.
        if (!var->ty.IsMissing() && ffi::StructuralEqual()(var->ty, call->ty)) {
          ffi::Array<ExprDoc> args;
          for (const Expr& arg : call->args) args.push_back(d->Translate(arg).value());
          if (auto doc = TranslateRelaxCall(d, call, &args)) return doc;
          return RawCall(d, call, false, args);
        }
      }
      inferred = Call::ReinferType(call);
    } catch (const ffi::Error&) {
      // A registered hook may require information absent from this Call.
    }
  }
  if (!inferred.has_value()) return RawCall(d, call, false);
  const Type& result_type = inferred.value();
  if (auto doc = TranslateTIRCallPrefix(d, call)) return doc;
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) args.push_back(d->Translate(arg).value());
  if (auto doc = TranslateFFIKernel(d, call, result_type, args)) return doc;
  if (auto doc = TranslateRelaxCall(d, call, &args)) return doc;
  if (auto doc = TranslateTIRCall(d, call, result_type, args)) return doc;
  return RawCall(d, call, true, args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"relax.call_tir", "relax.call_tir_with_grad", "relax.call_tir_inplace",
                           "relax.call_dps_packed"}) {
    OpDef(name).set_attr<FDocTranslate>(kOpCallTranslate,
                                        FDocTranslate::FromNative<&CallDefaultDocTranslate>());
  }
}

ffi::Optional<ExprDoc> CallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (auto op = call->op.as<Op>(); op && Op::HasAttrMap(kOpCallTranslate)) {
    static const auto& overrides = Op::GetAttrMap<ffi::Any>(kOpCallTranslate);
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
