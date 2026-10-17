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
#include <tvm/script/printer/doc_translator.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

// Semantic wrappers register their own Op hooks; ordinary calls use canonical exposure.

namespace {

ffi::Optional<ExprDoc> CallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (auto op = call->op.as<Op>();
      op && Op::HasAttrMap(tvm::script::printer::op_attr::kDocTranslateOpCall)) {
    static const auto& overrides =
        Op::GetAttrMap<ffi::Any>(tvm::script::printer::op_attr::kDocTranslateOpCall.name);
    if (overrides.count(op.value())) {
      ffi::Any hook = overrides[op.value()];
      if (hook.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
        return op_attr::kDocTranslateOpCall.ValueFrom(hook)
            .value()
            .CallExpected(d, input, destination)
            .value();
      }
      ffi::Any binding = nullptr;
      if (destination) binding = ffi::GetRef<ffi::ObjectRef>(destination);
      return hook.cast<ffi::Function>()
          .CallExpected<ffi::Optional<ExprDoc>>(d, input, binding)
          .value();
    }
  }
  if (auto doc = StandardCallDocTranslate(d, call)) return doc;
  return RawCallDocTranslate(d, call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<CallNode>().attr(tvm::script::printer::type_attr::kDocTranslate.name,
                                                FDocTranslate::FromNative<&CallDocTranslate>());
}

}  // namespace
}  // namespace details

ffi::Optional<ExprDoc> StandardCallDocTranslate(DocTranslatorObj* d, const CallNode* call) {
  auto maybe_op = call->op.as<Op>();
  if (!maybe_op) return std::nullopt;
  const Op& op = maybe_op.value();
  if (call->attrs.has_value() ? op->attrs_type_key != call->attrs->GetTypeKey()
                              : !op->attrs_type_key.empty()) {
    return RawCallDocTranslate(d, call);
  }
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) {
    args.push_back(d->Translate(arg).value());
  }
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (!call->ty_args.empty()) {
    ffi::Array<ExprDoc> types;
    for (const Type& type : call->ty_args) types.push_back(d->Translate(type).value());
    keys.push_back("ty_args");
    values.push_back(ListDoc(types));
  }
  bool default_attrs = false;
  if (call->attrs.has_value() && call->attrs.value().as<DictAttrsNode>()) {
    default_attrs = call->attrs.value().as_or_throw<DictAttrs>()->dict.empty();
    if (!default_attrs) {
      keys.push_back("attrs");
      values.push_back(d->Translate(call->attrs.value()).value());
    }
  } else if (call->attrs.has_value()) {
    std::vector<std::pair<ffi::String, ffi::Any>> fields;
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(call->attrs->type_index()), [&](const TVMFFIFieldInfo* field) {
          ffi::Any value = ffi::reflection::FieldGetter(field)(call->attrs.value());
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
    // Named constructors supply the registered attribute schema and defaults.
    default_attrs = fields.empty();
    std::sort(fields.begin(), fields.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& [key, value] : fields) {
      if (key == "ty" || key == "attrs" || key == "ty_args" || key == "loc" || key == "type_key" ||
          std::any_of(op->args_info.begin(), op->args_info.end(),
                      [&](const ArgumentInfo& info) { return info->name == key; })) {
        return RawCallDocTranslate(d, call);
      }
      keys.push_back(key);
      values.push_back(d->WithExtraStateScope<bool>("ir.explicit_value_type", true,
                                                    [&] { return d->Translate(value).value(); }));
    }
  }
  // Well-formed calls reconstruct their result from registered inference or a fixed type.
  // Builder-only inference is deliberately excluded: named construction cannot run it.
  bool omit_result = call->ty.as<MissingType>().has_value() ||
                     (Op::HasAttrMap(tvm::op_attr::kInferType) &&
                      Op::GetAttrMap(tvm::op_attr::kInferType).count(op)) ||
                     (Op::HasAttrMap(tvm::op_attr::kFixedReturnType) &&
                      Op::GetAttrMap(tvm::op_attr::kFixedReturnType).count(op));
  if (!omit_result) {
    if (default_attrs) return RawCallDocTranslate(d, call);
    keys.push_back("ty");
    values.push_back(d->Translate(call->ty).value());
  }
  return details::NamedCallCallee(op->name)->Call(args, keys, values);
}

}  // namespace printer
}  // namespace script
}  // namespace tvm
