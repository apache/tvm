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
#include <tvm/relax/expr.h>
#include <tvm/script/printer/doc_translator.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {
namespace {

// These semantic wrappers convert bare Python scalar tuple fields to 64-bit
// values. Use raw construction when ordinary tuple syntax would change the IR.
bool InlineTupleNeedsRawCall(DocTranslatorObj* d, const Expr& expr) {
  if (const auto* tuple = expr.as<TupleNode>()) {
    for (const Expr& field : tuple->fields) {
      if (InlineTupleNeedsRawCall(d, field)) return true;
    }
  } else if (const auto* imm = expr.as<IntImmNode>()) {
    DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
    auto implicit =
        ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.int_dtype", "int32"));
    return dtype == implicit && dtype != ffi::StringToDLDataType("int64");
  } else if (const auto* imm = expr.as<FloatImmNode>()) {
    DLDataType dtype = imm->ty.as_or_throw<PrimType>()->dtype;
    auto implicit =
        ffi::StringToDLDataType(d->GetExtraConfig<ffi::String>("ir.float_dtype", "void"));
    return dtype == implicit && dtype != ffi::StringToDLDataType("float64");
  }
  return false;
}

ffi::Optional<ExprDoc> InlineTupleCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                   const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (call->args.size() > 1 && InlineTupleNeedsRawCall(d, call->args[1])) return RawCall(d, call);
  if (auto doc = StandardCallDocTranslate(d, call)) return doc;
  return RawCall(d, call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"relax.call_builtin_with_ctx", "relax.make_closure",
                           "relax.invoke_closure", "relax.invoke_pure_closure"}) {
    OpDef(name).set_attr<FDocTranslate>(tvm::script::printer::op_attr::kOpCallDocTranslate,
                                        FDocTranslate::FromNative<&InlineTupleCallDocTranslate>());
  }
}

ffi::Optional<ExprDoc> CallDPSPackedDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (call->attrs.has_value() || call->args.size() != 2 || !call->args[1].as<TupleNode>() ||
      call->ty_args.size() != 1 || InlineTupleNeedsRawCall(d, call->args[1])) {
    return RawCall(d, call);
  }
  ExprDoc callee = d->Translate(call->args[0]).value();
  if (const auto* packed = call->args[0].as<relax::ExternFuncNode>()) {
    if (!ffi::StructuralEqual()(packed->ty, relax::ExternFunc(packed->global_symbol)->ty)) {
      return RawCall(d, call);
    }
    callee = LiteralDoc::Str(packed->global_symbol, std::nullopt);
  } else if (call->args[0].as<StringImmNode>()) {
    // A bare Python string now constructs ExternFunc, so retain this explicit IR value.
    callee = AnyValue(d, call->args[0]);
  }
  d->RecordOrigin(callee, call->args[0]);
  ExprDoc args = d->Translate(call->args[1]).value();
  ffi::Array<ffi::String> keys = {"ty_args"};
  ffi::Array<ExprDoc> values = {ListDoc({TypeValue(d, call->ty_args[0], false)})};
  bool omit_result = false;
  try {
    Call inferred(std::nullopt, call->op, call->args, call->attrs, call->ty_args);
    omit_result = ffi::StructuralEqual()(inferred->ty, call->ty);
  } catch (const ffi::Error&) {
    // An explicit result must also survive when result inference is unavailable.
  }
  if (!omit_result) {
    keys.push_back("ty");
    values.push_back(TypeValue(d, call->ty));
  }
  return NamespaceDoc("relax")->Attr("call_dps_packed")->Call({callee, args}, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("relax.call_dps_packed")
      .set_attr<FDocTranslate>(tvm::script::printer::op_attr::kOpCallDocTranslate,
                               FDocTranslate::FromNative<&CallDPSPackedDocTranslate>());
}

}  // namespace
}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
