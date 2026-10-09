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
#include <tvm/ir/op.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <string>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ExprDoc TensorOperandDoc(DocTranslatorObj* d, const Expr& value) {
  if (const auto* region = value.as<TensorRegionNode>()) {
    ExprDoc doc = TensorRegionValue(d, region, true);
    d->RecordOrigin(doc, value);
    return doc;
  }
  if (const auto* tuple = value.as<TupleNode>()) {
    if (tuple->fields.empty()) return LiteralDoc::None(std::nullopt);
    ffi::Array<ExprDoc> fields;
    for (const auto& field : tuple->fields) fields.push_back(TensorOperandDoc(d, field));
    return TupleDoc(fields);
  }
  return AnyValue(d, value);
}

ffi::Optional<ExprDoc> TensorCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  const auto op = call->op.as_or_throw<Op>();
  op.Validate(call);
  ffi::Array<ExprDoc> args;
  for (const auto& arg : call->args) args.push_back(TensorOperandDoc(d, arg));
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  ffi::reflection::ForEachFieldInfo(
      TVMFFIGetTypeInfo(call->attrs->type_index()), [&](const TVMFFIFieldInfo* field) {
        ffi::Any value = ffi::reflection::FieldGetter(field)(call->attrs);
        if ((field->flags & kTVMFFIFieldFlagBitMaskHasDefault) &&
            ffi::StructuralEqual()(
                value, ffi::AnyView::CopyFromTVMFFIAny(field->default_value_or_factory)))
          return;
        keys.push_back(ffi::String(field->name));
        values.push_back(AnyValue(d, value));
      });
  static const auto& names = Op::GetAttrMap<TScriptPrinterName>("TScriptPrinterName");
  return NamedCallCallee(names[op])->Call(args, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("script.printer.TensorCallDocTranslate", []() {
    return FDocTranslate::FromNative<&TensorCallDocTranslate>();
  });
}

ffi::Optional<ExprDoc> TIRxRegionDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* stmt =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const RegionStmtNode>(input);
  if (stmt->op.same_as(tirx::device_entry_op())) {
    return NamespaceDoc("tirx")->Attr("device_entry")->Call({});
  } else if (stmt->op.same_as(tirx::launch_thread_op())) {
    return NamespaceDoc("tirx")
        ->Attr("launch_thread")
        ->Call({LiteralDoc::Str(stmt->args[0].as_or_throw<StringImm>()->value, std::nullopt),
                d->Translate(stmt->args[1].as_or_throw<PrimExpr>()).value()});
  } else if (stmt->op.same_as(tirx::device_context_op())) {
    return NamespaceDoc("tirx")
        ->Attr("device_context")
        ->Call({d->Translate(stmt->args[0]).value(), d->Translate(stmt->args[1]).value()});
  } else if (stmt->op.same_as(tirx::compute_scope_op())) {
    return NamespaceDoc("tirx")
        ->Attr("compute_scope")
        ->Call({LiteralDoc::Str(stmt->args[0].as_or_throw<StringImm>()->value, std::nullopt)});
  } else if (stmt->op.same_as(tirx::parallel_launch_op())) {
    return NamespaceDoc("tirx")->Attr("parallel_launch")->Call({});
  }
  TVM_FFI_THROW(TypeError) << "Unexpected region operator " << stmt->op->name;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"tirx.device_entry", "tirx.launch_thread", "tirx.device_context",
                           "tirx.compute_scope", "tirx.parallel_launch"}) {
    OpDef(name).set_attr<FDocTranslate>(op_attr::kRegionDocTranslate,
                                        FDocTranslate::FromNative<&TIRxRegionDocTranslate>());
  }
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
