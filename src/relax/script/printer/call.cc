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
#include <tvm/relax/distributed/type.h>
#include <tvm/relax/op/op.h>

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

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

ffi::Optional<ExprDoc> RelaxSugarDocTranslate(DocTranslatorObj* d, const CallNode* call,
                                              const ffi::Array<ExprDoc>& args) {
  const Op& op = call->op.as_or_throw<Op>();

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

  return std::nullopt;
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

ffi::Optional<ExprDoc> RelaxSugarCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                  const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (HasRelaxCallResult(call, destination)) {
    ffi::Array<ExprDoc> args;
    for (const Expr& arg : call->args) args.push_back(d->Translate(arg).value());
    if (auto doc = RelaxSugarDocTranslate(d, call, args)) return doc;
    return RawCall(d, call, args);
  }
  return RawCall(d, call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"relax.call_py_func"}) {
    OpDef(name).set_attr<FDocTranslate>(kOpCallDocTranslate,
                                        FDocTranslate::FromNative<&RelaxSugarCallDocTranslate>());
  }
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
