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
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op_attr_types.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

ffi::Optional<ExprDoc> VarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const VarNode>(input);
  Var var = ffi::GetRef<Var>(node);
  IdDoc id = d->VarGetOrAllocId(var, false);
  if (destination == node && d->GetImplicitDefs().count(var)) {
    // Promote before translating the type, which may refer back to this Var.
    VarDoc(d, var);
    ffi::Optional<ExprDoc> rhs = std::nullopt;
    ffi::Optional<ExprDoc> annotation = std::nullopt;
    if (auto primitive = var->ty.as<PrimType>()) {
      rhs = NamespaceDoc("ir")->Attr("dynamic")->Call(
          {LiteralDoc::Str(var->name, std::nullopt)}, {"dtype"},
          {LiteralDoc::DataType(primitive.value()->dtype, std::nullopt)});
    } else if (var->ty.as<tirx::TensorTypeNode>()) {
      rhs = NamespaceDoc("tirx")->Attr("Var")->Call(
          {LiteralDoc::Str(var->name, std::nullopt), d->Translate(var->ty).value()});
    } else {
      annotation = d->Translate(var->ty).value();
      if (var->ty.as<PointerTypeNode>()) {
        // A module-level annotation alone does not bind a Python variable.
        rhs = annotation.value().as<CallDoc>() ? annotation : annotation.value()->Call({});
      }
    }
    // Only this type's referenced Vars must precede its declaration.
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        var->ty, [&](const Var& dependency) -> ffi::Expected<ffi::WalkResult> {
          if (d->GetImplicitDefs().count(dependency)) {
            if (auto rhs = d->Translate(dependency, dependency)) {
              d->Emit(AssignDoc(VarDoc(d, dependency), rhs.value(), std::nullopt), dependency);
            }
          }
          return ffi::WalkResult::Skip();
        });
    d->Emit(AssignDoc(VarDoc(d, var), rhs, annotation), var);
    return std::nullopt;
  }
  // Mutable scalar syntax binds a TensorLoad; resource uses need its buffer.
  if (IsScalarBuffer(d, var)) return IdDoc(id->name)->Attr("source");
  return IdDoc(id->name);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<VarNode>().attr(kDocTranslate,
                                               FDocTranslate::FromNative<&VarDocTranslate>());
}

bool CanTranslateExplicitResultCall(const CallNode* call) {
  return !call->attrs.defined() && call->ty_args.empty() && call->ty.as<PrimType>() &&
         std::all_of(call->args.begin(), call->args.end(), [](const Expr& arg) {
           return !arg->ty.as<MissingType>().has_value() && !arg.as<TensorRegionNode>();
         });
}

namespace {

ffi::Optional<ExprDoc> IndexMapDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* map =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::IndexMapNode>(input);
  auto translate_lambda = [&](const tirx::IndexMapNode* node) -> ExprDoc {
    ffi::Array<IdDoc> vars;
    for (const PrimVar& var : node->initial_indices) vars.push_back(VarDoc(d, var));
    ffi::Array<ExprDoc> values;
    for (const PrimExpr& expr : node->final_indices) values.push_back(d->Translate(expr).value());
    return LambdaDoc(vars, TupleDoc(values));
  };
  ExprDoc forward = translate_lambda(map);
  if (map->inverse_index_map.has_value()) {
    const auto* inverse = map->inverse_index_map.value().as<tirx::IndexMapNode>();
    TVM_FFI_CHECK(inverse, TypeError) << "IndexMap inverse must be an IndexMap";
    return NamespaceDoc("tirx")
        ->Attr("index_map")
        ->Call({forward}, {"inverse_index_map"}, {translate_lambda(inverse)});
  }
  return NamespaceDoc("tirx")->Attr("index_map")->Call({forward});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::IndexMapNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&IndexMapDocTranslate>());
}

ffi::Optional<ExprDoc> LLVMIntrinsicDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (call->attrs.defined() || !call->ty_args.empty() || call->args.empty() ||
      !call->ty.as<PrimType>()) {
    return RawCall(d, call);
  }
  const auto* id = call->args[0].as<IntImmNode>();
  // The named constructor uses an int32 intrinsic identifier. Other stored
  // representations retain their exact operand type through the full Call.
  if (!id || !ffi::StructuralEqual()(id->ty, PrimType::Int(32))) return RawCall(d, call);
  auto lookup = ffi::Function::GetGlobal("target.llvm_get_intrinsic_name");
  auto reverse = ffi::Function::GetGlobal("target.llvm_lookup_intrinsic_id");
  if (!lookup || !reverse) return RawCall(d, call);
  ffi::String name;
  try {
    name = (*lookup)(static_cast<int64_t>(id->value)).cast<ffi::String>();
    if (name.empty() || (*reverse)(name).cast<int64_t>() != static_cast<int64_t>(id->value)) {
      return RawCall(d, call);
    }
  } catch (const ffi::Error&) {
    return RawCall(d, call);
  }
  ExprDoc name_doc = LiteralDoc::Str(name, std::nullopt);
  d->RecordOrigin(name_doc, call->args[0]);
  ffi::Array<ExprDoc> args = {name_doc};
  for (size_t i = 1; i < call->args.size(); ++i) {
    args.push_back(MaterializeCallArgument(d, call->args[i], d->Translate(call->args[i]).value()));
  }
  return NamespaceDoc("tirx")
      ->Attr(call->op.same_as(tirx::call_llvm_intrin_op()) ? "call_llvm_intrin"
                                                           : "call_llvm_pure_intrin")
      ->Call(args, {"ty"}, {TypeValue(d, call->ty)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"tirx.call_llvm_intrin", "tirx.call_llvm_pure_intrin"}) {
    OpDef(name).set_attr<FDocTranslate>(kOpCallDocTranslate,
                                        FDocTranslate::FromNative<&LLVMIntrinsicDocTranslate>());
  }
}

bool IsPTXAddressCall(const CallNode* call) {
  auto op = call->op.as<Op>();
  return op && op.value()->name == "tirx.ptx.addr";
}

ffi::Optional<ExprDoc> ConsumedPTXAddressDocTranslate(DocTranslatorObj* d, const CallNode* call) {
  if (call->args.size() != 2 || call->attrs.defined() || !call->ty_args.empty())
    return std::nullopt;
  const Expr& base = call->args[0];
  const Expr& offset = call->args[1];
  if (const auto* nested = base.as<CallNode>(); nested && IsPTXAddressCall(nested)) {
    return std::nullopt;
  }
  if ((!base->ty.as<PointerType>() && !ffi::StructuralEqual()(base->ty, PrimType::UInt(32))) ||
      !ffi::StructuralEqual()(call->ty, base->ty)) {
    return std::nullopt;
  }
  auto offset_type = offset->ty.as<PrimType>();
  if (!offset_type || !offset_type.value().IsScalar() ||
      !offset_type.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) ||
      (offset_type.value().bits() != 8 && offset_type.value().bits() != 16 &&
       offset_type.value().bits() != 32 && offset_type.value().bits() != 64)) {
    return std::nullopt;
  }
  if (const auto* value = offset.as<IntImmNode>();
      value && (!ffi::StructuralEqual()(offset->ty, PrimType::Int(32)) ||
                value->value < std::numeric_limits<int32_t>::min() ||
                value->value > std::numeric_limits<int32_t>::max())) {
    return std::nullopt;
  }
  ExprDoc result = NamespaceDoc("tirx")->Attr("ptx")->Attr("addr")->Call(
      {d->Translate(base).value(), d->Translate(offset).value()});
  d->RecordOrigin(result, ffi::GetRef<Expr>(call));
  return result;
}

}  // namespace

namespace {

// Pointer helpers have semantic type parameters outside the stored value arguments.
ffi::Optional<ExprDoc> PointerCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  return RawCall(d, call);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"tirx.access_ptr", "tirx.ptr_byte_offset"}) {
    OpDef(name).set_attr<FDocTranslate>(kOpCallDocTranslate,
                                        FDocTranslate::FromNative<&PointerCallDocTranslate>());
  }
}

ffi::Optional<ExprDoc> TensorDataPtrDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (call->attrs.defined() || !call->ty_args.empty() || call->args.size() != 1)
    return RawCall(d, call);
  try {
    if (!ffi::StructuralEqual()(Call::ReinferType(call), call->ty)) return RawCall(d, call);
  } catch (const ffi::Error&) {
    return RawCall(d, call);
  }
  return d->Translate(call->args[0]).value()->Attr("data");
}

// PTX modifiers and operand tags use a dedicated reconstruction surface.
// This hook is installed by the backend when each table entry is registered.
ffi::Optional<ExprDoc> PTXCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (IsPTXAddressCall(call) || call->attrs.defined() || !call->ty_args.empty() ||
      !ffi::StructuralEqual()(call->ty, PrimType::Void()))
    return RawCall(d, call);
  const Op& op = call->op.as_or_throw<Op>();
  static const auto& names = Op::GetAttrMap<TScriptPrinterName>("TScriptPrinterName");
  if (!names.count(op)) return RawCall(d, call);
  auto can_roundtrip = ffi::Function::GetGlobal("script.printer.PTXCallCanRoundtrip");
  if (!can_roundtrip || !(*can_roundtrip)(ffi::GetRef<Call>(call)).cast<bool>())
    return RawCall(d, call);
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) {
    ExprDoc value = d->Translate(arg).value();
    if (const auto* string = arg.as<StringImmNode>()) {
      value = LiteralDoc::Str(string->value, std::nullopt);
    } else if (const auto* address = arg.as<CallNode>(); address && IsPTXAddressCall(address)) {
      auto translated = ConsumedPTXAddressDocTranslate(d, address);
      if (!translated) return RawCall(d, call);
      value = translated.value();
    } else if (arg.as<IntImmNode>() && ffi::StructuralEqual()(arg->ty, PrimType::Int(32))) {
      value = NamespaceDoc("tirx")->Attr("int32")->Call({value});
    } else if (arg->ty.as<MissingType>() || arg.as<TensorRegionNode>()) {
      return RawCall(d, call);
    }
    args.push_back(value);
  }
  return NamedCallCallee(names[op])->Call(args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tensor_data_ptr")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&TensorDataPtrDocTranslate>());
  ffi::reflection::GlobalDef().def("script.printer.PTXCallDocTranslate", []() {
    return ffi::Any(FDocTranslate::FromNative<&PTXCallDocTranslate>());
  });
}

}  // namespace
}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
