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
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/index_map.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/tile_primitive.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>

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

ffi::Optional<ExprDoc> StorageSyncDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (!CanTranslateExplicitResultCall(call) || call->args.empty() || call->args.size() > 3) {
    return RawCall(d, call);
  }
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) {
    args.push_back(MaterializeCallArgument(d, arg, d->Translate(arg).value()));
  }
  // Explicit None operands suppress the helper's defaults without changing
  // the stored argument list of native one- and two-operand calls.
  while (args.size() < 3) args.push_back(LiteralDoc::None(std::nullopt));
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (!ffi::StructuralEqual()(call->ty, PrimType::Void())) {
    keys.push_back("dtype");
    values.push_back(TypeValue(d, call->ty));
  }
  return NamespaceDoc("tirx")->Attr("tvm_storage_sync")->Call(args, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tvm_storage_sync")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&StorageSyncDocTranslate>());
}

ffi::Optional<ExprDoc> CallExternDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (!call->op.same_as(tirx::builtin::call_extern()) || !CanTranslateExplicitResultCall(call) ||
      call->args.empty()) {
    return RawCall(d, call);
  }
  const auto* name = call->args[0].as<StringImmNode>();
  if (!name) return RawCall(d, call);
  ExprDoc name_doc = LiteralDoc::Str(name->value, std::nullopt);
  d->RecordOrigin(name_doc, call->args[0]);
  ffi::Array<ExprDoc> args = {TypeValue(d, call->ty), name_doc};
  for (size_t i = 1; i < call->args.size(); ++i) {
    args.push_back(MaterializeCallArgument(d, call->args[i], d->Translate(call->args[i]).value()));
  }
  return NamespaceDoc("tirx")->Attr("call_extern")->Call(args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.call_extern")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&CallExternDocTranslate>());
}

ffi::Optional<ExprDoc> CUDAFuncCallDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  static const Op cuda_func_call = Op::Get("tirx.cuda.func_call");
  if (!call->op.same_as(cuda_func_call) || !CanTranslateExplicitResultCall(call) ||
      call->args.size() < 2) {
    return RawCall(d, call);
  }
  const auto* name = call->args[0].as<StringImmNode>();
  const auto* source = call->args.back().as<StringImmNode>();
  if (!name || !source) return RawCall(d, call);
  ExprDoc name_doc = LiteralDoc::Str(name->value, std::nullopt);
  ExprDoc source_doc = LiteralDoc::Str(source->value, std::nullopt);
  d->RecordOrigin(name_doc, call->args[0]);
  d->RecordOrigin(source_doc, call->args.back());
  ffi::Array<ExprDoc> args = {name_doc};
  for (size_t i = 1; i + 1 < call->args.size(); ++i) {
    args.push_back(MaterializeCallArgument(d, call->args[i], d->Translate(call->args[i]).value()));
  }
  ffi::Array<ffi::String> keys = {"source_code"};
  ffi::Array<ExprDoc> values = {source_doc};
  if (!ffi::StructuralEqual()(call->ty, PrimType::Void())) {
    keys.push_back("return_type");
    values.push_back(TypeValue(d, call->ty));
  }
  return NamespaceDoc("tirx")->Attr("cuda")->Attr("func_call")->Call(args, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.func_call")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&CUDAFuncCallDocTranslate>());
}

template <bool block_scaled>
ffi::Optional<ExprDoc> CUDAInstructionDescriptorDocTranslate(DocTranslatorObj* d,
                                                             ffi::AnyView input,
                                                             const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (!CanTranslateExplicitResultCall(call) || call->args.size() != (block_scaled ? 17 : 14) ||
      !ffi::StructuralEqual()(call->ty, PrimType::Void())) {
    return RawCall(d, call);
  }
  Op op = call->op.as_or_throw<Op>();
  if (op->args_info.size() != call->args.size()) return RawCall(d, call);
  constexpr size_t optional_begin = block_scaled ? 13 : 9;
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  ExprDoc descriptor = d->Translate(call->args[0]).value();
  for (size_t i = 1; i < call->args.size(); ++i) {
    const Expr& arg = call->args[i];
    if (i >= optional_begin) {
      const auto* value = arg.as<IntImmNode>();
      PrimType default_type = i == optional_begin ? PrimType::Int(32) : PrimType::Bool();
      int64_t default_value = i == optional_begin ? 1 : 0;
      if (value && value->value == default_value &&
          ffi::StructuralEqual()(value->ty, default_type)) {
        continue;
      }
    }
    const auto* string = arg.as<StringImmNode>();
    ExprDoc value = string ? LiteralDoc::Str(string->value, std::nullopt)
                           : MaterializeCallArgument(d, arg, d->Translate(arg).value());
    d->RecordOrigin(value, arg);
    keys.push_back(op->args_info[i]->name);
    values.push_back(value);
  }
  return NamespaceDoc("tirx")
      ->Attr("cuda")
      ->Attr("tcgen05")
      ->Attr(block_scaled ? "encode_instr_descriptor_block_scaled" : "encode_instr_descriptor")
      ->Call({descriptor}, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor")
      .set_attr<FDocTranslate>(
          kOpCallDocTranslate,
          FDocTranslate::FromNative<&CUDAInstructionDescriptorDocTranslate<false>>());
  OpDef("tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled")
      .set_attr<FDocTranslate>(
          kOpCallDocTranslate,
          FDocTranslate::FromNative<&CUDAInstructionDescriptorDocTranslate<true>>());
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
  ffi::Array<ExprDoc> args = {TypeValue(d, call->ty), name_doc};
  for (size_t i = 1; i < call->args.size(); ++i) {
    args.push_back(MaterializeCallArgument(d, call->args[i], d->Translate(call->args[i]).value()));
  }
  return NamespaceDoc("tirx")
      ->Attr(call->op.same_as(tirx::builtin::call_llvm_intrin()) ? "call_llvm_intrin"
                                                                 : "call_llvm_pure_intrin")
      ->Call(args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"tirx.call_llvm_intrin", "tirx.call_llvm_pure_intrin"}) {
    OpDef(name).set_attr<FDocTranslate>(kOpCallDocTranslate,
                                        FDocTranslate::FromNative<&LLVMIntrinsicDocTranslate>());
  }
}

ffi::Optional<ExprDoc> GetActiveLaneMaskDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                     const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (!call->op.same_as(tirx::builtin::get_active_lane_mask()) || call->attrs.defined() ||
      !call->ty_args.empty() || call->args.size() != 2) {
    return RawCall(d, call);
  }
  auto result = call->ty.as<PrimType>();
  if (!result ||
      !(result.value().MatchesCode(DLDataTypeCode::kDLBool) ||
        result.value().MatchesElementType(DLDataTypeCode::kDLUInt, 1)) ||
      !(result.value().IsScalableVector() || result.value().IsFixedLengthVector())) {
    return RawCall(d, call);
  }
  ffi::Array<ExprDoc> args = {TypeValue(d, call->ty)};
  for (const Expr& arg : call->args) {
    auto type = arg->ty.as<PrimType>();
    if (!type || !type.value().IsScalar() ||
        !type.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
      return RawCall(d, call);
    }
    args.push_back(MaterializeCallArgument(d, arg, d->Translate(arg).value()));
  }
  return NamespaceDoc("tirx")->Attr("get_active_lane_mask")->Call(args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.get_active_lane_mask")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&GetActiveLaneMaskDocTranslate>());
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

ffi::Optional<ExprDoc> TIRCallPrefixDocTranslate(DocTranslatorObj* d, const CallNode* call) {
  bool is_ptx = false;
  if (auto op = call->op.as<Op>()) {
    const std::string name = op.value()->name;
    is_ptx = name.rfind("tirx.ptx.", 0) == 0;
    bool descriptor = name == "tirx.cuda.tcgen05_encode_matrix_descriptor" ||
                      name == "tirx.cuda.tcgen05_encode_instr_descriptor" ||
                      name == "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled" ||
                      name == "tirx.cuda.wgmma_encode_matrix_descriptor" ||
                      name == "tirx.cuda.wgmma_noop_barrier";
    // AddrArg is only an intermediate object consumed by a PTX instruction.
    // A standalone address Call must remain an expression after reparsing.
    if (IsPTXAddressCall(call) ||
        ((is_ptx || descriptor) && !ffi::StructuralEqual()(call->ty, PrimType::Void()))) {
      return RawCall(d, call);
    }
  }
  if (auto op = call->op.as<Op>(); op && op.value()->name == "tirx.isnan") {
    auto input_type = call->args.size() == 1 ? call->args[0]->ty.as<PrimType>() : std::nullopt;
    // The named helper folds constants and widens float16 operands, and its
    // fixed-lane result constructor cannot represent scalable vectors.
    if (!input_type || input_type.value().IsScalableVector() ||
        !input_type.value().MatchesCode(DLDataTypeCode::kDLFloat) ||
        (input_type.value().bits() != 32 && input_type.value().bits() != 64) ||
        call->args[0].as<FloatImmNode>()) {
      return RawCall(d, call);
    }
  }
  if (call->op.same_as(tirx::builtin::buffer_data()) && !call->attrs.defined() &&
      call->ty_args.empty()) {
    TVM_FFI_CHECK(call->args.size() == 1, ValueError) << "buffer_data expects one buffer";
    return d->Translate(call->args[0]).value()->Attr("data");
  }
  return std::nullopt;
}

ffi::Optional<ExprDoc> FFIKernelDocTranslate(DocTranslatorObj* d, const CallNode* call,
                                             const Type& result_type,
                                             const ffi::Array<ExprDoc>& args) {
  if (call->op.same_as(tirx::builtin::call_ffi_kernel())) {
    const auto* attrs = call->attrs.as<tirx::CallFFIKernelAttr>();
    if (!attrs || !call->ty_args.empty()) return RawCall(d, call, args);
    ffi::Array<ExprDoc> launch_params;
    for (const ffi::String& param : attrs->launch_params) {
      launch_params.push_back(LiteralDoc::Str(param, std::nullopt));
    }
    return NamespaceDoc("tirx")
        ->Attr("call_ffi_kernel")
        ->Call(args, {"launch_params", "ret_ty"},
               {ListDoc(launch_params), TypeValue(d, result_type)});
  }
  return std::nullopt;
}

ffi::Optional<ExprDoc> TIRCallDocTranslate(DocTranslatorObj* d, const CallNode* call,
                                           const Type& result_type,
                                           const ffi::Array<ExprDoc>& args) {
  ffi::Optional<Op> op = call->op.as<Op>();
  bool is_ptx = op && op.value()->name.find("tirx.ptx.") == 0;
  static const auto& categories = Op::GetAttrMap<tirx::TIRxOpCategory>("TIRxOpCategory");
  // Canonical TIRx entry points with an eligible inference hook may require
  // an explicit dtype position. Use the inferred type for that argument.
  if (op.has_value() && categories.count(op.value()) && !call->attrs.defined() &&
      call->ty_args.empty()) {
    static const OpAttrMap<tirx::TScriptPrinterName>& names =
        Op::GetAttrMap<tirx::TScriptPrinterName>("TScriptPrinterName");
    static const OpAttrMap<tirx::TScriptDtypePrintLocation>& dtype_locations =
        Op::GetAttrMap<tirx::TScriptDtypePrintLocation>("TScriptDtypePrintLocation");
    // These low-level constructors have a parser signature that differs from
    // their stored Call argument list. Keep the lossless I.Call form until a
    // dedicated translation covers each signature.
    bool incompatible_signature = (op.value()->name == "tirx.cuda.ldg" && call->args.size() != 2) ||
                                  op.value()->name == "tirx.cuda.wait_until" ||
                                  op.value()->name == "tirx.cuda.mov_sreg";
    // Meaningful wrappers choose their result from operands, independently of
    // customizable inference hooks. Only use them when that choice is lossless.
    const std::string op_name = op.value()->name;
    int result_operand = -1;
    if (op_name == "tirx.cuda.atomic_add" || op_name == "tirx.cuda.atomic_cas" ||
        op_name == "tirx.cuda.__shfl_sync" || op_name == "tirx.cuda.__shfl_up_sync" ||
        op_name == "tirx.cuda.__shfl_down_sync" || op_name == "tirx.cuda.__shfl_xor_sync") {
      result_operand = 1;
    } else if (op_name == "tirx.cuda.warp_reduce" || op_name == "tirx.cuda.cta_reduce") {
      result_operand = 0;
    }
    if (result_operand >= 0 &&
        (call->args.size() <= static_cast<size_t>(result_operand) ||
         !ffi::StructuralEqual()(call->ty, call->args[result_operand]->ty))) {
      incompatible_signature = true;
    }
    if (op_name == "tirx.cuda.__activemask" &&
        !ffi::StructuralEqual()(call->ty, PrimType::UInt(32))) {
      incompatible_signature = true;
    }
    if (op_name == "tirx.cuda.ldg" && call->args.size() == 2) {
      auto dtype = call->args[1].as<StringImmNode>();
      auto result = call->ty.as<PrimType>();
      if (!dtype || !result || ffi::DLDataTypeToString(result.value()->dtype) != dtype->value) {
        incompatible_signature = true;
      }
    }
    // Published CUDA callables use canonical names. Late registrations without
    // a published callable retain lossless reconstruction.
    bool canonical_cuda = op_name.find("tirx.cuda.") == 0;
    if (canonical_cuda) {
      // Generated APIs validate construction; preserve provisional or invalid
      // Calls through the explicit unchecked reconstruction surface instead.
      try {
        op.value().Validate(call);
      } catch (const ffi::Error&) {
        return RawCall(d, call);
      }
      if (op_name.find("tirx.cuda.__shfl") == 0 && call->args.size() > 1 &&
          call->args[1].as<VarNode>() && call->args[1]->ty.as<tirx::TensorTypeNode>()) {
        return RawCall(d, call);
      }
      if ((result_operand >= 0 || op_name == "tirx.cuda.ldg") &&
          std::any_of(call->args.begin(), call->args.end(),
                      [](const Expr& arg) { return arg.as<TensorRegionNode>() != nullptr; })) {
        return RawCall(d, call);
      }
    }
    if (names.count(op.value()) && !incompatible_signature) {
      std::string name = names[op.value()];
      if (!name.empty()) {
        ExprDoc callee = NamedCallCallee(name);
        ffi::Array<ExprDoc> named_args;
        auto dtype_location = static_cast<tirx::ScriptDtypePrintLocation>(dtype_locations.get(
            op.value(), static_cast<int64_t>(tirx::ScriptDtypePrintLocation::kNone)));
        if (dtype_location == tirx::ScriptDtypePrintLocation::kFirst)
          named_args.push_back(TypeValue(d, result_type));
        for (size_t i = 0; i < call->args.size(); ++i) {
          if (auto string = call->args[i].as<StringImmNode>()) {
            named_args.push_back(LiteralDoc::Str(string->value, std::nullopt));
          } else {
            ExprDoc argument = args[i];
            // Canonical CUDA APIs preserve structured IR operands.
            if (op.value()->name.find("tirx.cuda.") == 0) {
              argument = MaterializeCallArgument(d, call->args[i], argument);
            }
            if (const auto* address = call->args[i].as<CallNode>();
                is_ptx && address && IsPTXAddressCall(address)) {
              // Use the address constructor only when it preserves its operands.
              auto translated = ConsumedPTXAddressDocTranslate(d, address);
              if (!translated) return RawCall(d, call);
              argument = translated.value();
            }
            bool reads_operand_type =
                is_ptx ||
                (i == 0 && (op.value()->name == "tirx.cuda.warp_reduce" ||
                            op.value()->name == "tirx.cuda.cta_reduce" ||
                            op.value()->name == "tirx.selector" ||
                            op.value()->name == "tirx.webgpu.subgroup_shuffle" ||
                            op.value()->name == "tirx.webgpu.subgroup_shuffle_up" ||
                            op.value()->name == "tirx.webgpu.subgroup_shuffle_down" ||
                            op.value()->name == "tirx.metal.simd_shuffle" ||
                            op.value()->name == "tirx.metal.simd_shuffle_up" ||
                            op.value()->name == "tirx.metal.simd_shuffle_down")) ||
                (i == 1 && (op.value()->name == "tirx.cuda.__shfl_sync" ||
                            op.value()->name == "tirx.cuda.__shfl_up_sync" ||
                            op.value()->name == "tirx.cuda.__shfl_down_sync" ||
                            op.value()->name == "tirx.cuda.__shfl_xor_sync" ||
                            op.value()->name == "tirx.tvm_warp_shuffle" ||
                            op.value()->name == "tirx.tvm_warp_shuffle_up" ||
                            op.value()->name == "tirx.tvm_warp_shuffle_down" ||
                            op.value()->name == "tirx.tvm_warp_shuffle_xor"));
            // These constructors inspect the operand's type before converting
            // Python values, so an int32 literal must remain an IR expression.
            if (reads_operand_type && call->args[i].as<IntImmNode>() &&
                ffi::StructuralEqual()(call->args[i]->ty, PrimType::Int(32))) {
              argument = NamespaceDoc("tirx")->Attr("int32")->Call({argument});
              d->RecordOrigin(argument, call->args[i]);
            }
            named_args.push_back(argument);
          }
        }
        if (dtype_location == tirx::ScriptDtypePrintLocation::kLast)
          named_args.push_back(TypeValue(d, result_type));
        return callee->Call(named_args);
      }
    }
  }

  return std::nullopt;
}

namespace {}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
