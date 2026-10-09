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

/*!
 * \file tirx/op/memory.cc
 * \brief TIRx memory operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/type.h>

namespace tvm {
namespace tirx {

template <size_t DataIndex, int ElementIndex>
Type InferTypePointerOffset(const CallNode* call) {
  TVM_FFI_CHECK_GT(call->args.size(), DataIndex, ValueError);
  Type element_type = PrimType::Void();
  if constexpr (ElementIndex >= 0) {
    TVM_FFI_CHECK_GT(call->args.size(), static_cast<size_t>(ElementIndex), ValueError);
    element_type = call->args[ElementIndex]->ty;
    if (element_type.as<MissingType>()) return Type::Missing();
  }
  auto pointer = call->args[DataIndex]->ty.as<PointerType>();
  return PointerType(element_type, pointer ? pointer.value()->storage_scope : "global");
}

Type InferTypeAccessPointer(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->ty_args.size(), 1U, ValueError);
  TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
  auto pointer = call->args[0]->ty.as<PointerType>();
  return PointerType(call->ty_args[0], pointer ? pointer.value()->storage_scope : "global");
}

Type InferTypeAddressOf(const CallNode* call) {
  TVM_FFI_CHECK_GE(call->args.size(), 1U, ValueError) << "Address type requires an object";
  if (const auto* load = call->args[0].as<TensorLoadNode>()) {
    return load->source.as_or_throw<TensorVar>().DataPointerType();
  }
  Var variable = call->args[0].as_or_throw<Var>();
  if (auto pointer = variable->ty.as<PointerType>();
      pointer && pointer.value()->element_type.as<TensorMapType>()) {
    return PrimType::UInt(64);
  }
  return PointerType(variable->ty.as_or_throw<PrimType>());
}

Type InferTypeMaskedLoad(const CallNode* call) {
  TVM_FFI_CHECK_GE(call->args.size(), 2U, ValueError)
      << "Masked load type requires a buffer and index operands";
  TensorVar buffer = call->args[0].as_or_throw<TensorVar>();
  ffi::Array<PrimExpr> indices;
  for (size_t i = 1; i + 1 < call->args.size(); ++i) {
    indices.push_back(call->args[i].as_or_throw<PrimExpr>());
  }
  // Ordinary load typing computes vector elements and scalable index lanes.
  return MakeTensorLoad(buffer, indices).ty();
}

ffi::Expected<Type> InferTypeTensorDataPtr(const CallNode* call) noexcept try {
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError)
      << "tirx.tensor_data_ptr expects one TensorVar argument";
  Type inferred = call->args[0].as_or_throw<TensorVar>().DataPointerType();
  if (call->ty.same_as(inferred) || ffi::StructuralEqual()(call->ty, inferred)) return call->ty;
  return inferred;
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

// Essential buffer properties follow operands; the supplied result type retains layout metadata.
template <int shape_index>
ffi::Expected<Type> InferTypeBuffer(const CallNode* call) noexcept try {
  if constexpr (shape_index == 0) {
    TVM_FFI_CHECK(call->args.size() == 3 || call->args.size() == 4, ValueError);
    if (call->args.size() == 4) {
      auto placement = call->args[3].as_or_throw<tvm::Tuple>();
      for (const Expr& address : placement->fields) address.as_or_throw<PrimExpr>();
    }
  } else {
    TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
  }
  tvm::Tuple shape = call->args[shape_index].as_or_throw<tvm::Tuple>();
  DLDataType dtype = call->args[shape_index + 1].as_or_throw<DataTypeImm>()->value;
  ffi::String scope = call->args[shape_index + 2].as_or_throw<StringImm>()->value;
  auto original = call->ty.as_or_throw<TensorType>();
  if (ffi::StructuralEqual()(shape->fields, original->shape) && dtype == original->dtype->dtype &&
      scope == original->storage_scope) {
    return original;
  }
  auto inferred = ffi::make_object<TensorTypeNode>(*original.get());
  inferred->shape =
      shape->fields.Map([](const Expr& extent) { return extent.as_or_throw<PrimExpr>(); });
  inferred->dtype = PrimType(dtype);
  inferred->storage_scope = scope;
  return TensorType(std::move(inferred));
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

ffi::Expected<void> ValidateAllocTensor(const CallNode* call) noexcept {
  auto inferred = InferTypeBuffer<0>(call);
  if (!inferred.has_value()) return ffi::Unexpected(inferred.error());
  return {};
}

ffi::Expected<void> ValidateDeclTensor(const CallNode* call) noexcept try {
  TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
  call->ty.as_or_throw<TensorType>();
  return {};
} catch (const ffi::Error& error) {
  return ffi::Unexpected(error);
} catch (const std::exception& error) {
  return ffi::Unexpected(ffi::Error("InternalError", error.what(), ""));
}

const Op& reinterpret_op() {
  static const Op op = Op::Get("tirx.reinterpret");
  return op;
}

Expr reinterpret(Type target_ty, Expr value, Span span) {
  if (value.as<StringImmNode>()) {
    TVM_FFI_CHECK(target_ty.as<PointerTypeNode>(), TypeError)
        << "String reinterpret requires a pointer target, but got " << target_ty;
    return Call(std::move(target_ty), tirx::reinterpret_op(), {std::move(value)}, {}, {},
                std::move(span));
  }
  if (auto target_dtype = target_ty.as<PrimType>()) {
    if (auto prim_value = value.as<PrimExpr>()) {
      PrimType target_prim = target_dtype.value();
      PrimType value_dtype = prim_value.value().ty();
      if (value_dtype == target_prim) return value;
      if (!target_prim.IsScalableVector() && !value_dtype.IsScalableVector()) {
        int value_bits = value_dtype.bits() * value_dtype.lanes();
        int target_bits = target_prim.bits() * target_prim.lanes();
        TVM_FFI_ICHECK(value_bits == target_bits ||
                       ((value_dtype.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn) ||
                         target_prim.MatchesCode(DLDataTypeCode::kDLFloat4_e2m1fn)) &&
                        value_dtype.StorageBytes() == target_prim.StorageBytes()))
            << "Reinterpret requires size match " << target_prim << " vs " << value_dtype;
      }
    } else {
      TVM_FFI_CHECK(value->ty.as<PointerTypeNode>(), TypeError)
          << "Reinterpret source must be PrimType or PointerType, but got " << value->ty;
      TVM_FFI_CHECK(
          target_dtype.value().IsScalar() && target_dtype.value().bits() == 64 &&
              target_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt),
          TypeError)
          << "Pointer reinterpret requires a scalar 64-bit integer target, but got "
          << target_dtype.value();
    }
  } else {
    TVM_FFI_CHECK(target_ty.as<PointerTypeNode>(), TypeError)
        << "Reinterpret target must be PrimType or PointerType, but got " << target_ty;
    if (auto source_dtype = value->ty.as<PrimType>()) {
      TVM_FFI_CHECK(
          source_dtype.value().IsScalar() && source_dtype.value().bits() == 64 &&
              source_dtype.value().MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt),
          TypeError)
          << "Pointer reinterpret requires a scalar 64-bit integer source, but got "
          << source_dtype.value();
    } else {
      TVM_FFI_CHECK(value->ty.as<PointerTypeNode>(), TypeError)
          << "Reinterpret source must be PrimType or PointerType, but got " << value->ty;
    }
  }
  return Call(std::move(target_ty), tirx::reinterpret_op(), {std::move(value)}, {}, {},
              std::move(span));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.reinterpret")
      .signature(sig::arg("x", "The input value."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.reinterpret"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& address_of_op() {
  static const Op op = Op::Get("tirx.address_of");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.address_of")
      .signature(sig::arg("obj", "The referenced object."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeAddressOf>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.address_of"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& isnullptr_op() {
  static const Op op = Op::Get("tirx.isnullptr");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.isnullptr")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Bool())
      .signature(sig::arg("x", "The input value."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.isnullptr"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& prefetch_op() {
  static const Op op = Op::Get("tirx.prefetch");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.prefetch")
      .signature(sig::arg("ptr", "The pointer."), sig::arg<IntExpr>("rw", "The read/write mode."),
                 sig::arg<IntExpr>("locality", "The locality hint."),
                 sig::arg<IntExpr>("cache_type", "The cache policy."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& access_ptr_op() {
  static const Op op = Op::Get("tirx.access_ptr");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.access_ptr")
      .signature(sig::ty_arg<PrimType>("access_dtype", "The accessed element type."),
                 sig::arg("data", "The input data."), sig::arg<IntExpr>("offset", "The offset."),
                 sig::arg<IntExpr>("extent", "The extent."),
                 sig::arg<IntExpr>("rw_mask", "The read/write mask."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.access_ptr"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeAccessPointer>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));
}

const Op& ptr_byte_offset_op() {
  static const Op op = Op::Get("tirx.ptr_byte_offset");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.ptr_byte_offset")
      .signature(sig::arg("data", "Base pointer."),
                 sig::arg<IntExpr>("byte_offset", "Offset in bytes."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.ptr_byte_offset"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& static_handle_op() {
  static const Op op = Op::Get("tirx.static_handle");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.static_handle")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kSpecialCallArg));
}

const Op& handle_add_byte_offset_op() {
  static const Op op = Op::Get("tirx.handle_add_byte_offset");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.handle_add_byte_offset")
      .signature(sig::arg("handle", "The handle."), sig::arg<IntExpr>("offset", "The offset."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                    ffi::String("tirx.handle_add_byte_offset"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypePointerOffset<0, -1>>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& atomic_add_op() {
  static const Op op = Op::Get("tirx.atomic_add");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.atomic_add")
      .signature(sig::arg("ptr", "The pointer."), sig::arg("value", "The value to use."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& assume_aligned_op() {
  static const Op op = Op::Get("tirx.assume_aligned");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.assume_aligned")
      .signature(sig::arg<TensorVar>("tensor", "The tensor whose base address is aligned."),
                 sig::arg<IntImm>("alignment_bytes", "The constant byte alignment."))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.assume_aligned"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kEmbedInfo));
}

const Op& undef_op() {
  static const Op op = Op::Get("tirx.undef");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.undef")
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Int(32))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.undef"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));
}

const Op& masked_load_op() {
  static const Op op = Op::Get("tirx.masked_load");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.masked_load")
      .signature(sig::arg<TensorVar>("buffer", "The buffer."),
                 sig::arg<PrimExpr>("index", "The index."), sig::var_args<PrimExpr>("args"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeMaskedLoad>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.masked_load"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kReadState));
}

const Op& masked_store_op() {
  static const Op op = Op::Get("tirx.masked_store");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.masked_store")
      .signature(sig::arg<TensorVar>("buffer", "The buffer."),
                 sig::arg<PrimExpr>("value", "The value to use."),
                 sig::arg<PrimExpr>("index", "The index."), sig::var_args<PrimExpr>("args"))
      .set_attr<TFixedReturnType>("TFixedReturnType", PrimType::Void())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.masked_store"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kUpdateState));
}

const Op& alloc_tensor_op() {
  static const Op op = Op::Get("tirx.alloc_tensor");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.alloc_tensor")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeBuffer<0>>())
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateAllocTensor>())
      .add_arg("shape", "The tuple of buffer extents.")
      .add_arg("dtype", "The buffer data type.")
      .add_arg("scope", "The storage scope.")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

const Op& decl_tensor_op() {
  static const Op op = Op::Get("tirx.decl_tensor");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.decl_tensor")
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeBuffer<1>>())
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallNode*)>::FromNative<&ValidateDeclTensor>())
      .add_arg("data", "The existing data pointer.")
      .add_arg("shape", "The tuple of buffer extents.")
      .add_arg("dtype", "The buffer data type.")
      .add_arg("scope", "The storage scope.")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& tensor_data_ptr_op() {
  static const Op op = Op::Get("tirx.tensor_data_ptr");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.tensor_data_ptr")
      .signature(sig::arg("tensor", "The tensor variable."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeTensorDataPtr>())
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tirx.reinterpret", [](Type dtype, Expr value, Span span) {
    return reinterpret(dtype, value, span);
  });
}

}  // namespace tirx
}  // namespace tvm
