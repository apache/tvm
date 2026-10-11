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

#include <tvm/relax/distributed/type.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/tensor.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/printer.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "../../../script/printer/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> GenericConstDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object*) {
  const auto* constant =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const GenericConstNode>(input);
  if (auto dtype = constant->value.as<DLDataType>()) {
    return LiteralDoc::DataType(dtype.value(), std::nullopt);
  }
  auto tensor = constant->value.as<runtime::Tensor>();
  if (!tensor.has_value()) return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  const runtime::Tensor& data = tensor.value();
  DLDataType dtype = data.DataType();
  auto print_const = [&](ExprDoc value) -> ExprDoc {
    if (constant->ty.as<relax::distributed::DTensorTypeNode>()) {
      return NamespaceDoc("relax")->Attr("dist")->Attr("const")->Call(
          {value, d->Translate(constant->ty).value()});
    }
    return NamespaceDoc("relax")->Attr("const")->Call(
        {value, LiteralDoc::DataType(dtype, std::nullopt)});
  };
  if (data->ndim != 0) return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  if (data->device.device_type != kDLCPU)
    return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  if (dtype.lanes != 1 || dtype.bits == 0 || dtype.bits > 64 || dtype.bits % 8 != 0) {
    return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  }
  alignas(double) uint8_t scalar_bytes[sizeof(double)]{};
  data.CopyToBytes(scalar_bytes, dtype.bits / 8);
  auto read_scalar = [&](auto value) {
    std::memcpy(&value, scalar_bytes, sizeof(value));
    return value;
  };
  ExprDoc scalar(ffi::UnsafeInit{});
  if (dtype == (DLDataType{kDLInt, 8, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int8_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLInt, 16, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int16_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLInt, 32, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int32_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLInt, 64, 1})) {
    scalar = LiteralDoc::Int(read_scalar(int64_t{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLFloat, 16, 1})) {
    uint16_t bits = read_scalar(uint16_t{});
    uint16_t exponent = (bits >> 10) & 31;
    uint16_t fraction = bits & 1023;
    double value = exponent == 31
                       ? (fraction ? std::numeric_limits<double>::quiet_NaN()
                                   : std::numeric_limits<double>::infinity())
                       : (exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24)
                                        : std::ldexp(static_cast<double>(fraction | 1024),
                                                     static_cast<int>(exponent) - 25));
    if (bits & 0x8000) value = -value;
    scalar = LiteralDoc::Float(value, std::nullopt);
  } else if (dtype == (DLDataType{kDLFloat, 32, 1})) {
    scalar = LiteralDoc::Float(read_scalar(float{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLFloat, 64, 1})) {
    scalar = LiteralDoc::Float(read_scalar(double{}), std::nullopt);
  } else if (dtype == (DLDataType{kDLBool, 8, 1})) {
    scalar = LiteralDoc::Boolean(read_scalar(uint8_t{}), std::nullopt);
  } else {
    return AddMetadata(d, ffi::GetRef<GenericConst>(constant));
  }
  return print_const(scalar);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<GenericConstNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&GenericConstDocTranslate>());
}

ffi::Optional<ExprDoc> ShapeExprDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* shape =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const relax::ShapeExprNode>(input);
  ffi::Array<ExprDoc> dimensions;
  for (const PrimExpr& dim : shape->values) dimensions.push_back(RelaxShapeDim(d, dim));
  return NamespaceDoc("relax")->Attr("shape")->Call({ListDoc(dimensions)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::ShapeExprNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&ShapeExprDocTranslate>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<relax::DataflowVarNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&VarDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script

TVM_FFI_STATIC_INIT_BLOCK() {
  using script::printer::details::RegisterScriptRepr;
  ffi::reflection::GlobalDef().def("script.printer.ReprPrintRelax",
                                   [](const ffi::ObjectRef& obj, const PrinterConfig& config) {
                                     return tvm::Script(obj, config);
                                   });
  RegisterScriptRepr<relax::BindingBlockNode>();
  RegisterScriptRepr<relax::DataflowBlockNode>();
  RegisterScriptRepr<relax::DataflowVarNode>();
  RegisterScriptRepr<relax::ExternFuncNode>();
  RegisterScriptRepr<relax::FuncTypeNode>();
  RegisterScriptRepr<relax::FunctionNode>();
  RegisterScriptRepr<relax::IfExprNode>();
  RegisterScriptRepr<relax::MatchCastNode>();
  RegisterScriptRepr<relax::PackedFuncTypeNode>();
  RegisterScriptRepr<relax::SeqExprNode>();
  RegisterScriptRepr<relax::ShapeExprNode>();
  RegisterScriptRepr<relax::ShapeTypeNode>();
  RegisterScriptRepr<relax::TensorTypeNode>();
  RegisterScriptRepr<relax::VarBindingNode>();
  RegisterScriptRepr<relax::distributed::DTensorTypeNode>();
  RegisterScriptRepr<relax::distributed::DeviceMeshNode>();
  RegisterScriptRepr<relax::distributed::PlacementNode>();
}

}  // namespace tvm
