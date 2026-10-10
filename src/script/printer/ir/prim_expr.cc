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
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>

#include <cstring>
#include <optional>

#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

ffi::Optional<ExprDoc> LambdaExprDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* lambda =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const LambdaExprNode>(input);
  VarScope scope(d);
  ffi::Array<IdDoc> args;
  ffi::Array<ExprDoc> types;
  for (const Var& var : lambda->vars) {
    args.push_back(VarDoc(d, var));
    types.push_back(TypeValue(d, var->ty, false));
  }
  ExprDoc body = d->Translate(lambda->body).value();
  return NamespaceDoc("ir")->Attr("Lambda")->Call({ListDoc(types), LambdaDoc(args, body)});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<LambdaExprNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&LambdaExprDocTranslate>());
}

ffi::Array<Doc> TensorIndices(DocTranslatorObj* d, const ffi::Array<PrimExpr>& indices,
                              bool store) {
  ffi::Array<Doc> docs;
  for (const PrimExpr& index : indices) {
    if (const auto* ramp = index.as<prim::RampNode>(); store && ramp) {
      if (const auto* stride = ramp->stride.as<IntImmNode>()) {
        ffi::Optional<ExprDoc> step = std::nullopt;
        if (stride->value != 1) step = d->Translate(ramp->stride).value();
        SliceDoc slice(d->Translate(ramp->base).value(),
                       d->Translate(ramp->base + ramp->lanes * ramp->stride).value(), step);
        d->RecordOrigin(slice, index);
        docs.push_back(slice);
        continue;
      }
    }
    docs.push_back(d->Translate(index).value());
  }
  return docs;
}

ExprDoc TensorRegionValue(DocTranslatorObj* d, const TensorRegionNode* region,
                          bool require_region) {
  ffi::Array<Doc> slices;
  for (const Range& range : region->region) {
    ExprDoc start = d->Translate(range->min).value();
    if (const auto* extent = range->extent.as<IntImmNode>();
        !require_region && extent && extent->value == 1) {
      slices.push_back(start);
    } else {
      ExprDoc end = [&]() -> ExprDoc {
        auto lower = range->min.as<IntImmNode>();
        auto length = range->extent.as<IntImmNode>();
        if (lower && length) {
          auto last = (lower->value + length->value).as<int64_t>();
          if (last) {
            return LiteralDoc::Int(*last, std::nullopt);
          }
        }
        return OperationDoc(OperationDocNode::Kind::kAdd,
                            {start, d->Translate(range->extent).value()});
      }();
      slices.push_back(SliceDoc(start, end, std::nullopt));
    }
  }
  return d->Translate(region->source).value()[slices];
}

namespace {

ffi::Optional<ExprDoc> TensorRegionDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object*) {
  const auto* region =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorRegionNode>(input);
  return TensorRegionValue(d, region, false);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TensorRegionNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TensorRegionDocTranslate>());
}

ffi::Optional<ExprDoc> TensorLoadDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object* destination) {
  const auto* load =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorLoadNode>(input);
  static ffi::reflection::TypeAttrColumn column(kTensorLoadDocTranslate);
  ffi::AnyView hook = column[load->source->ty->type_index()];
  ffi::Any value = ffi::GetRef<TensorLoad>(load);
  if (hook.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr) {
    return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<ffi::Optional<ExprDoc>>(
               reinterpret_cast<decltype(DocTranslatorVTable::translate)>(hook.cast<void*>())(
                   d, value, destination))
        .value();
  }
  if (hook.type_index() == ffi::TypeIndex::kTVMFFIFunction) {
    ffi::Any binder = destination ? ffi::Any(ffi::GetRef<ffi::ObjectRef>(destination)) : nullptr;
    return hook.cast<ffi::Function>()
        .CallExpected<ffi::Optional<ExprDoc>>(d, value, binder)
        .value();
  }
  TVM_FFI_CHECK(hook.type_index() == ffi::TypeIndex::kTVMFFINone, TypeError)
      << "TensorLoad type hook must be a native pointer or ffi.Function";
  return IndexDoc(d->Translate(load->source).value(), TensorIndices(d, load->indices));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TensorLoadNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TensorLoadDocTranslate>());
}

ffi::Optional<ExprDoc> TupleDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                         const ffi::Object*) {
  const auto* tuple =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleNode>(input);
  ffi::Array<ExprDoc> fields;
  for (const Expr& field : tuple->fields) fields.push_back(d->Translate(field).value());
  return TupleDoc(fields);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleNode>().attr(kDocTranslate,
                                                 FDocTranslate::FromNative<&TupleDocTranslate>());
}

ffi::Optional<ExprDoc> TupleGetItemDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                const ffi::Object*) {
  const auto* item =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TupleGetItemNode>(input);
  return d->Translate(item->tuple).value()[{LiteralDoc::Int(item->index, std::nullopt)}];
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TupleGetItemNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&TupleGetItemDocTranslate>());
}

template <typename T, OperationDocNode::Kind kind, PrimExpr (*operation)(PrimExpr, PrimExpr, Span)>
ffi::Optional<ExprDoc> BinaryOpDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                            const ffi::Object*) {
  const auto* node = ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const T>(input);
  ExprDoc a = d->Translate(node->a).value();
  ExprDoc b = d->Translate(node->b).value();
  // Replay operator overloads before choosing sugar. Even one constant can
  // simplify an explicit node, while two plain Python literals fold earlier.
  if (!(a.as<LiteralDocNode>() && b.as<LiteralDocNode>())) {
    try {
      PrimExpr replay = operation(node->a, node->b, Span());
      if (const auto* result = replay.template as<T>();
          result && result->a.same_as(node->a) && result->b.same_as(node->b)) {
        return OperationDoc(kind, {a, b});
      }
    } catch (const ffi::Error&) {
      // A constructor can preserve operands rejected by a simplifying helper.
    }
  }
  return NamespaceDoc("tirx")->Attr(std::strrchr(T::_type_key, '.') + 1)->Call({a, b});
}

template <typename T, OperationDocNode::Kind kind, PrimExpr (*operation)(PrimExpr, Span)>
ExprDoc UnaryOpDocTranslate(DocTranslatorObj* d, const T* node) {
  ExprDoc value = d->Translate(node->a).value();
  if (!value.as<LiteralDocNode>()) {
    try {
      PrimExpr replay = operation(node->a, Span());
      if (const auto* result = replay.template as<T>(); result && result->a.same_as(node->a)) {
        return OperationDoc(kind, {value});
      }
    } catch (const ffi::Error&) {
      // Preserve the explicit node even when the overloaded spelling rejects it.
    }
  }
  return NamespaceDoc("tirx")->Attr(std::strrchr(T::_type_key, '.') + 1)->Call({value});
}

template <typename T, PrimExpr (*operation)(PrimExpr, PrimExpr, Span)>
ExprDoc BinaryHelperDocTranslate(DocTranslatorObj* d, const T* node, const char* helper) {
  ExprDoc a = d->Translate(node->a).value();
  ExprDoc b = d->Translate(node->b).value();
  try {
    PrimExpr replay = operation(node->a, node->b, Span());
    if (const auto* result = replay.template as<T>();
        result && result->a.same_as(node->a) && result->b.same_as(node->b)) {
      return NamespaceDoc("tirx")->Attr(helper)->Call({a, b});
    }
  } catch (const ffi::Error&) {
    // Keep the explicit constructor when the helper cannot preserve its inputs.
  }
  return NamespaceDoc("tirx")->Attr(std::strrchr(T::_type_key, '.') + 1)->Call({a, b});
}

ffi::Optional<ExprDoc> BitwiseNotDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::BitwiseNotNode>(input);
  return UnaryOpDocTranslate<prim::BitwiseNotNode, OperationDocNode::Kind::kInvert,
                             tvm::bitwise_neg>(d, node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::BitwiseNotNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BitwiseNotDocTranslate>());
}

ffi::Optional<ExprDoc> NotDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::NotNode>(input);
  return UnaryOpDocTranslate<prim::NotNode, OperationDocNode::Kind::kNot, tvm::logical_not>(d,
                                                                                            node);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::NotNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&NotDocTranslate>());
}

ffi::Optional<ExprDoc> StringImmDocTranslate(DocTranslatorObj*, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* imm =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const StringImmNode>(input);
  return LiteralDoc::Str(imm->value, std::nullopt);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<StringImmNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&StringImmDocTranslate>());
}

ffi::Optional<ExprDoc> CastDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object*) {
  const auto* cast =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::CastNode>(input);
  return NamespaceDoc("tirx")->Attr("Cast")->Call(
      {LiteralDoc::DataType(cast->ty.as_or_throw<PrimType>()->dtype, std::nullopt),
       d->Translate(cast->value).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::CastNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&CastDocTranslate>());
}

ffi::Optional<ExprDoc> SelectDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                          const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::SelectNode>(input);
  return NamespaceDoc("tirx")->Attr("Select")->Call({d->Translate(node->condition).value(),
                                                     d->Translate(node->true_value).value(),
                                                     d->Translate(node->false_value).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::SelectNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&SelectDocTranslate>());
}

ffi::Optional<ExprDoc> RampDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::RampNode>(input);
  return NamespaceDoc("tirx")->Attr("Ramp")->Call({d->Translate(node->base).value(),
                                                   d->Translate(node->stride).value(),
                                                   d->Translate(node->lanes).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::RampNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&RampDocTranslate>());
}

ffi::Optional<ExprDoc> BroadcastDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::BroadcastNode>(input);
  return NamespaceDoc("tirx")
      ->Attr("Broadcast")
      ->Call({d->Translate(node->value).value(), d->Translate(node->lanes).value()});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::BroadcastNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BroadcastDocTranslate>());
}

ffi::Optional<ExprDoc> ShuffleDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                           const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::ShuffleNode>(input);
  ExprDoc vectors = AnyValue(d, node->vectors);
  ExprDoc indices = AnyValue(d, node->indices);
  d->RecordOrigin(vectors, node->vectors);
  d->RecordOrigin(indices, node->indices);
  return NamespaceDoc("tirx")->Attr("Shuffle")->Call({vectors, indices});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::ShuffleNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&ShuffleDocTranslate>());
}

ffi::Optional<ExprDoc> LetDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::LetNode>(input);
  ExprDoc value = d->Translate(node->value).value();
  IdDoc var = VarDoc(d, node->var);
  if (auto type = node->var->ty.as<PrimType>()) {
    ExprDoc declaration = NamespaceDoc("ir")->Attr("dynamic")->Call(
        {LiteralDoc::Str(node->var->name, std::nullopt)}, {"dtype"},
        {LiteralDoc::DataType(type.value()->dtype, std::nullopt)});
    d->Emit(AssignDoc(var, declaration, std::nullopt), node->var);
  }
  ExprDoc body = d->Translate(node->body).value();
  return NamespaceDoc("tirx")->Attr("Let")->Call({body}, {"where"}, {DictDoc({var}, {value})});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::LetNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&LetDocTranslate>());
}

ffi::Optional<ExprDoc> DivDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::DivNode>(input);
  PrimType a_type = node->a.ty();
  PrimType b_type = node->b.ty();
  if (a_type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
      b_type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
    return NamespaceDoc("tirx")->Attr("Div")->Call(
        {d->Translate(node->a).value(), d->Translate(node->b).value()});
  }
  return BinaryOpDocTranslate<prim::DivNode, OperationDocNode::Kind::kDiv, tvm::div>(d, input,
                                                                                     destination);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::DivNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&DivDocTranslate>());
  ffi::reflection::TypeAttrDef<prim::AddNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::AddNode, OperationDocNode::Kind::kAdd, tvm::add>>());
  ffi::reflection::TypeAttrDef<prim::SubNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::SubNode, OperationDocNode::Kind::kSub, tvm::sub>>());
  ffi::reflection::TypeAttrDef<prim::MulNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::MulNode, OperationDocNode::Kind::kMult, tvm::mul>>());
  ffi::reflection::TypeAttrDef<prim::FloorDivNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BinaryOpDocTranslate<
                         prim::FloorDivNode, OperationDocNode::Kind::kFloorDiv, tvm::floordiv>>());
  ffi::reflection::TypeAttrDef<prim::FloorModNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BinaryOpDocTranslate<
                         prim::FloorModNode, OperationDocNode::Kind::kMod, tvm::floormod>>());
  ffi::reflection::TypeAttrDef<prim::LShiftNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BinaryOpDocTranslate<
                         prim::LShiftNode, OperationDocNode::Kind::kLShift, tvm::left_shift>>());
  ffi::reflection::TypeAttrDef<prim::RShiftNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BinaryOpDocTranslate<
                         prim::RShiftNode, OperationDocNode::Kind::kRShift, tvm::right_shift>>());
  ffi::reflection::TypeAttrDef<prim::BitwiseAndNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::BitwiseAndNode, OperationDocNode::Kind::kBitAnd, tvm::bitwise_and>>());
  ffi::reflection::TypeAttrDef<prim::BitwiseOrNode>().attr(
      kDocTranslate, FDocTranslate::FromNative<&BinaryOpDocTranslate<
                         prim::BitwiseOrNode, OperationDocNode::Kind::kBitOr, tvm::bitwise_or>>());
  ffi::reflection::TypeAttrDef<prim::BitwiseXorNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<&BinaryOpDocTranslate<
          prim::BitwiseXorNode, OperationDocNode::Kind::kBitXor, tvm::bitwise_xor>>());
  ffi::reflection::TypeAttrDef<prim::LTNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::LTNode, OperationDocNode::Kind::kLt, tvm::less>>());
  ffi::reflection::TypeAttrDef<prim::LENode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::LENode, OperationDocNode::Kind::kLtE, tvm::less_equal>>());
  ffi::reflection::TypeAttrDef<prim::EQNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::EQNode, OperationDocNode::Kind::kEq, tvm::equal>>());
  ffi::reflection::TypeAttrDef<prim::NENode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::NENode, OperationDocNode::Kind::kNotEq, tvm::not_equal>>());
  ffi::reflection::TypeAttrDef<prim::GTNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::GTNode, OperationDocNode::Kind::kGt, tvm::greater>>());
  ffi::reflection::TypeAttrDef<prim::GENode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::GENode, OperationDocNode::Kind::kGtE, tvm::greater_equal>>());
  ffi::reflection::TypeAttrDef<prim::AndNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::AndNode, OperationDocNode::Kind::kAnd, tvm::logical_and>>());
  ffi::reflection::TypeAttrDef<prim::OrNode>().attr(
      kDocTranslate,
      FDocTranslate::FromNative<
          &BinaryOpDocTranslate<prim::OrNode, OperationDocNode::Kind::kOr, tvm::logical_or>>());
}

ffi::Optional<ExprDoc> ModDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::ModNode>(input);
  return BinaryHelperDocTranslate<prim::ModNode, tvm::truncmod>(d, node, "truncmod");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::ModNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&ModDocTranslate>());
}

ffi::Optional<ExprDoc> MinDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::MinNode>(input);
  return BinaryHelperDocTranslate<prim::MinNode, tvm::min>(d, node, "min");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::MinNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&MinDocTranslate>());
}

ffi::Optional<ExprDoc> MaxDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object*) {
  const auto* node =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const prim::MaxNode>(input);
  return BinaryHelperDocTranslate<prim::MaxNode, tvm::max>(d, node, "max");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<prim::MaxNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&MaxDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
