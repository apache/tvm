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
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

bool IsScalarBuffer(DocTranslatorObj* d, const Expr& source) {
  auto var = source.as<Var>();
  if (!var.has_value()) return false;
  IdDoc id = d->VarGetOrAllocId(var.value(), false);
  auto column = d->GetOrCreateExtraState<ffi::Dict<IdDoc, bool>>("tirx.buffer_as_mutable_var");
  auto found = column.find(id);
  return found != column.end() && (*found).second;
}

namespace {

ffi::Optional<ExprDoc> TensorVarDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                             const ffi::Object* destination) {
  Var var = input.cast<Var>();
  IdDoc id = d->VarGetOrAllocId(var, false);
  if (destination == var.get() && d->GetImplicitDefs().count(var)) {
    VarDoc(d, var);
    ExprDoc rhs = NamespaceDoc("tirx")->Attr("Var")->Call(
        {LiteralDoc::Str(var->name, std::nullopt), d->Translate(var->ty).value()});
    EmitVarDefinition(d, var, rhs);
    return std::nullopt;
  }
  // Mutable scalar syntax binds a TensorLoad; resource uses need its allocation.
  return IsScalarBuffer(d, var) ? IdDoc(id->name)->Attr("source") : ExprDoc(IdDoc(id->name));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::TensorTypeNode>().attr(
      type_attr::kDocTranslateVarByTy, FDocTranslate::FromNative<&TensorVarDocTranslate>());
}

ffi::Optional<ExprDoc> BufferOperationDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                   const ffi::Object* destination) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  // Surface buffer constructors emit a binding, so they cannot replace an
  // allocation expression nested inside another Call or statement.
  if (!destination) return RawCall(d, call);
  TVM_FFI_CHECK(destination->IsInstance<VarNode>(), TypeError)
      << "Buffer operation destination must be a variable";
  auto var = ffi::GetRef<Var>(static_cast<const VarNode*>(destination));
  if (!ffi::StructuralEqual()(var->ty, call->ty)) return RawCall(d, call);
  bool is_alloc = call->op.same_as(tirx::alloc_tensor_op());
  size_t shape_index = is_alloc ? 0 : 1;
  auto buffer = call->ty.as<tirx::TensorType>();
  if (!buffer || (call->args.size() != shape_index + 3 && !(is_alloc && call->args.size() == 4)) ||
      !call->ty_args.empty() || call->attrs.defined() != is_alloc ||
      (call->attrs.defined() && !call->attrs.as<DictAttrsNode>())) {
    return RawCall(d, call);
  }
  auto shape = call->args[shape_index].as<TupleNode>();
  auto dtype = call->args[shape_index + 1].as<DataTypeImmNode>();
  auto scope = call->args[shape_index + 2].as<StringImmNode>();
  if (!shape || !dtype || !scope || !ffi::StructuralEqual()(shape->fields, buffer.value()->shape) ||
      dtype->value != buffer.value()->dtype->dtype ||
      scope->value != buffer.value()->storage_scope || scope->value.empty() ||
      buffer.value()->data_alignment <= 0 || buffer.value()->offset_factor == 0) {
    return RawCall(d, call);
  }
  ffi::Map<ffi::String, ffi::Any> annotations;
  if (is_alloc) annotations = call->attrs.as_or_throw<DictAttrs>()->dict;
  for (const auto& [key, value] : annotations) {
    if (value.type_index() == ffi::TypeIndex::kTVMFFIInt ||
        value.type_index() == ffi::TypeIndex::kTVMFFIBool ||
        value.type_index() == ffi::TypeIndex::kTVMFFIFloat) {
      return RawCall(d, call);
    }
  }
  ffi::Optional<Expr> data = is_alloc ? std::nullopt : ffi::Optional<Expr>(call->args[0]);
  if (!is_alloc) {
    const auto* pointer = data.value()->ty.as<PtrTypeNode>();
    if (!pointer || pointer->storage_scope != scope->value) return RawCall(d, call);
  }
  CallDoc rhs = d->Translate(buffer.value()).value().as_or_throw<CallDoc>();
  if (rhs->callee.as_or_throw<AttrAccessDoc>()->name != "Tensor") return RawCall(d, call);
  ffi::String method = is_alloc ? "alloc_tensor" : "decl_tensor";
  if (is_alloc && (scope->value == "local" || scope->value == "shared")) {
    method = scope->value == "local" ? "alloc_local" : "alloc_shared";
    for (size_t i = 0; i < rhs->kwargs_keys.size(); ++i) {
      if (rhs->kwargs_keys[i] == "scope") {
        rhs->kwargs_keys.erase(rhs->kwargs_keys.begin() + i);
        rhs->kwargs_values.erase(rhs->kwargs_values.begin() + i);
        break;
      }
    }
  }
  rhs->callee = NamespaceDoc("tirx")->Attr(method);
  if (is_alloc && call->args.size() == 4) {
    auto placement = call->args[3].as<tvm::TupleNode>();
    if (!placement) return RawCall(d, call);
    rhs->kwargs_keys.push_back("allocated_addr");
    rhs->kwargs_values.push_back(AnyValue(d, placement->fields));
  }
  if (data.has_value()) {
    rhs->kwargs_keys.insert(rhs->kwargs_keys.begin(), "data");
    rhs->kwargs_values.insert(rhs->kwargs_values.begin(), d->Translate(data.value()).value());
  }
  if (!annotations.empty()) {
    CallDoc doc = rhs.as_or_throw<CallDoc>();
    doc->kwargs_keys.push_back("annotations");
    doc->kwargs_values.push_back(AnyValue(d, annotations));
  }
  IdDoc lhs = VarDoc(d, var);
  AssignDoc allocation(lhs, rhs, std::nullopt);
  if (is_alloc && d->GetExtraConfig<bool>("tirx.scalar_buffer_as_mutable_var", true) &&
      annotations.empty() && call->args.size() == 3 && scope->value == "local" &&
      buffer.value()->IsScalar(true)) {
    ExprDoc annotation = d->Translate(buffer.value()->dtype).value();
    // T.bool is a type annotation but has no mutable-declaration syntax.
    if (annotation.as<AttrAccessDoc>() && !buffer.value()->dtype.MatchesCode(kDLBool)) {
      auto scalars = d->GetOrCreateExtraState<ffi::Dict<IdDoc, bool>>("tirx.buffer_as_mutable_var");
      scalars.Set(d->VarGetOrAllocId(var, true), true);
      allocation->rhs = std::nullopt;
      allocation->annotation = annotation;
    }
  }
  d->Emit(allocation, ffi::GetRef<Call>(call));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char* name : {"tirx.alloc_tensor", "tirx.decl_tensor"}) {
    OpDef(name).set_attr<FDocTranslate>(tvm::script::printer::op_attr::kOpCallDocTranslate,
                                        FDocTranslate::FromNative<&BufferOperationDocTranslate>());
  }
}

ffi::Optional<ExprDoc> TensorTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* buffer =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::TensorTypeNode>(input);
  bool default_offset =
      ffi::StructuralEqual()(buffer->elem_offset, IntImm(PrimType(buffer->DefaultIndexType()), 0));
  // The buffer type constructor normalizes these fields.
  // Keep the explicit type constructor when that normalization would change the IR.
  if (buffer->storage_scope.empty() || buffer->data_alignment <= 0 || buffer->offset_factor == 0) {
    return NamespaceDoc("ir")
        ->Attr("make_node")
        ->Call({LiteralDoc::Str("tirx.TensorType", std::nullopt)},
               {"dtype", "storage_scope", "shape", "strides", "elem_offset", "data_alignment",
                "offset_factor", "layout"},
               {TypeValue(d, buffer->dtype, false),
                LiteralDoc::Str(buffer->storage_scope, std::nullopt), AnyValue(d, buffer->shape),
                AnyValue(d, buffer->strides), d->Translate(buffer->elem_offset).value(),
                LiteralDoc::Int(buffer->data_alignment, std::nullopt),
                LiteralDoc::Int(buffer->offset_factor, std::nullopt),
                buffer->layout.has_value() ? d->Translate(buffer->layout.value()).value()
                                           : LiteralDoc::None(std::nullopt)});
  }
  ffi::Array<ExprDoc> shape;
  for (const PrimExpr& extent : buffer->shape) shape.push_back(d->Translate(extent).value());
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (buffer->storage_scope != "global") {
    keys.push_back("scope");
    values.push_back(LiteralDoc::Str(buffer->storage_scope, std::nullopt));
  }
  if (!buffer->strides.empty()) {
    ffi::Array<ExprDoc> strides;
    for (const PrimExpr& stride : buffer->strides) strides.push_back(d->Translate(stride).value());
    keys.push_back("strides");
    values.push_back(TupleDoc(strides));
  }
  // A nondefault offset_factor makes an omitted offset symbolic in the parser.
  if ((!default_offset || buffer->offset_factor != 1)) {
    keys.push_back("elem_offset");
    values.push_back(d->Translate(buffer->elem_offset).value());
  }
  if (buffer->data_alignment != 64) {
    keys.push_back("align");
    values.push_back(LiteralDoc::Int(buffer->data_alignment, std::nullopt));
  }
  if (buffer->offset_factor != 1) {
    keys.push_back("offset_factor");
    values.push_back(LiteralDoc::Int(buffer->offset_factor, std::nullopt));
  }
  if (buffer->layout.has_value()) {
    keys.push_back("layout");
    if (ffi::StructuralEqual()(buffer->layout.value(),
                               tirx::TileLayoutNode::DefaultLayout(buffer->shape))) {
      values.push_back(LiteralDoc::Str("default", std::nullopt));
    } else {
      values.push_back(d->Translate(buffer->layout.value()).value());
    }
  }
  if (!buffer->layout.has_value()) {
    keys.push_back("layout");
    values.push_back(LiteralDoc::None(std::nullopt));
  }
  return NamespaceDoc("tirx")->Attr("Tensor")->Call(
      {TupleDoc(shape), LiteralDoc::DataType(buffer->dtype->dtype, std::nullopt)}, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::TensorTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&TensorTypeDocTranslate>());
}

}  // namespace

namespace {

ffi::Optional<ExprDoc> TensorStoreDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                               const ffi::Object* destination) {
  const auto* store =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorStoreNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  auto tensor_var = store->dest.as<tvm::tirx::TensorVar>();
  bool scalar = tensor_var.has_value() && IsScalarBuffer(d, tensor_var.value());
  ExprDoc buffer = scalar ? ExprDoc(VarDoc(d, store->dest.as<Var>().value(), false))
                          : d->Translate(store->dest).value();
  ExprDoc value = d->Translate(store->value).value();
  ExprDoc lhs = scalar ? buffer : ExprDoc(IndexDoc(buffer, TensorIndices(d, store->indices, true)));
  d->Emit(AssignDoc(lhs, value, std::nullopt), ffi::GetRef<ffi::ObjectRef>(store));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TensorStoreNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&TensorStoreDocTranslate>());
}

ffi::Optional<ExprDoc> TIRxTensorLoadDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                  const ffi::Object*) {
  const auto* load =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const TensorLoadNode>(input);
  if (IsScalarBuffer(d, load->source)) {
    return VarDoc(d, load->source.as<Var>().value(), false);
  }
  ExprDoc source = d->Translate(load->source).value();
  return IndexDoc(source, TensorIndices(d, load->indices));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::TensorTypeNode>().attr(
      tvm::script::printer::type_attr::kDocTranslateTensorLoadBySourceTy,
      FDocTranslate::FromNative<&TIRxTensorLoadDocTranslate>());
}

ffi::Optional<ExprDoc> IterDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                        const ffi::Object*) {
  const auto* iter =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::IterNode>(input);
  ExprDoc axis = LiteralDoc::Str(iter->axis.name(), std::nullopt);
  d->RecordOrigin(axis, iter->axis);
  return NamespaceDoc("tirx")->Attr("Iter")->Call(
      {d->Translate(iter->extent).value(), d->Translate(iter->stride).value(), axis});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::IterNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&IterDocTranslate>());
}

ffi::Optional<ExprDoc> TileLayoutDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* layout =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::TileLayoutNode>(input);
  auto iters = [&](const ffi::Array<tirx::Iter>& source) {
    ffi::Array<ExprDoc> docs;
    for (const tirx::Iter& iter : source) {
      docs.push_back(d->Translate(iter).value());
    }
    return ListDoc(docs);
  };
  ffi::Array<ffi::String> keys = {"shard", "replica"};
  ffi::Array<ExprDoc> values = {iters(layout->shard), iters(layout->replica)};
  if (!layout->offset.empty()) {
    std::vector<std::pair<tirx::Axis, PrimExpr>> sorted(layout->offset.begin(),
                                                        layout->offset.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first.name() < b.first.name(); });
    ffi::Array<ExprDoc> offset_keys;
    ffi::Array<ExprDoc> offset_values;
    for (const auto& [axis, offset] : sorted) {
      offset_keys.push_back(LiteralDoc::Str(axis.name(), std::nullopt));
      offset_values.push_back(d->Translate(offset).value());
    }
    keys.push_back("offset");
    values.push_back(DictDoc(offset_keys, offset_values));
  }
  return NamespaceDoc("tirx")->Attr("TileLayout")->Attr("from_iters")->Call({}, keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::TileLayoutNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&TileLayoutDocTranslate>());
}

ffi::Optional<ExprDoc> ComposeLayoutDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                 const ffi::Object*) {
  const auto* layout =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::ComposeLayoutNode>(
          input);
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (!layout->swizzle_inner) {
    keys.push_back("swizzle_inner");
    values.push_back(LiteralDoc::Boolean(false, std::nullopt));
  }
  return NamespaceDoc("tirx")
      ->Attr("ComposeLayout")
      ->Call({LiteralDoc::Int(layout->per_element, std::nullopt),
              LiteralDoc::Int(layout->swizzle_len, std::nullopt),
              LiteralDoc::Int(layout->atom_len, std::nullopt),
              d->Translate(layout->tile_layout).value()},
             keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::ComposeLayoutNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&ComposeLayoutDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
