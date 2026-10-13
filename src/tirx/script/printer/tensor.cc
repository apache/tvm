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
#include <tvm/tirx/type.h>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include "../../../script/printer/ir/utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> TensorTypeDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* tensor =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::TensorTypeNode>(input);
  // The tensor type constructor normalizes these fields.
  // Keep the explicit type constructor when that normalization would change the IR.
  if (tensor->data_alignment <= 0 || tensor->offset_factor == 0) {
    return NamespaceDoc("ir")
        ->Attr("make_node")
        ->Call({LiteralDoc::Str("tirx.TensorType", std::nullopt)},
               {"dtype", "scope", "shape", "strides", "elem_offset", "data_alignment",
                "offset_factor", "layout"},
               {TypeValue(d, tensor->dtype, false), d->Translate(tensor->storage_scope).value(),
                AnyValue(d, tensor->shape), AnyValue(d, tensor->strides),
                ReflectedValue(d, tensor->elem_offset),
                LiteralDoc::Int(tensor->data_alignment, std::nullopt),
                LiteralDoc::Int(tensor->offset_factor, std::nullopt),
                tensor->layout.has_value() ? d->Translate(tensor->layout.value()).value()
                                           : LiteralDoc::None(std::nullopt)});
  }
  ffi::Array<ExprDoc> shape;
  for (const PrimExpr& dim : tensor->shape) {
    // Python integer shape inputs use the shared PrimExpr conversion (int32).
    if (const auto* imm = dim.as<IntImmNode>();
        imm && imm->ty.as_or_throw<PrimType>()->dtype == PrimType::Int(32)->dtype) {
      ExprDoc literal = LiteralDoc::Int(ffi::GetRef<IntImm>(imm), std::nullopt);
      d->RecordOrigin(literal, dim);
      shape.push_back(literal);
    } else if (dim.as<IntImmNode>()) {
      shape.push_back(ReflectedValue(d, dim));
    } else {
      ExtraStateScope<bool> scalar_context(d, "ir.explicit_scalar_dtype", false);
      shape.push_back(d->Translate(dim).value());
    }
  }
  ExtraStateScope<bool> annotation(d, "ir.type_value", false);
  ffi::Array<ExprDoc> args{TupleDoc(shape), d->Translate(tensor->dtype).value()};
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  auto optional = [&](const char* key, ExprDoc value) {
    keys.push_back(key);
    values.push_back(value);
  };
  if (tensor->storage_scope != runtime::MemoryScope::Global()) {
    optional("scope", d->Translate(tensor->storage_scope).value());
  }
  if (!tensor->strides.empty()) optional("strides", ReflectedValue(d, tensor->strides));
  if (!ffi::StructuralEqual()(tensor->elem_offset, IntImm(tirx::DefaultIndexPrimType(), 0))) {
    optional("elem_offset", ReflectedValue(d, tensor->elem_offset));
  }
  if (tensor->data_alignment != static_cast<int>(runtime::kAllocAlignment)) {
    optional("data_alignment", LiteralDoc::Int(tensor->data_alignment, std::nullopt));
  }
  if (tensor->offset_factor != 1) {
    optional("offset_factor", LiteralDoc::Int(tensor->offset_factor, std::nullopt));
  }
  if (tensor->layout.has_value()) optional("layout", d->Translate(tensor->layout.value()).value());
  return NamespaceDoc("tirx")->Attr("Tensor")->Call(args, keys, values);
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
  ExprDoc tensor = d->Translate(store->dest).value();
  ExprDoc value = d->Translate(store->value).value();
  ExprDoc lhs = IndexDoc(tensor, TensorIndices(d, store->indices, true));
  d->Emit(AssignDoc(lhs, value, std::nullopt), ffi::GetRef<ffi::ObjectRef>(store));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<TensorStoreNode>().attr(
      tvm::script::printer::type_attr::kDocTranslate,
      FDocTranslate::FromNative<&TensorStoreDocTranslate>());
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
