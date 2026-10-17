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
#include <tvm/script/printer/doc_translator.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

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

ffi::Optional<ExprDoc> ForDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination) {
  const auto* loop =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ffi::String method;
  switch (loop->kind) {
    case ForKind::kDefault:
      method = "serial";
      break;
    case ForKind::kParallel:
      method = "parallel";
      break;
    case ForKind::kVectorized:
      method = "vectorized";
      break;
    case ForKind::kUnrolled:
      method = "unroll";
      break;
    default:
      TVM_FFI_THROW(TypeError) << "printer unknown loop kind";
  }
  ExprDoc begin = d->Translate(loop->begin).value();
  ExprDoc end = d->Translate(loop->end).value();
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  bool unroll_option = false;
  if (loop->kind == ForKind::kDefault && loop->annotations.size() == 1) {
    const auto& [key, value] = *loop->annotations.begin();
    auto boolean = value.as<bool>();
    auto integer = value.as<int64_t>();
    if (key == tvm::tirx::attr::kDisableUnroll.name && boolean.value_or(false)) {
      keys.push_back("unroll");
      values.push_back(LiteralDoc::Boolean(false, std::nullopt));
      unroll_option = true;
    } else if (key == tvm::tirx::attr::kPragmaUnroll &&
               (boolean.value_or(false) || (integer.has_value() && integer.value() > 0))) {
      keys.push_back("unroll");
      values.push_back(d->WithExtraStateScope<bool>("ir.explicit_value_type", true,
                                                    [&] { return d->Translate(value).value(); }));
      unroll_option = true;
    }
  }
  if (!unroll_option && !loop->annotations.empty()) {
    std::vector<std::pair<ffi::String, ffi::Any>> sorted;
    for (const auto& [key, value] : loop->annotations) sorted.emplace_back(key, value);
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    ffi::Array<ExprDoc> annotation_keys;
    ffi::Array<ExprDoc> annotation_values;
    for (const auto& [key, value] : sorted) {
      annotation_keys.push_back(LiteralDoc::Str(key, std::nullopt));
      annotation_values.push_back(d->WithExtraStateScope<bool>(
          "ir.explicit_value_type", true, [&] { return d->Translate(value).value(); }));
    }
    keys.push_back("annotations");
    values.push_back(DictDoc(annotation_keys, annotation_values));
  }
  if (!prim::IsOne(loop->step)) {
    keys.push_back("step");
    values.push_back(d->Translate(loop->step).value());
  }
  // For requires the bounds and index to have the same type. Their typed
  // expressions therefore carry the index dtype without a separate keyword.
  ForDoc doc(ffi::UnsafeInit{});
  {
    IdDoc var = VarDoc(d, loop->var);
    ffi::Array<ExprDoc> bounds = {begin, end};
    ExprDoc callee = NamespaceDoc("tirx")->Attr(method);
    if (loop->kind == ForKind::kDefault && loop->annotations.empty()) {
      callee = IdDoc("range");
      // range's step is positional. The builder restores an omitted unit step.
      if (!prim::IsOne(loop->step)) {
        bounds.push_back(values.back());
        keys.pop_back();
        values.pop_back();
      } else if (prim::IsZero(loop->begin) && loop->begin.ty() == PrimType::Int(32)) {
        bounds.erase(bounds.begin());
      }
    }
    doc = ForDoc(var, callee->Call(bounds, keys, values),
                 ToStmtDocArray(d->WithDocScope([&]() { d->Translate(loop->body); })));
  }
  d->Emit(doc, ffi::GetRef<ffi::ObjectRef>(loop));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<ForNode>().attr(tvm::script::printer::type_attr::kDocTranslate.name,
                                               FDocTranslate::FromNative<&ForDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
