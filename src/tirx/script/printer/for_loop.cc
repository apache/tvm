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
#include <tvm/sym/analyzer.h>

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

namespace {

ffi::Optional<ExprDoc> ForDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                       const ffi::Object* destination) {
  const auto* loop =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const tirx::ForNode>(input);
  TVM_FFI_CHECK(destination == nullptr, TypeError)
      << "printer statement-only node cannot fulfill a destination";
  ffi::String method;
  switch (loop->kind) {
    case tirx::ForKind::kSerial:
      method = "serial";
      break;
    case tirx::ForKind::kParallel:
      method = "parallel";
      break;
    case tirx::ForKind::kVectorized:
      method = "vectorized";
      break;
    case tirx::ForKind::kUnrolled:
      method = "unroll";
      break;
    case tirx::ForKind::kThreadBinding:
      method = "thread_binding";
      break;
    default:
      TVM_FFI_THROW(TypeError) << "printer unknown loop kind";
  }
  ExprDoc min = d->Translate(loop->min).value();
  ExprDoc extent = d->Translate(loop->extent).value();
  ExprDoc end = OperationDoc(OperationDocNode::Kind::kAdd, {min, extent});
  if (prim::is_zero(loop->min)) {
    end = extent;
  } else if (loop->min.as<IntImmNode>() && loop->extent.as<IntImmNode>()) {
    // Python integer addition can widen the endpoint before the builder sees
    // it. Keep the IR operation and its dtype for literal bounds as well.
    end = NamespaceDoc("tirx")->Attr("Add")->Call({min, extent});
  }
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (loop->kind == tirx::ForKind::kThreadBinding) {
    TVM_FFI_CHECK(loop->thread_binding.has_value(), TypeError)
        << "printer thread-binding loop lacks thread tag";
    keys.push_back("thread");
    values.push_back(LiteralDoc::Str(loop->thread_binding.value(), std::nullopt));
  }
  bool unroll_option = false;
  if (loop->kind == tirx::ForKind::kSerial && loop->annotations.size() == 1) {
    const auto& [key, value] = *loop->annotations.begin();
    auto boolean = value.as<bool>();
    auto integer = value.as<int64_t>();
    if (key == "disable_unroll" && boolean.value_or(false)) {
      keys.push_back("unroll");
      values.push_back(LiteralDoc::Boolean(false, std::nullopt));
      unroll_option = true;
    } else if (key == "pragma_unroll" &&
               (boolean.value_or(false) || (integer.has_value() && integer.value() > 0))) {
      keys.push_back("unroll");
      values.push_back(AnyValue(d, value));
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
      annotation_values.push_back(AnyValue(d, value));
    }
    keys.push_back("annotations");
    values.push_back(DictDoc(annotation_keys, annotation_values));
  }
  if (loop->step.has_value()) {
    TVM_FFI_CHECK(loop->kind != tirx::ForKind::kThreadBinding, TypeError)
        << "printer thread-binding loop cannot have a step";
    keys.push_back("step");
    values.push_back(d->Translate(loop->step.value()).value());
  }
  // For requires the bounds and index to have the same type. Their typed
  // expressions therefore carry the index dtype without a separate keyword.
  ForDoc doc(ffi::UnsafeInit{});
  {
    IdDoc var = VarDoc(d, loop->loop_var);
    ffi::Array<ExprDoc> bounds = {min, end};
    if (loop->kind == tirx::ForKind::kThreadBinding && prim::is_zero(loop->min) &&
        loop->min.ty() == loop->extent.ty()) {
      bounds = {end};
    }
    ExprDoc callee = NamespaceDoc("tirx")->Attr(method);
    if (loop->kind == tirx::ForKind::kSerial && loop->annotations.empty()) {
      callee = IdDoc("range");
      // range's step is positional. Retain even an explicit unit step because
      // it is part of the source For node, unlike an absent step.
      if (loop->step.has_value()) {
        bounds.push_back(values.back());
        keys.pop_back();
        values.pop_back();
      } else if (prim::is_zero(loop->min) && loop->min.ty() == PrimType::Int(32)) {
        bounds.erase(bounds.begin());
      }
    }
    doc = ForDoc(var, callee->Call(bounds, keys, values), Body(loop->body, d));
  }
  d->Emit(doc, ffi::GetRef<ffi::ObjectRef>(loop));
  return std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<tirx::ForNode>().attr(kDocTranslate,
                                                     FDocTranslate::FromNative<&ForDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
