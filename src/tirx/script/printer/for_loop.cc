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

#include <optional>

#include "../../../script/printer/ir/utils.h"
#include "utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {

namespace {

ffi::Optional<ExprDoc> TIRxForIteratorDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                                   const ffi::Object*) {
  const auto* loop =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const ForNode>(input);
  bool use_thread_binding =
      tvm::tirx::GetThreadBinding(loop).has_value() && !loop->step.has_value();
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (use_thread_binding) {
    keys.push_back("thread");
    values.push_back(LiteralDoc::Str(tvm::tirx::GetThreadBinding(loop).value(), std::nullopt));
  }
  bool unroll_option = false;
  if (loop->kind == ForKind::kDefault && loop->annotations.size() == 1) {
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
  auto annotations = loop->annotations;
  if (use_thread_binding) annotations.erase("thread_binding");
  if (unroll_option) annotations.clear();
  return ForIterator(d, loop, keys, values, annotations, use_thread_binding);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<ForNode>().attr(
      type_attr::kForIteratorDocTranslate,
      FDocTranslate::FromNative<&TIRxForIteratorDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
