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
#include <tvm/ir/op.h>
#include <tvm/s_tir/stmt.h>

#include "../../../tirx/script/printer/utils.h"

namespace tvm {
namespace script {
namespace printer {
namespace details {
namespace {

ffi::Optional<ExprDoc> CpAsyncRawDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  // This constructor takes an element dtype followed by the five stored operands.
  // The dtype is carried by the Call itself, not derived from a pointer argument.
  if (!CanTranslateExplicitResultCall(call) || call->args.size() != 5) {
    return RawCall(d, call);
  }
  ffi::Array<ExprDoc> args = {TypeValue(d, call->ty)};
  for (const Expr& arg : call->args) {
    args.push_back(MaterializeCallArgument(d, arg, d->Translate(arg).value()));
  }
  return NamespaceDoc("tirx")->Attr("s_tir")->Attr("cp_async_raw")->Call(args);
}

ffi::Optional<ExprDoc> AsyncQueueDocTranslate(DocTranslatorObj* d, ffi::AnyView input,
                                              const ffi::Object*) {
  const auto* call =
      ffi::details::AnyUnsafe::RawObjectPtrFromAnyViewAfterCheck<const CallNode>(input);
  if (!CanTranslateExplicitResultCall(call) ||
      !ffi::StructuralEqual()(call->ty, PrimType::Void())) {
    return RawCall(d, call);
  }
  ffi::Array<ExprDoc> args;
  for (const Expr& arg : call->args) {
    args.push_back(MaterializeCallArgument(d, arg, d->Translate(arg).value()));
  }
  return NamespaceDoc("s_tir")
      ->Attr(call->op.same_as(s_tir::async_commit()) ? "async_commit" : "async_wait")
      ->Call(args);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("s_tir.async_commit")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&AsyncQueueDocTranslate>());
  OpDef("s_tir.async_wait")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&AsyncQueueDocTranslate>());
  OpDef("tirx.s_tir.cp_async_raw")
      .set_attr<FDocTranslate>(kOpCallDocTranslate,
                               FDocTranslate::FromNative<&CpAsyncRawDocTranslate>());
}

}  // namespace

}  // namespace details
}  // namespace printer
}  // namespace script
}  // namespace tvm
