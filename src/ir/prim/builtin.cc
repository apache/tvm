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
#include <tvm/ir/op_attr_types.h>
#include <tvm/ir/prim/builtin.h>

namespace tvm {
namespace prim {
namespace builtin {

Type InferTypeVScale(const CallNode*) { return PrimType::Int(32); }

#define TVM_DEFINE_CACHED_OP_GETTER(Name, RegisteredName) \
  const Op& Name() {                                      \
    static const Op op = Op::Get(RegisteredName);         \
    return op;                                            \
  }

TVM_DEFINE_CACHED_OP_GETTER(likely, "prim.likely")
TVM_DEFINE_CACHED_OP_GETTER(if_then_else, "prim.if_then_else")
TVM_DEFINE_CACHED_OP_GETTER(vscale, "prim.vscale")
TVM_DEFINE_CACHED_OP_GETTER(ceil, "prim.ceil")
TVM_DEFINE_CACHED_OP_GETTER(log2, "prim.log2")
TVM_DEFINE_CACHED_OP_GETTER(clz, "prim.clz")

#undef TVM_DEFINE_CACHED_OP_GETTER

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.likely")
      .signature(sig::arg("x", "The input value."))
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kExprAnnotation))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.if_then_else")
      .signature(sig::arg("condition", "The condition."),
                 sig::arg("true_value", "The value when the condition is true."),
                 sig::arg("false_value", "The value when the condition is false."))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.vscale")
      .signature()
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeVScale>())
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.ceil")
      .signature(sig::arg("x", "The input value."))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.log2")
      .signature(sig::arg("x", "The input value."))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.clz")
      .signature(sig::arg("x", "The input value."))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

}  // namespace builtin
}  // namespace prim
}  // namespace tvm
