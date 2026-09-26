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

#include "../op_getter.h"

namespace tvm {
namespace prim {
namespace builtin {

TVM_DEFINE_CACHED_OP_GETTER(likely, "prim.likely")
TVM_DEFINE_CACHED_OP_GETTER(if_then_else, "prim.if_then_else")
TVM_DEFINE_CACHED_OP_GETTER(vscale, "prim.vscale")
TVM_DEFINE_CACHED_OP_GETTER(ceil, "prim.ceil")
TVM_DEFINE_CACHED_OP_GETTER(log2, "prim.log2")
TVM_DEFINE_CACHED_OP_GETTER(clz, "prim.clz")

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.likely")
      .arg_types<PrimExpr>()
      .arg("x", "")
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kExprAnnotation))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.if_then_else")
      .arg_types<PrimExpr, Expr, Expr>()
      .arg("condition", "")
      .arg("true_value", "")
      .arg("false_value", "")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.vscale")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.ceil")
      .arg_types<PrimExpr>()
      .arg("x", "")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.log2")
      .arg_types<PrimExpr>()
      .arg("x", "")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.clz")
      .arg_types<PrimExpr>()
      .arg("x", "")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

}  // namespace builtin
}  // namespace prim
}  // namespace tvm
