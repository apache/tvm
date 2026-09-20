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
#define PRIM_DEFINE_BUILTIN_FUNC(OpName)            \
  const Op& OpName() {                              \
    static const Op& op = Op::Get("prim." #OpName); \
    return op;                                      \
  }                                                 \
  TVM_REGISTER_OP("prim." #OpName)

PRIM_DEFINE_BUILTIN_FUNC(likely)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kExprAnnotation))
    .set_attr<bool>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(if_then_else)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
PRIM_DEFINE_BUILTIN_FUNC(vscale).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

PRIM_DEFINE_BUILTIN_FUNC(ceil)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<bool>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(log2)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<bool>("TVectorizable", true);

PRIM_DEFINE_BUILTIN_FUNC(clz).set_num_inputs(1).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

#undef PRIM_DEFINE_BUILTIN_FUNC
}  // namespace builtin
}  // namespace prim
}  // namespace tvm
