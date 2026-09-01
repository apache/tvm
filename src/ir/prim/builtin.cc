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
#include <tvm/ir/prim/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace prim {
namespace builtin {
using namespace tvm::tirx;

#define PRIM_DEFINE_BUILTIN_FUNC(OpName)                           \
  const Op& OpName() {                                             \
    static const Op& op = Op::Get("ir.prim." #OpName);             \
    return op;                                                     \
  }                                                                \
  TVM_REGISTER_OP("ir.prim." #OpName)                              \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName) \
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"), 1)

PRIM_DEFINE_BUILTIN_FUNC(likely)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kExprAnnotation))
    .set_attr<TVectorizable>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(bitwise_and)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(bitwise_or)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(bitwise_xor)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(bitwise_not)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(shift_left)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(shift_right)
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
PRIM_DEFINE_BUILTIN_FUNC(if_then_else)
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
PRIM_DEFINE_BUILTIN_FUNC(vscale).set_attr<TCallEffectKind>(
    "TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

#undef PRIM_DEFINE_BUILTIN_FUNC
}  // namespace builtin
}  // namespace prim
}  // namespace tvm
