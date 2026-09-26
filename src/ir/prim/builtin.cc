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

Op likely() {
  static const Op op = Op::Get("prim.likely");
  return op;
}

Op if_then_else() {
  static const Op op = Op::Get("prim.if_then_else");
  return op;
}

Op vscale() {
  static const Op op = Op::Get("prim.vscale");
  return op;
}

Op ceil() {
  static const Op op = Op::Get("prim.ceil");
  return op;
}

Op log2() {
  static const Op op = Op::Get("prim.log2");
  return op;
}

Op clz() {
  static const Op op = Op::Get("prim.clz");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.likely")
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind",
                                 static_cast<int64_t>(CallEffectKind::kExprAnnotation))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.if_then_else")
      .set_num_inputs(3)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.vscale")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));

  OpDef("prim.ceil")
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.log2")
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>("TVectorizable", true);

  OpDef("prim.clz")
      .set_num_inputs(1)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

}  // namespace builtin
}  // namespace prim
}  // namespace tvm
