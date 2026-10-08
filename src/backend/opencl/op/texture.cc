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

#include <tvm/backend/opencl/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace backend {
namespace opencl {

using namespace tirx;

const Op& texture2d_store_op() {
  static const Op op = Op::Get("tirx.opencl.texture2d_store");
  return op;
}

const Op& texture2d_load_op() {
  static const Op op = Op::Get("tirx.opencl.texture2d_load");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.opencl.texture2d_store")
      .signature(sig::arg("texture", "The texture."), sig::arg<IntExpr>("x", "The input value."),
                 sig::arg<IntExpr>("y", "The second input value."),
                 sig::arg<IntExpr>("z", "The third input value."),
                 sig::arg<IntExpr>("channel_size", "The number of channels."),
                 sig::arg<PrimExpr>("value", "The value to use."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TVectorizable>("TVectorizable", true)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.opencl.texture2d_load")
      .signature(sig::arg("texture", "The texture."), sig::arg<IntExpr>("x", "The input value."),
                 sig::arg<IntExpr>("y", "The second input value."),
                 sig::arg<IntExpr>("z", "The third input value."),
                 sig::arg<IntExpr>("channel_size", "The number of channels."),
                 sig::arg<PrimExpr>("element_index", "The element index within a texture channel."))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TVectorizable>("TVectorizable", true)
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace opencl
}  // namespace backend
}  // namespace tvm
