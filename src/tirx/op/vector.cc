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

/*!
 * \file tirx/op/vector.cc
 * \brief TIRx vector operations.
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/vector.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tirx {

template <bool Combine>
Type InferTypeVectorPart(const CallNode* call) {
  TVM_FFI_CHECK_EQ(call->args.size(), Combine ? 2U : 1U, ValueError);
  PrimType input = call->args[0]->ty.as_or_throw<PrimType>();
  if constexpr (Combine) {
    TVM_FFI_CHECK(ffi::StructuralEqual()(input, call->args[1]->ty), TypeError)
        << "vectorcombine requires equal input vector types";
  }
  int lanes = input.IsScalableVector() ? input.VScaleFactor() : input.lanes();
  if constexpr (!Combine) {
    TVM_FFI_CHECK(lanes > 1 && lanes % 2 == 0, TypeError)
        << "vector half requires an even lane count";
  }
  int result_lanes = Combine ? lanes * 2 : lanes / 2;
  return input.IsScalableVector()
             ? PrimType::ScalableVector(input.code(), input.bits(), result_lanes)
             : input.WithLanes(result_lanes);
}

const Op& vectorhigh_op() {
  static const Op op = Op::Get("tirx.vectorhigh");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.vectorhigh")
      .signature(sig::arg<PrimExpr>("vec", "The input vector."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeVectorPart<false>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vectorhigh"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& vectorlow_op() {
  static const Op op = Op::Get("tirx.vectorlow");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.vectorlow")
      .signature(sig::arg<PrimExpr>("vec", "The input vector."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeVectorPart<false>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vectorlow"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& vectorcombine_op() {
  static const Op op = Op::Get("tirx.vectorcombine");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.vectorcombine")
      .signature(sig::arg<PrimExpr>("vec1", "The first input vector."),
                 sig::arg<PrimExpr>("vec2", "The second input vector."))
      .set_attr<FInferType>("FInferType", FInferType::FromNative<&InferTypeVectorPart<true>>())
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.vectorcombine"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

const Op& get_active_lane_mask_op() {
  static const Op op = Op::Get("tirx.get_active_lane_mask");
  return op;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.get_active_lane_mask")
      .signature(sig::arg<IntExpr>("base", "The base value."),
                 sig::arg<IntExpr>("limit", "The limit value."))
      .set_attr<TScriptPrinterName>("TScriptPrinterName", ffi::String("tirx.get_active_lane_mask"))
      .set_attr<TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure));
}

}  // namespace tirx
}  // namespace tvm
