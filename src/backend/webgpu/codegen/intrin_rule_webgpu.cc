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
 * \file intrin_rule_webgpu.cc
 * \brief WebGPU intrinsic rules.
 */
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op_attr_types.h>

#include "../../../target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;

// warp-level primitives. Follows implementation in intrin_rule_metal.cc
struct WebGPUWarpIntrinsic {
  const Op operator()(PrimType t, const Op& orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      static const Op webgpu_subgroup_shuffle_op = Op::Get("tirx.webgpu.subgroup_shuffle");
      return webgpu_subgroup_shuffle_op;
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      static const Op webgpu_subgroup_shuffle_up_op = Op::Get("tirx.webgpu.subgroup_shuffle_up");
      return webgpu_subgroup_shuffle_up_op;
    } else {
      TVM_FFI_ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      static const Op webgpu_subgroup_shuffle_down_op =
          Op::Get("tirx.webgpu.subgroup_shuffle_down");
      return webgpu_subgroup_shuffle_down_op;
    }
  }
};

template <typename T>
static PrimExpr DispatchWebGPUShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  PrimExpr lane = call->args[2].as_or_throw<PrimExpr>();
  PrimExpr lane_or_delta = prim::Cast(PrimType::UInt(32, lane.ty().lanes()), lane);
  ffi::Array<PrimExpr> webgpu_args{call->args[1].as_or_throw<PrimExpr>(), lane_or_delta};
  return Call(e.ty(), T()(e.ty(), call->op.as_or_throw<Op>()), webgpu_args).as_or_throw<PrimExpr>();
}

void RegisterWebGPUIntrinRules() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  // See full list of builtin: https://www.w3.org/TR/WGSL/#builtin-functions

  struct ReturnAbs {
    std::string operator()(PrimType t, std::string name) const { return "abs"; }
  };

  OpDef("tirx.fabs")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<ReturnAbs>);

  OpDef("tirx.acos")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.acosh")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.asin")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.asinh")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.atan")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.atan2")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.cos")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.cosh")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.exp")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.exp2")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.floor")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.fma")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.log")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.pow")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  struct ReturnRound {
    std::string operator()(PrimType t, std::string name) const { return "round"; }
  };

  // WGSL round() uses ties-to-even (banker's rounding), matching IEEE 754 and ONNX Round spec.
  OpDef("tirx.nearbyint")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<ReturnRound>);

  OpDef("tirx.round")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.sin")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.sinh")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.sqrt")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.tan")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  OpDef("tirx.tanh")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchNumericalStableTanh);

  OpDef("tirx.trunc")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

  // extra dispatch
  OpDef("tirx.erf")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchFastErf);

  // warp-level primitives. Follows implementation in intrin_rule_metal.cc
  OpDef("tirx.tvm_warp_shuffle")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                 DispatchWebGPUShuffle<WebGPUWarpIntrinsic>);

  OpDef("tirx.tvm_warp_shuffle_up")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                 DispatchWebGPUShuffle<WebGPUWarpIntrinsic>);

  OpDef("tirx.tvm_warp_shuffle_down")
      .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                 DispatchWebGPUShuffle<WebGPUWarpIntrinsic>);

  // Register low-level WebGPU device intrinsics.
  OpDef("tirx.webgpu.subgroup_shuffle")
      .arg<Expr>("var", "The variable to sync.")
      .arg<Expr>("lane", "The source thread id.")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("webgpu"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName",
                                          ffi::String("webgpu.subgroup_shuffle"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "subgroupShuffle")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.webgpu.subgroup_shuffle_up")
      .arg<Expr>("var", "The variable to sync.")
      .arg<Expr>("delta", "The source lane id offset to be added.")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("webgpu"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName",
                                          ffi::String("webgpu.subgroup_shuffle_up"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "subgroupShuffleUp")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));

  OpDef("tirx.webgpu.subgroup_shuffle_down")
      .arg<Expr>("var", "The variable to sync.")
      .arg<Expr>("delta", "The source lane id offset to be subtracted.")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("webgpu"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName",
                                          ffi::String("webgpu.subgroup_shuffle_down"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "subgroupShuffleDown")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque));
}

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
