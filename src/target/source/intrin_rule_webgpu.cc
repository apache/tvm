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
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;

// warp-level primitives. Follows implementation in intrin_rule_metal.cc
struct WebGPUWarpIntrinsic {
  const Op operator()(DataType t, const Op& orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tir.webgpu.subgroup_shuffle");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tir.webgpu.subgroup_shuffle_up");
    } else {
      TVM_FFI_ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tir.webgpu.subgroup_shuffle_down");
    }
  }
};

template <typename T>
static PrimExpr DispatchWebGPUShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  ffi::Array<PrimExpr> webgpu_args{{call->args[1], call->args[2]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), webgpu_args);
}

// See full list of builtin: https://www.w3.org/TR/WGSL/#builtin-functions

struct ReturnAbs {
  std::string operator()(DataType t, std::string name) const { return "abs"; }
};

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<ReturnAbs>);

TVM_REGISTER_OP("tirx.acos")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.acosh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.asin")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.asinh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.atan2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.cos")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.exp")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.fma")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.log")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.pow")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sin")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.tan")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchNumericalStableTanh);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

// extra dispatch
TVM_REGISTER_OP("tirx.erf").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchFastErf);

// warp-level primitives. Follows implementation in intrin_rule_metal.cc
TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                               DispatchWebGPUShuffle<WebGPUWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                               DispatchWebGPUShuffle<WebGPUWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                               DispatchWebGPUShuffle<WebGPUWarpIntrinsic>);

// Register low-level builtin ops.
TVM_REGISTER_OP("tir.webgpu.subgroup_shuffle")
    .set_num_inputs(2)
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane", "Expr", "The source thread id.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "subgroupShuffle")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tir.webgpu.subgroup_shuffle_up")
    .set_num_inputs(2)
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be added.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "subgroupShuffleUp")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tir.webgpu.subgroup_shuffle_down")
    .set_num_inputs(2)
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be subtracted.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "subgroupShuffleDown")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
