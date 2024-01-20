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
 * \file intrin_rule_metal.cc
 * \brief Metal intrinsic rules.
 */
#include <tvm/tir/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
using tir::FLowerIntrinsic;

struct MetalWarpIntrinsic {
  const Op operator()(DataType t, const Op& orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tir.metal.simd_shuffle");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tir.metal.simd_shuffle_up");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tir.metal.simd_shuffle_down");
    }
  }
};

template <typename T>
static PrimExpr DispatchMetalShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  Array<PrimExpr> metal_args{{call->args[1], call->args[2]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), metal_args);
}

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp10")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchNumericalStableTanh);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fmod")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchFastErf);

TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchMetalShuffle<MetalWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchMetalShuffle<MetalWarpIntrinsic>);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("metal.FLowerIntrinsic", DispatchMetalShuffle<MetalWarpIntrinsic>);

// Register low-level builtin ops.
TVM_REGISTER_OP("tir.metal.simd_shuffle")
    .set_num_inputs(2)
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane", "Expr", "The source thread id.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "simd_shuffle")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tir.metal.simd_shuffle_up")
    .set_num_inputs(2)
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be added.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "simd_shuffle_up")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tir.metal.simd_shuffle_down")
    .set_num_inputs(2)
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be subtracted.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "simd_shuffle_down")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
