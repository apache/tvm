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
 * \file intrin_rule_cuda.cc
 * \brief CUDA intrinsic rules.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "../../../target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, CUDA fast math.
using tirx::FLowerIntrinsic;

struct CUDAMath {
  std::string operator()(const PrimType& ty, std::string name) const {
    if (ty.MatchesCode(DLDataTypeCode::kDLFloat)) {
      switch (ty.bits()) {
        case 64:
          // Use nearbyint (ties-to-even) for round to match constant-folding semantics.
          if (name == "round") return "nearbyint";
          return name;
        case 32:
          if (name == "round") return "nearbyintf";
          return name + 'f';
        case 16: {
          if (name == "fabs") {
            return "__habs";
          } else if (name == "round") {
            return "hrint";
          } else {
            return "h" + name;
          }
        }
        default:
          return "";
      }
    } else if (ty.MatchesCode(DLDataTypeCode::kDLBfloat) && ty.bits() == 16) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (ty.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt)) {
      switch (ty.bits()) {
        case 32:
          return "__" + name;
        case 64:
          return "__" + name + "ll";
        default:
          return "";
      }
    }
    return "";
  }
};

struct CUDAFastMath : public CUDAMath {
  std::string operator()(const PrimType& ty, std::string name) const {
    if (ty.MatchesCode(DLDataTypeCode::kDLFloat) && ty.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return CUDAMath::operator()(ty, name);
    }
    return "";
  }
};

struct CUDAFastMathTan : public CUDAMath {
  std::string operator()(const PrimType& ty, std::string name) const {
    if (ty.MatchesCode(DLDataTypeCode::kDLFloat)) {
      switch (ty.bits()) {
        case 64:
          return name;
        // `__tanf` seems to produce some values too deviant from numpy tan version.
        // So, let's use just `tanf` instead.
        case 32:
          return name + 'f';
        case 16:
          return 'h' + name;
        default:
          return "";
      }
    }
    return "";
  }
};

struct CUDAPopcount {
  std::string operator()(const PrimType& ty, std::string name) const {
    if (ty.MatchesCode(DLDataTypeCode::kDLUInt)) {
      switch (ty.bits()) {
        case 32:
          return "__popc";
        case 64:
          return "__popcll";
        default:
          return "";
      }
    }
    return "";
  }
};

struct CUDAWarpIntrinsic {
  const Op operator()(const PrimType& ty, const Op& orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      static const Op cuda_shfl_sync_op = Op::Get("tirx.cuda.__shfl_sync");
      return cuda_shfl_sync_op;
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      static const Op cuda_shfl_up_sync_op = Op::Get("tirx.cuda.__shfl_up_sync");
      return cuda_shfl_up_sync_op;
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_down())) {
      static const Op cuda_shfl_down_sync_op = Op::Get("tirx.cuda.__shfl_down_sync");
      return cuda_shfl_down_sync_op;
    } else {
      TVM_FFI_ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_xor()));
      static const Op cuda_shfl_xor_sync_op = Op::Get("tirx.cuda.__shfl_xor_sync");
      return cuda_shfl_xor_sync_op;
    }
  }
};

static PrimExpr DispatchCUDAWarpActiveMask(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  static const Op cuda_active_mask_op = Op::Get("tirx.cuda.__activemask");
  ffi::Array<PrimExpr> args = call->args.as_or_throw<ffi::Array<PrimExpr>>();
  return Call(e.ty(), cuda_active_mask_op, args).as_or_throw<PrimExpr>();
}

template <typename T>
static PrimExpr DispatchCUDAShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  ffi::Array<PrimExpr> cuda_args{
      call->args[0].as_or_throw<PrimExpr>(), call->args[1].as_or_throw<PrimExpr>(),
      call->args[2].as_or_throw<PrimExpr>(), call->args[3].as_or_throw<PrimExpr>()};
  return Call(e.ty(), T()(e.ty(), call->op.as_or_throw<Op>()), cuda_args).as_or_throw<PrimExpr>();
}

void RegisterCudaIntrinRules() {
  // clang-format off
  OpDef("prim.clz")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                                 DispatchPureExtern<CUDAMath, /*dtype_from_arg=*/true>);

  OpDef("tirx.floor")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.trunc")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.fabs")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.round")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.nearbyint")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.exp")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.exp2")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.exp10")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.erf")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.log")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.log10")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.tan")
      // Now the fast math version of tan and the default version of tan are same.
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic",
                                 DispatchPureExtern<CUDAFastMathTan>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.cos")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.cosh")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.sin")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.sinh")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.atan")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.tanh")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.sqrt")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.rsqrt")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.pow")
      .set_attr<FLowerIntrinsic>("cuda.fastmath.FLowerIntrinsic", DispatchPureExtern<CUDAFastMath>)
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  OpDef("tirx.popcount")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAPopcount>);

  OpDef("tirx.tvm_warp_shuffle")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAShuffle<CUDAWarpIntrinsic>);

  OpDef("tirx.tvm_warp_shuffle_up")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAShuffle<CUDAWarpIntrinsic>);

  OpDef("tirx.tvm_warp_shuffle_down")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAShuffle<CUDAWarpIntrinsic>);

  OpDef("tirx.tvm_warp_shuffle_xor")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAShuffle<CUDAWarpIntrinsic>);

  OpDef("tirx.tvm_warp_activemask")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAWarpActiveMask);

  OpDef("tirx.fmod")
      .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchPureExtern<CUDAMath>);

  // Register low-level CUDA device intrinsics.
  // TODO(tvm-team): consider make CUDA its own subfolder and create a file for low-level builtins.
  OpDef("tirx.cuda.__shfl_sync")
      .arg("mask", "The thread mask.")
      .arg("var", "The variable to sync.")
      .arg("lane", "The source thread id.")
      .arg("width", "The warp thread width, must be a power of 2.")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.__shfl_sync"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_sync")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<bool>("cuda.need_warp_shuffle", true);

  OpDef("tirx.cuda.__shfl_up_sync")
      .arg("mask", "The thread mask.")
      .arg("var", "The variable to sync.")
      .arg("delta", "The source lane id offset to be added.")
      .arg("width", "The warp thread width, must be a power of 2.")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.__shfl_up_sync"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_up_sync")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<bool>("cuda.need_warp_shuffle", true);

  OpDef("tirx.cuda.__shfl_down_sync")
      .arg("mask", "The thread mask.")
      .arg("var", "The variable to sync.")
      .arg("delta", "The source lane id offset to be subtracted.")
      .arg("width", "The warp thread width, must be a power of 2.")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName",
                                          ffi::String("cuda.__shfl_down_sync"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_down_sync")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<bool>("cuda.need_warp_shuffle", true);

  OpDef("tirx.cuda.__shfl_xor_sync")
      .arg("mask", "The thread mask.")
      .arg("var", "The variable to sync.")
      .arg("lane_mask", "The lane mask.")
      .arg("width", "The warp thread width, must be a power of 2.")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.__shfl_xor_sync"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_xor_sync")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kOpaque))
      .set_attr<bool>("cuda.need_warp_shuffle", true);

  OpDef("tirx.cuda.__activemask")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("device_intrin"))
      .set_attr<tirx::TDeviceIntrinsicNamespace>("TDeviceIntrinsicNamespace", ffi::String("cuda"))
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", ffi::String("cuda.__activemask"))
      .set_attr<TGlobalSymbol>("TGlobalSymbol", "__activemask")
      .set_attr<TCallEffectKind>("TCallEffectKind", static_cast<int64_t>(CallEffectKind::kPure))
      .set_attr<bool>("cuda.need_warp_shuffle", true);
  // clang-format on
}

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaIntrinRules(); }

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
