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
 * \file intrin_rule_default.cc
 * \brief Default intrinsic rules.
 */
#include "intrin_rule.h"

#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace codegen {
using namespace tvm::prim;

namespace intrin {
using tirx::FLowerIntrinsic;

// `prim.round` is ties-to-even (see include/tvm/ir/prim/op.h), and constant
// folding implements it with std::nearbyint. The C library's round()/roundf()
// is ties-AWAY-from-zero, so lowering through FloatSuffix would disagree with
// the folder and with every other backend. Rename to nearbyint before the
// float suffix is applied, as the CUDA rule already does.
//
// Like nearbyint in general, this honours the current floating-point
// environment: it is ties-to-even under the default FE_TONEAREST. The only
// other backend a host fesetround() can reach is llvm, which lowers to
// llvm.nearbyint (also mode-sensitive; llvm.roundeven is the
// mode-independent one). Every other backend that registers prim.round --
// cuda, nvptx, rocm, hexagon, metal, opencl, vulkan, webgpu -- emits code
// for a separate device whose rounding mode is fixed at RNE, so the
// host's mode cannot reach it, whichever intrinsic the rule names.
struct FloatSuffixTiesToEven {
  std::string operator()(const PrimType& ty, std::string name) const {
    if (name == "round") name = "nearbyint";
    return FloatSuffix()(ty, name);
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.exp")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.erf")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.log")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.log10")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.log1p")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.tanh")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.tan")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.trunc")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.atan")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.atanh")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.atan2")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.cos")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.acos")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.cosh")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.acosh")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.sin")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.asin")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.sinh")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.asinh")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.hypot")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.nextafter")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.copysign")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.ldexp")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.sqrt")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.floor")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.round")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic",
                                 DispatchPureExtern<FloatSuffixTiesToEven>);

  OpDef("prim.nearbyint")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);

  OpDef("prim.pow")
      .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", DispatchPureExtern<FloatSuffix>);
}

PrimExpr DispatchFastErf(const PrimExpr& e) {
  DLOG(WARNING) << "fast_erf will be used instead of erf";
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 1);
  PrimExpr arg = call->args[0].as_or_throw<PrimExpr>();
  PrimType arg_ty = arg.ty();
  int bits = arg_ty.bits();
  if (arg_ty.code() == DLDataTypeCode::kDLFloat && (bits == 16 || bits == 32)) {
    return tirx::fast_erf_float_expr(arg, bits);
  } else {
    TVM_FFI_THROW(InternalError) << "Unsupported type in Metal fast_erf";
  }
}

PrimExpr DispatchNumericalStableTanh(const PrimExpr& e) {
  using tvm::prim::MakeConst;
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
  PrimType x_ty = x.ty();
  PrimExpr one = MakeConst(x_ty, 1);
  PrimExpr two = MakeConst(x_ty, 2);
  PrimExpr neg_two = MakeConst(x_ty, -2);

  PrimExpr exp_neg2x = exp(neg_two * x);
  PrimExpr exp_pos2x = exp(two * x);

  PrimExpr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
  PrimExpr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
  // MakeConst can handle both vector and scalar types.
  return prim::Select(x >= MakeConst(x_ty, 0), tanh_pos, tanh_neg);
}

}  // namespace intrin

namespace legalize {

using namespace tirx;

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("prim.rsqrt")
      .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    PrimExpr arg = call->args[0].as_or_throw<PrimExpr>();
    auto one = MakeConst(arg.ty(), 1);
    return one / sqrt(arg);
  });

  OpDef("prim.sigmoid")
      .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    PrimExpr arg = call->args[0].as_or_throw<PrimExpr>();
    auto one = MakeConst(arg.ty(), 1);
    return one / (one + exp(-arg));
  });

  OpDef("tirx.isfinite")
      .signature(sig::arg("x", "The input value."))
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
        const CallNode* call = e.as<CallNode>();
        TVM_FFI_ICHECK(call != nullptr);
        return isfinite(call->args[0].as_or_throw<PrimExpr>());
      });

  OpDef("tirx.isinf")
      .signature(sig::arg("x", "The input value."))
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<FLegalize>("default.FLegalize", [](const PrimExpr& e) -> PrimExpr {
        const CallNode* call = e.as<CallNode>();
        TVM_FFI_ICHECK(call != nullptr);
        return isinf(call->args[0].as_or_throw<PrimExpr>());
      });
  // clang-format on
}

}  // namespace legalize
}  // namespace codegen
}  // namespace tvm
