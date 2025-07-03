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
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include "intrin_rule_llvm.h"

#include <llvm/IR/Intrinsics.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <limits>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace llvm {
namespace intrin {
using tir::FLowerIntrinsic;

TVM_REGISTER_OP("tir.prefetch")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 4>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>(
    "llvm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::exp2, 1>);

TVM_REGISTER_OP("tir.fma").set_attr<FLowerIntrinsic>(
    "llvm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>(
    "llvm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::log2, 1>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::log10, 1>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>(
    "llvm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 2>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>(
    "llvm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>(
    "llvm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               ::tvm::codegen::intrin::DispatchNumericalStableTanh);
}  // namespace intrin

namespace legalize {
using tir::FLegalize;

TVM_REGISTER_OP("tir.exp10")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      using tir::make_zero;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
      PrimExpr ln10 = make_const(x.dtype(), 2.302585093);
      PrimExpr ret = exp(x * ln10);
      return ret;
    });

TVM_REGISTER_OP("tir.tan").set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  const PrimExpr& x = call->args[0];
  PrimExpr tan_x = sin(x) / cos(x);
  return tan_x;
});

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      using tir::make_zero;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
      PrimExpr two = make_const(x.dtype(), 2);
      PrimExpr neg_one = make_const(x.dtype(), -1);
      PrimExpr exp_negx = exp(neg_one * x);
      PrimExpr exp_posx = exp(x);
      PrimExpr ret = (exp_posx + exp_negx) / two;
      return ret;
    });

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      using tir::make_zero;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
      PrimExpr two = make_const(x.dtype(), 2);
      PrimExpr neg_one = make_const(x.dtype(), -1);
      PrimExpr exp_negx = exp(neg_one * x);
      PrimExpr exp_posx = exp(x);
      PrimExpr ret = (exp_posx - exp_negx) / two;
      return ret;
    });

TVM_REGISTER_OP("tir.asin")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
      PrimExpr x2 = x * x;
      PrimExpr term1 = x;
      PrimExpr term3 = term1 * x2 / make_const(x.dtype(), 6);
      PrimExpr term5 = term3 * x2 * make_const(x.dtype(), 9) / make_const(x.dtype(), 40);
      PrimExpr term7 = term5 * x2 * make_const(x.dtype(), 25) / make_const(x.dtype(), 112);
      PrimExpr term9 = term7 * x2 * make_const(x.dtype(), 1225) / make_const(x.dtype(), 3456);
      PrimExpr term11 = term9 * x2 * make_const(x.dtype(), 3969) / make_const(x.dtype(), 28160);
      PrimExpr series = term1 + term3 + term5 + term7 + term9 + term11;
      /* --- domain limit check --- */
      PrimExpr lower = make_const(x.dtype(), -1.0);
      PrimExpr upper = make_const(x.dtype(), 1.0);
      PrimExpr out_range = tir::Or(x<lower, x> upper);
      // Use a quiet NaN constant
      PrimExpr nan_const = make_const(x.dtype(), std::numeric_limits<double>::quiet_NaN());
      // select: if out of [-1,1] → NaN, else → series
      return tir::Select(out_range, nan_const, series);
    });

TVM_REGISTER_OP("tir.acos")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr) << "Invalid call node in acos legalization";
      const PrimExpr& x = call->args[0];
      PrimExpr half_pi = make_const(x.dtype(), M_PI / 2);
      PrimExpr asin_x = asin(x);
      return half_pi - asin_x;
    });

TVM_REGISTER_OP("tir.atan")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr) << "Invalid call node in atan legalization";
      const PrimExpr& x = call->args[0];
      PrimExpr one = make_const(x.dtype(), 1.0);
      PrimExpr denom = sqrt(x * x + one);
      return asin(x / denom);
    });

TVM_REGISTER_OP("tir.asinh")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr) << "Invalid call node in asinh legalization";
      const PrimExpr& x = call->args[0];
      PrimExpr one = make_const(x.dtype(), 1.0);
      PrimExpr sqrt_val = sqrt(x * x + one);
      return log(x + sqrt_val);
    });

TVM_REGISTER_OP("tir.acosh")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr) << "Invalid call node in acosh legalization";
      const PrimExpr& x = call->args[0];
      PrimExpr one = make_const(x.dtype(), 1.0);
      PrimExpr sqrt_val = sqrt(x * x - one);
      return log(x + sqrt_val);
    });

TVM_REGISTER_OP("tir.atanh")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tir::make_const;
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr) << "Invalid call node in atanh legalization";
      const PrimExpr& x = call->args[0];
      PrimExpr one = make_const(x.dtype(), 1.0);
      return (log(one + x) - log(one - x)) * make_const(x.dtype(), 0.5);
    });

TVM_REGISTER_OP("tir.erf").set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
  using tir::make_const;
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr) << "Invalid call node in erf legalization";
  const PrimExpr& x = call->args[0];
  PrimExpr abs_x = tvm::abs(x);
  PrimExpr t = make_const(x.dtype(), 1.0) /
               (make_const(x.dtype(), 1.0) + make_const(x.dtype(), 0.3275911) * abs_x);
  PrimExpr a1 = make_const(x.dtype(), 0.254829592);
  PrimExpr a2 = make_const(x.dtype(), -0.284496736);
  PrimExpr a3 = make_const(x.dtype(), 1.421413741);
  PrimExpr a4 = make_const(x.dtype(), -1.453152027);
  PrimExpr a5 = make_const(x.dtype(), 1.061405429);
  PrimExpr poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t);
  PrimExpr approx = make_const(x.dtype(), 1.0) - poly * exp(-abs_x * abs_x);
  return tvm::tir::Select(x < 0, -approx, approx);
});

TVM_REGISTER_OP("tir.clz").set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1);
  Array<PrimExpr> cargs;
  cargs.push_back(IntImm(DataType::UInt(32), ::llvm::Intrinsic::ctlz));
  cargs.push_back(IntImm(DataType::UInt(32), 2));
  cargs.push_back(call->args[0]);
  cargs.push_back(IntImm(DataType::Int(1), 1));  // is_zero_undef
  // LLVM requires that the return type must match the first argument type
  auto clz = tir::Call(call->args[0]->dtype, tir::builtin::call_llvm_intrin(), cargs);
  return cast(call->dtype, clz);
});

}  // namespace legalize
}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
