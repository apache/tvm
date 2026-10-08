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
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op_attr_types.h>

#include <limits>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
using namespace tvm::prim;

namespace llvm {
namespace intrin {
using tirx::FLowerIntrinsic;

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tirx.prefetch")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 4>);

  OpDef("prim.exp")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

  OpDef("prim.exp2")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::exp2, 1>);

  OpDef("prim.fma")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

  OpDef("prim.log")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log2, 1>);

  OpDef("prim.log10")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log10, 1>);

  OpDef("prim.sqrt")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

  OpDef("prim.floor")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

  OpDef("prim.trunc")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

  OpDef("prim.fabs")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

  OpDef("prim.round")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("prim.nearbyint")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("prim.pow")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 2>);

  OpDef("prim.popcount")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

  OpDef("prim.cos")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

  OpDef("prim.sin")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

  OpDef("prim.tanh")
      .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                                 ::tvm::codegen::intrin::DispatchNumericalStableTanh);
}
}  // namespace intrin

namespace legalize {
using tirx::FLegalize;

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("prim.exp10")
      .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    using tvm::prim::MakeConst;
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
    PrimExpr ln10 = MakeConst(x.ty(), 2.302585093);
    PrimExpr ret = exp(x * ln10);
    return ret;
  });

  OpDef("prim.tan")
      .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
    PrimExpr tan_x = sin(x) / cos(x);
    return tan_x;
  });

  OpDef("prim.asin")
      .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    using namespace intrin;
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    return ::tvm::codegen::intrin::DispatchPureExtern<::tvm::codegen::intrin::FloatSuffix>(e);
  });

  OpDef("prim.acos")
      .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    using namespace intrin;
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr) << "Invalid call node in acos legalization";
    return ::tvm::codegen::intrin::DispatchPureExtern<::tvm::codegen::intrin::FloatSuffix>(e);
  });

  OpDef("prim.atanh")
      .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    using tvm::prim::MakeConst;
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr) << "Invalid call node in atanh legalization";
    PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
    PrimType x_ty = x.ty();
    PrimExpr one = MakeConst(x_ty, 1.0);
    return (log(one + x) - log(one - x)) * MakeConst(x_ty, 0.5);
  });

  OpDef("prim.clz")
      .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    TVM_FFI_ICHECK_EQ(call->args.size(), 1);
    ffi::Array<PrimExpr> cargs;
    cargs.push_back(IntImm(PrimType::UInt(32), ::llvm::Intrinsic::ctlz));
    cargs.push_back(call->args[0].as_or_throw<PrimExpr>());
    cargs.push_back(IntImm(PrimType::Int(1), 1));  // is_zero_undef
    // LLVM requires that the return type must match the first argument type
    auto clz =
        Call(call->args[0]->ty.as_or_throw<PrimType>(), tirx::call_llvm_intrin_op(), cargs)
            .as_or_throw<PrimExpr>();
    return cast(call->ty.as_or_throw<PrimType>(), clz);
  });
  // clang-format on
}

}  // namespace legalize
}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
