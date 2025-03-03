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
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

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
