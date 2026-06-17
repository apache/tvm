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
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <limits>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace llvm {
namespace intrin {
using tirx::FLowerIntrinsic;

TVM_REGISTER_OP("tirx.prefetch")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 4>);

TVM_REGISTER_OP("tirx.exp")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::exp2, 1>);

TVM_REGISTER_OP("tirx.fma")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

TVM_REGISTER_OP("tirx.log")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::log2, 1>);

TVM_REGISTER_OP("tirx.log10")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::log10, 1>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_OP("tirx.pow")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 2>);

TVM_REGISTER_OP("tirx.popcount")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

TVM_REGISTER_OP("tirx.cos")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

TVM_REGISTER_OP("tirx.sin")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("llvm.FLowerIntrinsic",
                               ::tvm::codegen::intrin::DispatchNumericalStableTanh);
}  // namespace intrin

namespace legalize {
using tirx::FLegalize;

TVM_REGISTER_OP("tirx.exp10")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tirx::MakeConst;
      const tirx::CallNode* call = e.as<tirx::CallNode>();
      TVM_FFI_ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
      PrimExpr ln10 = MakeConst(x.dtype(), 2.302585093);
      PrimExpr ret = exp(x * ln10);
      return ret;
    });

TVM_REGISTER_OP("tirx.tan")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const tirx::CallNode* call = e.as<tirx::CallNode>();
      TVM_FFI_ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
      PrimExpr tan_x = sin(x) / cos(x);
      return tan_x;
    });

TVM_REGISTER_OP("tirx.asin")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using namespace intrin;
      const tirx::CallNode* call = e.as<tirx::CallNode>();
      TVM_FFI_ICHECK(call != nullptr);
      return ::tvm::codegen::intrin::DispatchPureExtern<::tvm::codegen::intrin::FloatSuffix>(e);
    });

TVM_REGISTER_OP("tirx.acos")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using namespace intrin;
      const tirx::CallNode* call = e.as<tirx::CallNode>();
      TVM_FFI_ICHECK(call != nullptr) << "Invalid call node in acos legalization";
      return ::tvm::codegen::intrin::DispatchPureExtern<::tvm::codegen::intrin::FloatSuffix>(e);
    });

TVM_REGISTER_OP("tirx.atanh")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      using tirx::MakeConst;
      const tirx::CallNode* call = e.as<tirx::CallNode>();
      TVM_FFI_ICHECK(call != nullptr) << "Invalid call node in atanh legalization";
      const PrimExpr& x = call->args[0];
      PrimExpr one = MakeConst(x.dtype(), 1.0);
      return (log(one + x) - log(one - x)) * MakeConst(x.dtype(), 0.5);
    });

TVM_REGISTER_OP("tirx.clz")
    .set_attr<FLegalize>("llvm.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const tirx::CallNode* call = e.as<tirx::CallNode>();
      TVM_FFI_ICHECK(call != nullptr);
      TVM_FFI_ICHECK_EQ(call->args.size(), 1);
      ffi::Array<PrimExpr> cargs;
      cargs.push_back(IntImm(DataType::UInt(32), ::llvm::Intrinsic::ctlz));
      cargs.push_back(call->args[0]);
      cargs.push_back(IntImm(DataType::Int(1), 1));  // is_zero_undef
      // LLVM requires that the return type must match the first argument type
      auto clz = tirx::Call(call->args[0]->dtype, tirx::builtin::call_llvm_intrin(), cargs);
      return cast(call->dtype, clz);
    });

}  // namespace legalize
}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
