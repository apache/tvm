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
 * \file intrin_rule_rocm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <llvm/IR/Intrinsics.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <sstream>

#include "../intrin_rule.h"
#include "intrin_rule_llvm.h"

namespace tvm {
namespace codegen {

inline PrimExpr DispatchPureExternOCML(const PrimExpr& e) {
  // NOTE: OCML dispatch fails to work properly with vectorization, and thus should be used with
  // extreme caution.
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);

  const OpNode* op = call->op.as<OpNode>();
  ICHECK(op != nullptr);
  std::string name = op->name;
  ICHECK_EQ(name.substr(0, 4), "tir.");

  std::ostringstream intrinsic_name;
  intrinsic_name << "__ocml_" << name.substr(4) << "_f" << call->dtype.bits();

  Array<PrimExpr> new_args = {StringImm(intrinsic_name.str())};
  for (auto arg : call->args) {
    new_args.push_back(arg);
  }

  return Call(call->dtype, builtin::call_pure_extern(), new_args);
}

inline PrimExpr DispatchShuffle(const PrimExpr& e) {
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  PrimExpr var = call->args[1];
  ICHECK_EQ(var.dtype().bits(), 32);

  // get own lane in self (__lane_id)
  PrimExpr minus_one = tir::make_const(DataType::Int(32), -1);
  PrimExpr zero = tir::make_zero(DataType::Int(32));
  PrimExpr lo = Call(DataType::Int(32), builtin::call_pure_extern(),
                     {StringImm("llvm.amdgcn.mbcnt.lo"), minus_one, zero});
  PrimExpr self = Call(DataType::Int(32), builtin::call_pure_extern(),
                       {StringImm("llvm.amdgcn.mbcnt.hi"), minus_one, lo});

  // compute lane to get from
  PrimExpr width = call->args[3];
  PrimExpr index;
  if (call->op.same_as(builtin::tvm_warp_shuffle())) {
    PrimExpr src_lane = call->args[2];
    index = src_lane + (self & ~(width - 1));
  } else if (call->op.same_as(builtin::tvm_warp_shuffle_up())) {
    PrimExpr delta = call->args[2];
    index = self - delta;
    index = Select(index < (self & ~(width - 1)), self, index);
  } else {
    ICHECK(call->op.same_as(builtin::tvm_warp_shuffle_down()));
    PrimExpr delta = call->args[2];
    index = self + delta;
    index = Select((self & (width - 1)) + delta >= width, self, index);
  }
  // reinterprete var as int32
  bool is_int32 = var.dtype().is_int() && var.dtype().bits() == 32;
  PrimExpr source = is_int32 ? var : reinterpret(DataType::Int(32), var);
  PrimExpr res = Call(DataType::Int(32), builtin::call_pure_extern(),
                      {StringImm("llvm.amdgcn.ds.bpermute"), index << 2, source});
  if (!is_int32) {
    res = reinterpret(var.dtype(), res);
  }
  return res;
}

namespace llvm {
using tir::FLowerIntrinsic;

// dummy because we don't have the activemask
TVM_REGISTER_OP("tir.tvm_warp_activemask")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", [](const PrimExpr& e) -> PrimExpr {
      PrimExpr zero = tir::make_zero(DataType::Int(32));
      return zero;
    });

TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchShuffle);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchShuffle);

TVM_REGISTER_OP("tir.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchShuffle);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>(
    "rocm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::exp2, 1>);

TVM_REGISTER_OP("tir.fma").set_attr<FLowerIntrinsic>(
    "rocm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>(
    "rocm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::log2, 1>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::log10, 1>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>(
    "rocm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 2>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>(
    "rocm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>(
    "rocm.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                               ::tvm::codegen::intrin::DispatchNumericalStableTanh);

TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                                     ::tvm::codegen::intrin::DispatchFastErf);

// TVM_REGISTER_OP("tir.tan").set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
//                                                      DispatchPureExternOCML);

// TVM_REGISTER_OP("tir.cosh")
//     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchPureExternOCML);

// TVM_REGISTER_OP("tir.sinh")
//     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchPureExternOCML);

// TVM_REGISTER_OP("tir.atan")
//     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchPureExternOCML);

// TVM_REGISTER_OP("tir.exp10")
//     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
//                                DispatchLLVMPureIntrin<::llvm::Intrinsic::exp10, 1>);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
