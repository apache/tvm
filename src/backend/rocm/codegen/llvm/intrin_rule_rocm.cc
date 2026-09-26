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
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <sstream>

#include "../../../../target/intrin_rule.h"
#include "../../../../target/llvm/intrin_rule_llvm.h"

namespace tvm {
namespace codegen {
using namespace tvm::prim;

inline PrimExpr DispatchPureExternOCML(const PrimExpr& e) {
  // NOTE: OCML dispatch fails to work properly with vectorization, and thus should be used with
  // extreme caution.
  using namespace tirx;
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);

  const OpNode* op = call->op.as<OpNode>();
  TVM_FFI_ICHECK(op != nullptr);
  std::string name = op->name;
  TVM_FFI_ICHECK(name.substr(0, 5) == "tirx." || name == "prim.ceil" || name == "prim.log2")
      << "Unexpected intrinsic name: " << name;

  std::ostringstream intrinsic_name;
  PrimType call_ty = call->ty.as_or_throw<PrimType>();
  intrinsic_name << "__ocml_" << name.substr(5) << "_f" << call_ty.bits();

  ffi::Array<Expr> new_args = {StringImm(intrinsic_name.str())};
  for (PrimExpr arg : call->args.as_or_throw<ffi::Array<PrimExpr>>()) {
    new_args.push_back(arg);
  }

  return Call(call_ty, tirx::builtin::call_pure_extern(), new_args).as_or_throw<PrimExpr>();
}

inline PrimExpr DispatchShuffle(const PrimExpr& e) {
  using namespace tirx;
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  ffi::Array<PrimExpr> args = call->args.as_or_throw<ffi::Array<PrimExpr>>();
  PrimExpr var = args[1];
  PrimType var_ty = var.ty();
  TVM_FFI_ICHECK_EQ(var_ty.bits(), 32);

  // get own lane in self (__lane_id)
  PrimExpr minus_one = IntImm::Int32(-1);
  PrimExpr zero = IntImm::Int32(0);
  PrimType i32_ty = PrimType::Int(32);
  PrimExpr lo = Call(i32_ty, tirx::builtin::call_pure_extern(),
                     ffi::Array<Expr>{StringImm("llvm.amdgcn.mbcnt.lo"), minus_one, zero})
                    .as_or_throw<PrimExpr>();
  PrimExpr self = Call(i32_ty, tirx::builtin::call_pure_extern(),
                       ffi::Array<Expr>{StringImm("llvm.amdgcn.mbcnt.hi"), minus_one, lo})
                      .as_or_throw<PrimExpr>();

  // compute lane to get from
  PrimExpr width = args[3];
  PrimExpr index;
  if (call->op.same_as(tirx::builtin::tvm_warp_shuffle())) {
    PrimExpr src_lane = args[2];
    index = src_lane + (self & ~(width - 1));
  } else if (call->op.same_as(tirx::builtin::tvm_warp_shuffle_up())) {
    PrimExpr delta = args[2];
    index = self - delta;
    index = prim::Select(index < (self & ~(width - 1)), self, index);
  } else {
    TVM_FFI_ICHECK(call->op.same_as(tirx::builtin::tvm_warp_shuffle_down()));
    PrimExpr delta = args[2];
    index = self + delta;
    index = prim::Select((self & (width - 1)) + delta >= width, self, index);
  }
  // reinterprete var as int32
  bool is_int32 = var_ty.MatchesElementType(DLDataTypeCode::kDLInt, 32);
  PrimExpr source = is_int32 ? var : reinterpret(PrimType::Int(32), var);
  PrimExpr res = Call(i32_ty, tirx::builtin::call_pure_extern(),
                      ffi::Array<Expr>{StringImm("llvm.amdgcn.ds.bpermute"), index << 2, source})
                     .as_or_throw<PrimExpr>();
  if (!is_int32) {
    res = reinterpret(var_ty, res);
  }
  return res;
}

namespace llvm {
using tirx::FLowerIntrinsic;

void RegisterROCMIntrinRules() {
  // dummy because we don't have the activemask
  OpDef("tirx.tvm_warp_activemask")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", [](const PrimExpr& e) -> PrimExpr {
        PrimExpr zero = IntImm::Int32(0);
        return zero;
      });

  OpDef("tirx.tvm_warp_shuffle")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchShuffle);

  OpDef("tirx.tvm_warp_shuffle_up")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchShuffle);

  OpDef("tirx.tvm_warp_shuffle_down")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchShuffle);

  OpDef("tirx.floor")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

  OpDef("tirx.round")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("tirx.nearbyint")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("tirx.trunc")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

  OpDef("tirx.fabs")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

  OpDef("tirx.exp")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

  OpDef("tirx.exp2")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::exp2, 1>);

  OpDef("tirx.fma")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

  OpDef("tirx.log")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log2, 1>);

  OpDef("tirx.log10")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log10, 1>);

  OpDef("tirx.sqrt")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

  OpDef("tirx.pow")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 2>);

  OpDef("tirx.cos")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

  OpDef("tirx.sin")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

  OpDef("tirx.tanh")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                 ::tvm::codegen::intrin::DispatchNumericalStableTanh);

  OpDef("tirx.erf")
      .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", ::tvm::codegen::intrin::DispatchFastErf);

  // OpDef("tirx.tan").set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
  //                                                      DispatchPureExternOCML);

  // OpDef("tirx.cosh")
  //     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchPureExternOCML);

  // OpDef("tirx.sinh")
  //     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchPureExternOCML);

  // OpDef("tirx.atan")
  //     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic", DispatchPureExternOCML);

  // OpDef("tirx.exp10")
  //     .set_attr<FLowerIntrinsic>("rocm.FLowerIntrinsic",
  //                                DispatchLLVMPureIntrin<::llvm::Intrinsic::exp10, 1>);
}

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
