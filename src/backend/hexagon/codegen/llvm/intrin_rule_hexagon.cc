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

#ifdef TVM_LLVM_VERSION

#include <llvm/IR/Intrinsics.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "../../../../target/llvm/intrin_rule_llvm.h"

namespace tvm {
namespace codegen {
using namespace tvm::prim;

namespace llvm {
using tirx::FLowerIntrinsic;

std::string tvm_qhl_ahf_ceil = "tvm_vect_qhmath_hvx_ceil_ahf";
std::string tvm_qhl_ahf_cos = "tvm_vect_qhmath_hvx_cos_ahf";
std::string tvm_qhl_ahf_exp = "tvm_vect_qhmath_hvx_exp_ahf";
std::string tvm_qhl_ahf_floor = "tvm_vect_qhmath_hvx_floor_ahf";
std::string tvm_qhl_ahf_sin = "tvm_vect_qhmath_hvx_sin_ahf";
std::string tvm_qhl_ahf_pow = "tvm_vect_qhmath_hvx_pow_ahf";
std::string tvm_qhl_ahf_sqrt = "tvm_vect_qhmath_hvx_sqrt_ahf";

inline PrimExpr TVMExternCall(const CallNode* call, const std::string& fname) {
  ffi::Array<Expr> new_args = {StringImm(fname)};
  for (PrimExpr arg : call->args.as_or_throw<ffi::Array<PrimExpr>>()) {
    new_args.push_back(arg);
  }
  return Call(call->ty.as_or_throw<PrimType>(), tirx::builtin::call_pure_extern(), new_args)
      .as_or_throw<PrimExpr>();
}

template <std::string& tvm_wrapper, unsigned id, int num_sign>
inline PrimExpr DispatchTVMQHLWrapperFp16(const PrimExpr& e) {
  using namespace tirx;
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  ffi::Array<PrimExpr> new_args;
#if ENABLE_QHL
  // Check target for qfloat enablement
  const auto f = tvm::ffi::Function::GetGlobal("target.TargetCurrent");
  TVM_FFI_ICHECK(f.has_value()) << "target.TargetCurrent is not registered";
  const auto ret = (*f)(true);
  bool useqhl = true;
  if (auto opt_target = ret.as<Target>()) {
    const std::string tstring = opt_target.value()->str();
    useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
  }

  // Enable QHL library for FP16 data type
  PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
  PrimType x_ty = x.ty();
  if (x_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16) &&
      (x_ty.IsFixedLengthVector() || x_ty.IsScalableVector()) && useqhl) {
    return TVMExternCall(call, tvm_wrapper);
  }
#endif
  new_args.push_back(IntImm(PrimType::UInt(32), id));
  new_args.push_back(IntImm(PrimType::UInt(32), num_sign));
  ffi::Array<PrimExpr> call_args = call->args.as_or_throw<ffi::Array<PrimExpr>>();
  new_args.insert(new_args.end(), call_args.begin(), call_args.end());
  return Call(call->ty.as_or_throw<PrimType>(), tirx::builtin::call_llvm_pure_intrin(), new_args)
      .as_or_throw<PrimExpr>();
}

void RegisterHexagonIntrinRules() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  // clang-format off
  OpDef("tirx.fma")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

  OpDef("tirx.log")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

  OpDef("tirx.trunc")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

  OpDef("tirx.fabs")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

  OpDef("tirx.round")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("tirx.ctpop")
      .arg<Expr>("x", "")
      .set_attr<tirx::TIRxOpCategory>("TIRxOpCategory", ffi::String("builtin"))
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

  OpDef("tirx.tanh")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](const PrimExpr& e) {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
    PrimType x_ty = x.ty();

#if ENABLE_QHL
    // Check target for qfloat enablement
    const auto f = tvm::ffi::Function::GetGlobal("target.TargetCurrent");
    TVM_FFI_ICHECK(f.has_value()) << "target.TargetCurrent is not registered";
    const auto ret = (*f)(true);
    bool useqhl = true;
    if (auto opt_target = ret.as<Target>()) {
      const std::string tstring = opt_target.value()->str();
      useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
    }

    // Enable QHL library for FP16 data type
    if (x_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16) &&
        (x_ty.IsFixedLengthVector() || x_ty.IsScalableVector()) && useqhl) {
      std::string tvm_wrapper("tvm_vect_qhmath_hvx_tanh_ahf");
      return TVMExternCall(call, tvm_wrapper);
    }
#endif
    PrimExpr one = tvm::prim::MakeConst(x_ty, 1);
    PrimExpr two = tvm::prim::MakeConst(x_ty, 2);
    PrimExpr neg_two = tvm::prim::MakeConst(x_ty, -2);

    PrimExpr exp_neg2x = exp(neg_two * x);
    PrimExpr exp_pos2x = exp(two * x);

    PrimExpr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
    PrimExpr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
    // MakeConst can handle both vector and scalar types.
    PrimExpr tanh_x = prim::Select(x >= tvm::prim::MakeConst(x_ty, 0), tanh_pos, tanh_neg);
    return tanh_x;
  });

  OpDef("tirx.tan")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](const PrimExpr& e) {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
    PrimType x_ty = x.ty();
#if ENABLE_QHL
    // Check target for qfloat enablement
    const auto f = tvm::ffi::Function::GetGlobal("target.TargetCurrent");
    TVM_FFI_ICHECK(f.has_value()) << "target.TargetCurrent is not registered";
    const auto ret = (*f)(true);
    bool useqhl = true;
    if (auto opt_target = ret.as<Target>()) {
      const std::string tstring = opt_target.value()->str();
      useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
    }

    // Enable QHL library for FP16 data type
    if (x_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16) &&
        (x_ty.IsFixedLengthVector() || x_ty.IsScalableVector()) && useqhl) {
      std::string tvm_wrapper("tvm_vect_qhmath_hvx_tan_ahf");
      return TVMExternCall(call, tvm_wrapper);
    }
#endif
    PrimExpr tan_x = sin(x) / cos(x);
    return tan_x;
  });

  OpDef("tirx.nearbyint")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                                 DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

  OpDef("tirx.sigmoid")
      .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](const PrimExpr& e) {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    PrimExpr x = call->args[0].as_or_throw<PrimExpr>();
    PrimType x_ty = x.ty();
#if ENABLE_QHL
    // Check target for qfloat enablement
    const auto f = tvm::ffi::Function::GetGlobal("target.TargetCurrent");
    TVM_FFI_ICHECK(f.has_value()) << "target.TargetCurrent is not registered";
    const auto ret = (*f)(true);
    bool useqhl = true;
    if (auto opt_target = ret.as<Target>()) {
      const std::string tstring = opt_target.value()->str();
      useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
    }

    PrimExpr MinBound = tvm::prim::MakeConst(x_ty, -8);
    PrimExpr MaxBound = tvm::prim::MakeConst(x_ty, 8);
    const PrimExpr v1 = prim::Max(x, MinBound);
    const PrimExpr v2 = prim::Min(v1, MaxBound);

    ffi::Array<tvm::PrimExpr> new_args = {v2};
    const Call new_call = Call(call->ty.as_or_throw<PrimType>(), call->op, new_args);

    // Enable QHL library for FP16 data type
    if (x_ty.MatchesElementType(DLDataTypeCode::kDLFloat, 16) &&
        (x_ty.IsFixedLengthVector() || x_ty.IsScalableVector()) && useqhl) {
      std::string tvm_wrapper("tvm_vect_qhmath_hvx_sigmoid_ahf");
      return TVMExternCall(new_call.get(), tvm_wrapper);
    }
#endif
    PrimExpr one = tvm::prim::MakeConst(x_ty, 1);
    return one / (one + exp(-x));
  });

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>(
          "hexagon.FLowerIntrinsic",
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_ceil, ::llvm::Intrinsic::ceil, 1>);

  OpDef("tirx.cos")
      .set_attr<FLowerIntrinsic>(
          "hexagon.FLowerIntrinsic",
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_cos, ::llvm::Intrinsic::cos, 1>);

  OpDef("tirx.exp")
      .set_attr<FLowerIntrinsic>(
          "hexagon.FLowerIntrinsic",
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_exp, ::llvm::Intrinsic::exp, 1>);

  OpDef("tirx.floor")
      .set_attr<FLowerIntrinsic>(
          "hexagon.FLowerIntrinsic",
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_floor, ::llvm::Intrinsic::floor, 1>);

  OpDef("tirx.sin")
      .set_attr<FLowerIntrinsic>(
          "hexagon.FLowerIntrinsic",
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_sin, ::llvm::Intrinsic::sin, 1>);

  OpDef("tirx.pow")
      .set_attr<FLowerIntrinsic>(
          "hexagon.FLowerIntrinsic",
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_pow, ::llvm::Intrinsic::pow, 2>);

  OpDef("tirx.sqrt")
      .set_attr<FLowerIntrinsic>(
          "hexagon.FLowerIntrinsic",
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_sqrt, ::llvm::Intrinsic::sqrt, 1>);
  // clang-format on
}

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
