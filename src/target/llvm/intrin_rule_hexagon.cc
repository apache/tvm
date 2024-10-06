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
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "intrin_rule_llvm.h"

#define TVM_REGISTER_QHL_OP_FP16(INTRIN_FUNC, WRAPPER_FUNC, NUM_SIGN)                          \
  std::string tvm_qhl_ahf_##INTRIN_FUNC = WRAPPER_FUNC;                                        \
  TVM_REGISTER_OP("tir." #INTRIN_FUNC)                                                         \
      .set_attr<FLowerIntrinsic>(                                                              \
          "hexagon.FLowerIntrinsic",                                                           \
          DispatchTVMQHLWrapperFp16<tvm_qhl_ahf_##INTRIN_FUNC, ::llvm::Intrinsic::INTRIN_FUNC, \
                                    NUM_SIGN>);

namespace tvm {
namespace codegen {
namespace llvm {
using tir::FLowerIntrinsic;

inline PrimExpr TVMExternCall(const tir::CallNode* call, const std::string& fname) {
  Array<PrimExpr> new_args = {tir::StringImm(fname)};
  for (PrimExpr arg : call->args) {
    new_args.push_back(arg);
  }
  return tir::Call(call->dtype, tir::builtin::call_pure_extern(), new_args);
}

template <std::string& tvm_wrapper, unsigned id, int num_sign>
inline PrimExpr DispatchTVMQHLWrapperFp16(const PrimExpr& e) {
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  Array<PrimExpr> new_args;
#if ENABLE_QHL
  // Check target for qfloat enablement
  const auto* f = tvm::runtime::Registry::Get("target.TargetCurrent");
  ICHECK(f != nullptr);
  const auto ret = (*f)(true);
  const Target t = ret.AsObjectRef<Target>();
  bool useqhl = true;
  if (t.defined()) {
    const std::string tstring = t->str();
    useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
  }

  // Enable QHL library for FP16 data type
  const PrimExpr& x = call->args[0];
  if (x->dtype.is_float16() && x->dtype.is_vector() && useqhl) {
    return TVMExternCall(call, tvm_wrapper);
  }
#endif
  new_args.push_back(IntImm(DataType::UInt(32), id));
  new_args.push_back(IntImm(DataType::UInt(32), num_sign));
  new_args.insert(new_args.end(), call->args.begin(), call->args.end());
  return tir::Call(call->dtype, tir::builtin::call_llvm_pure_intrin(), new_args);
}

TVM_REGISTER_OP("tir.fma").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_OP("tir.ctpop")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);
TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](const PrimExpr& e) {
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];

#if ENABLE_QHL
      // Check target for qfloat enablement
      const auto* f = tvm::runtime::Registry::Get("target.TargetCurrent");
      ICHECK(f != nullptr);
      const auto ret = (*f)(true);
      const Target t = ret.AsObjectRef<Target>();
      bool useqhl = true;
      if (t.defined()) {
        const std::string tstring = t->str();
        useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
      }

      // Enable QHL library for FP16 data type
      if (x->dtype.is_float16() && x->dtype.is_vector() && useqhl) {
        std::string tvm_wrapper("tvm_vect_qhmath_hvx_tanh_ahf");
        return TVMExternCall(call, tvm_wrapper);
      }
#endif
      PrimExpr one = tir::make_const(x.dtype(), 1);
      PrimExpr two = tir::make_const(x.dtype(), 2);
      PrimExpr neg_two = tir::make_const(x.dtype(), -2);

      PrimExpr exp_neg2x = exp(neg_two * x);
      PrimExpr exp_pos2x = exp(two * x);

      PrimExpr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
      PrimExpr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
      PrimExpr tanh_x = tir::Select(x >= tir::make_zero(x.dtype()), tanh_pos, tanh_neg);
      return tanh_x;
    });

TVM_REGISTER_OP("tir.tan").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", [](const PrimExpr& e) {
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
#if ENABLE_QHL
      // Check target for qfloat enablement
      const auto* f = tvm::runtime::Registry::Get("target.TargetCurrent");
      ICHECK(f != nullptr);
      const auto ret = (*f)(true);
      const Target t = ret.AsObjectRef<Target>();
      bool useqhl = true;
      if (t.defined()) {
        const std::string tstring = t->str();
        useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
      }

      // Enable QHL library for FP16 data type
      if (x->dtype.is_float16() && x->dtype.is_vector() && useqhl) {
        std::string tvm_wrapper("tvm_vect_qhmath_hvx_tan_ahf");
        return TVMExternCall(call, tvm_wrapper);
      }
#endif
      PrimExpr tan_x = sin(x) / cos(x);
      return tan_x;
    });

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_OP("tir.sigmoid")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic", [](const PrimExpr& e) {
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      const PrimExpr& x = call->args[0];
#if ENABLE_QHL
      // Check target for qfloat enablement
      const auto* f = tvm::runtime::Registry::Get("target.TargetCurrent");
      ICHECK(f != nullptr);
      const auto ret = (*f)(true);
      const Target t = ret.AsObjectRef<Target>();
      bool useqhl = true;
      if (t.defined()) {
        const std::string tstring = t->str();
        useqhl = tstring.find("+hvx-qfloat") != std::string::npos;
      }

      PrimExpr MinBound = tir::make_const(x.dtype(), -8);
      PrimExpr MaxBound = tir::make_const(x.dtype(), 8);
      const PrimExpr v1 = tir::Max(x, MinBound);
      const PrimExpr v2 = tir::Min(v1, MaxBound);

      Array<tvm::PrimExpr> new_args = {v2};
      const tir::Call new_call = tir::Call(call->dtype, call->op, new_args);

      // Enable QHL library for FP16 data type
      if (x->dtype.is_float16() && x->dtype.is_vector() && useqhl) {
        std::string tvm_wrapper("tvm_vect_qhmath_hvx_sigmoid_ahf");
        return TVMExternCall(new_call.get(), tvm_wrapper);
      }
#endif
      PrimExpr one = tir::make_const(x.dtype(), 1);
      return one / (one + exp(-x));
    });

TVM_REGISTER_QHL_OP_FP16(ceil, "tvm_vect_qhmath_hvx_ceil_ahf", 1)

TVM_REGISTER_QHL_OP_FP16(cos, "tvm_vect_qhmath_hvx_cos_ahf", 1)

TVM_REGISTER_QHL_OP_FP16(exp, "tvm_vect_qhmath_hvx_exp_ahf", 1)

TVM_REGISTER_QHL_OP_FP16(floor, "tvm_vect_qhmath_hvx_floor_ahf", 1)

TVM_REGISTER_QHL_OP_FP16(sin, "tvm_vect_qhmath_hvx_sin_ahf", 1)

TVM_REGISTER_QHL_OP_FP16(pow, "tvm_vect_qhmath_hvx_pow_ahf", 2)

TVM_REGISTER_QHL_OP_FP16(sqrt, "tvm_vect_qhmath_hvx_sqrt_ahf", 1)

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
