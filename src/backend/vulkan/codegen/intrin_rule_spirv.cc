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
 * \file intrin_rule_spirv.cc
 */
#include <GLSL.std.450.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op_attr_types.h>

#include "../../../target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace spirv {
// num_signature means number of arguments used to query signature
template <unsigned id>
PrimExpr CallGLSLIntrin(PrimExpr e, const ffi::Array<PrimExpr>& args) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  ffi::Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(PrimType::UInt(32), id));

  for (PrimExpr arg : args) {
    cargs.push_back(arg);
  }
  return Call(call->ty.as_or_throw<PrimType>(), tirx::call_spirv_pure_glsl450_op(), cargs)
      .as_or_throw<PrimExpr>();
}

template <unsigned id>
PrimExpr CallGLSLIntrin(PrimExpr e) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  ffi::Array<PrimExpr> args = call->args.as_or_throw<ffi::Array<PrimExpr>>();
  return CallGLSLIntrin<id>(e, args);
}

template <unsigned id>
inline PrimExpr DispatchGLSLPureIntrin(const PrimExpr& e) {
  return CallGLSLIntrin<id>(e);
}

namespace intrin {
using tirx::FLowerIntrinsic;

void RegisterVulkanLowerIntrinRules() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  OpDef("prim.floor")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Floor>);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Ceil>);

  OpDef("prim.round")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                                 DispatchGLSLPureIntrin<GLSLstd450RoundEven>);

  OpDef("prim.nearbyint")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                                 DispatchGLSLPureIntrin<GLSLstd450RoundEven>);

  OpDef("prim.trunc")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Trunc>);

  OpDef("prim.fabs")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450FAbs>);

  OpDef("prim.exp")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Exp>);

  OpDef("prim.exp2")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Exp2>);

  OpDef("prim.sin")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sin>);

  OpDef("prim.cos")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Cos>);

  OpDef("prim.tan")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Tan>);

  OpDef("prim.asin")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Asin>);

  OpDef("prim.acos")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Acos>);

  OpDef("prim.atan")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Atan>);

  OpDef("prim.sinh")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sinh>);

  OpDef("prim.cosh")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Cosh>);

  OpDef("prim.tanh")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Tanh>);

  OpDef("prim.asinh")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Asinh>);

  OpDef("prim.acosh")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Acosh>);

  OpDef("prim.atanh")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Atanh>);

  OpDef("prim.atan2")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Atan2>);

  OpDef("prim.log")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Log>);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Log2>);

  OpDef("prim.sqrt")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sqrt>);

  OpDef("prim.pow")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Pow>);

  OpDef("prim.erf")
      .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", codegen::intrin ::DispatchFastErf);
}
}  // namespace intrin

namespace legalize {
using tirx::FLegalize;

void RegisterVulkanLegalizeRules() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  // clang-format off
  OpDef("prim.clz")
      .set_attr<FLegalize>("vulkan.FLegalize", [](const PrimExpr& e) -> PrimExpr {
    const CallNode* call = e.as<CallNode>();
    TVM_FFI_ICHECK(call != nullptr);
    TVM_FFI_ICHECK_EQ(call->args.size(), 1);
    PrimExpr arg = call->args[0].as_or_throw<PrimExpr>();
    PrimType arg_ty = arg.ty();
    PrimExpr msb = [&]() -> PrimExpr {
      if (arg_ty.bits() == 64) {
        // SPIR-V FindUMsb intrinsic only supports 32 bit input
        auto int32 = PrimType::Int(32);
        PrimExpr arg_hi32 = tvm::prim::Cast(int32, arg >> 32);
        PrimExpr arg_lo32 = tvm::prim::Cast(int32, arg);
        PrimExpr msb_hi = CallGLSLIntrin<GLSLstd450FindUMsb>(e, {arg_hi32});
        PrimExpr msb_lo = CallGLSLIntrin<GLSLstd450FindUMsb>(e, {arg_lo32});
        return tvm::if_then_else(arg_hi32 == 0, msb_lo, msb_hi + 32);
      } else if (arg_ty.bits() == 32) {
        return CallGLSLIntrin<GLSLstd450FindUMsb>(e);
      } else {
        TVM_FFI_THROW(InternalError) << "SPIR-V clz only supports a 32 bit or 64 bit integer.";
      }
    }();
    return PrimExpr(arg_ty.bits() - 1) - msb;
  });
  // clang-format on
}
}  // namespace legalize

void RegisterVulkanIntrinRules() {
  intrin::RegisterVulkanLowerIntrinRules();
  legalize::RegisterVulkanLegalizeRules();
}

}  // namespace spirv
}  // namespace codegen
}  // namespace tvm
