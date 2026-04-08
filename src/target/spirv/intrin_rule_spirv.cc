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
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace spirv {
// num_signature means number of arguments used to query signature
template <unsigned id>
PrimExpr CallGLSLIntrin(PrimExpr e, const ffi::Array<PrimExpr>& args) {
  const tirx::CallNode* call = e.as<tirx::CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  ffi::Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));

  for (PrimExpr arg : args) {
    cargs.push_back(arg);
  }
  return tirx::Call(call->dtype, tirx::builtin::call_spirv_pure_glsl450(), cargs);
}

template <unsigned id>
PrimExpr CallGLSLIntrin(PrimExpr e) {
  const tirx::CallNode* call = e.as<tirx::CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  return CallGLSLIntrin<id>(e, call->args);
}

template <unsigned id>
inline PrimExpr DispatchGLSLPureIntrin(const PrimExpr& e) {
  return CallGLSLIntrin<id>(e);
}

namespace intrin {
using tirx::FLowerIntrinsic;
TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                               DispatchGLSLPureIntrin<GLSLstd450RoundEven>);

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                               DispatchGLSLPureIntrin<GLSLstd450RoundEven>);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Trunc>);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450FAbs>);

TVM_REGISTER_OP("tirx.exp")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Exp>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Exp2>);

TVM_REGISTER_OP("tirx.sin")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sin>);

TVM_REGISTER_OP("tirx.cos")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Cos>);

TVM_REGISTER_OP("tirx.tan")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Tan>);

TVM_REGISTER_OP("tirx.asin")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Asin>);

TVM_REGISTER_OP("tirx.acos")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Acos>);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Atan>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sinh>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Cosh>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Tanh>);

TVM_REGISTER_OP("tirx.asinh")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Asinh>);

TVM_REGISTER_OP("tirx.acosh")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Acosh>);

TVM_REGISTER_OP("tirx.atanh")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Atanh>);

TVM_REGISTER_OP("tirx.atan2")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Atan2>);

TVM_REGISTER_OP("tirx.log")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Log>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Log2>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sqrt>);

TVM_REGISTER_OP("tirx.pow")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Pow>);

TVM_REGISTER_OP("tirx.erf")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", codegen::intrin ::DispatchFastErf);
}  // namespace intrin

namespace legalize {
using tirx::FLegalize;
TVM_REGISTER_OP("tirx.clz")
    .set_attr<FLegalize>("vulkan.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const tirx::CallNode* call = e.as<tirx::CallNode>();
      TVM_FFI_ICHECK(call != nullptr);
      TVM_FFI_ICHECK_EQ(call->args.size(), 1);
      PrimExpr arg = call->args[0];
      PrimExpr msb;
      if (arg.dtype().bits() == 64) {
        // SPIR-V FindUMsb intrinsic only supports 32 bit input
        auto int32 = DataType::Int(32);
        PrimExpr arg_hi32 = tvm::tirx::Cast(int32, arg >> 32);
        PrimExpr arg_lo32 = tvm::tirx::Cast(int32, arg);
        PrimExpr msb_hi = CallGLSLIntrin<GLSLstd450FindUMsb>(e, {arg_hi32});
        PrimExpr msb_lo = CallGLSLIntrin<GLSLstd450FindUMsb>(e, {arg_lo32});
        msb = tvm::if_then_else(arg_hi32 == 0, msb_lo, msb_hi + 32);
      } else if (arg.dtype().bits() == 32) {
        msb = CallGLSLIntrin<GLSLstd450FindUMsb>(e);
      } else {
        TVM_FFI_THROW(InternalError) << "SPIR-V clz only supports a 32 bit or 64 bit integer.";
      }
      return PrimExpr(arg.dtype().bits() - 1) - msb;
    });
}  // namespace legalize
}  // namespace spirv
}  // namespace codegen
}  // namespace tvm
