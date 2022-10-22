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
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace codegen {
namespace spirv {
// num_signature means number of arguments used to query signature
template <unsigned id>
PrimExpr CallGLSLIntrin(PrimExpr e, const Array<PrimExpr>& args) {
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));

  for (PrimExpr arg : args) {
    cargs.push_back(arg);
  }
  return tir::Call(call->dtype, tir::builtin::call_spirv_pure_glsl450(), cargs);
}

template <unsigned id>
PrimExpr CallGLSLIntrin(PrimExpr e) {
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  return CallGLSLIntrin<id>(e, call->args);
}

template <unsigned id>
inline PrimExpr DispatchGLSLPureIntrin(const PrimExpr& e) {
  return CallGLSLIntrin<id>(e);
}

namespace intrin {
using tir::FLowerIntrinsic;
TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Round>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Round>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Trunc>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450FAbs>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Exp>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Sin>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Cos>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Log>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Log2>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sqrt>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Pow>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Tanh>);

// WebGPU rules.
TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Round>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Round>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Trunc>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450FAbs>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Exp>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Log>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Sqrt>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchGLSLPureIntrin<GLSLstd450Pow>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Tanh>);
}  // namespace intrin

namespace legalize {
using tir::FLegalize;
TVM_REGISTER_OP("tir.clz").set_attr<FLegalize>(
    "vulkan.FLegalize", [](const PrimExpr& e) -> PrimExpr {
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      ICHECK_EQ(call->args.size(), 1);
      PrimExpr arg = call->args[0];
      PrimExpr msb;
      if (arg.dtype().bits() == 64) {
        // SPIR-V FindUMsb intrinsic only supports 32 bit input
        auto int32 = DataType::Int(32);
        PrimExpr arg_hi32 = tvm::tir::Cast(int32, arg >> 32);
        PrimExpr arg_lo32 = tvm::tir::Cast(int32, arg);
        PrimExpr msb_hi = CallGLSLIntrin<GLSLstd450FindUMsb>(e, {arg_hi32});
        PrimExpr msb_lo = CallGLSLIntrin<GLSLstd450FindUMsb>(e, {arg_lo32});
        msb = tvm::if_then_else(arg_hi32 == 0, msb_lo, msb_hi + 32);
      } else if (arg.dtype().bits() == 32) {
        msb = CallGLSLIntrin<GLSLstd450FindUMsb>(e);
      } else {
        LOG(FATAL) << "SPIR-V clz only supports a 32 bit or 64 bit integer.";
      }
      return PrimExpr(arg.dtype().bits() - 1) - msb;
    });
}  // namespace legalize
}  // namespace spirv
}  // namespace codegen
}  // namespace tvm
