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
using tir::FLowerIntrinsic;

// num_signature means number of arguments used to query signature

template <unsigned id>
PrimExpr CallGLSLIntrin(const PrimExpr& e) {
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));

  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  return tir::Call(call->dtype, tir::builtin::call_spirv_pure_glsl450(), cargs);
}

template <unsigned id>
inline PrimExpr DispatchGLSLPureIntrin(const PrimExpr& e) {
  return CallGLSLIntrin<id>(e);
}

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("vulkan.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_OP("tir.round")
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

TVM_REGISTER_OP("tir.clz").set_attr<FLowerIntrinsic>(
    "vulkan.FLowerIntrinsic", [](const PrimExpr& e) -> PrimExpr {
      const tir::CallNode* call = e.as<tir::CallNode>();
      ICHECK(call != nullptr);
      ICHECK_EQ(call->args.size(), 1);
      PrimExpr arg = call->args[0];
      PrimExpr msb = CallGLSLIntrin<GLSLstd450FindUMsb>(e);
      return PrimExpr(arg.dtype().bits() - 1) - msb;
    });

// WebGPU rules.
TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_OP("tir.round")
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

}  // namespace spirv
}  // namespace codegen
}  // namespace tvm
