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

namespace tvm {
namespace codegen {
namespace spirv {

using namespace runtime;

// num_signature means number of arguments used to query signature
template <unsigned id>
inline void DispatchGLSLPureIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));

  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = tir::Call(call->dtype, tir::builtin::call_spirv_pure_glsl450(), cargs);
}

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.floor")
    .set_body(DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.ceil").set_body(DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.round")
    .set_body(DispatchGLSLPureIntrin<GLSLstd450Round>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.trunc")
    .set_body(DispatchGLSLPureIntrin<GLSLstd450Trunc>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.fabs").set_body(DispatchGLSLPureIntrin<GLSLstd450FAbs>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.exp").set_body(DispatchGLSLPureIntrin<GLSLstd450Exp>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.log").set_body(DispatchGLSLPureIntrin<GLSLstd450Log>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.sqrt").set_body(DispatchGLSLPureIntrin<GLSLstd450Sqrt>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.pow").set_body(DispatchGLSLPureIntrin<GLSLstd450Pow>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.vulkan.tanh").set_body(DispatchGLSLPureIntrin<GLSLstd450Tanh>);

// WebGPU rules.
TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.floor")
    .set_body(DispatchGLSLPureIntrin<GLSLstd450Floor>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.ceil").set_body(DispatchGLSLPureIntrin<GLSLstd450Ceil>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.round")
    .set_body(DispatchGLSLPureIntrin<GLSLstd450Round>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.trunc")
    .set_body(DispatchGLSLPureIntrin<GLSLstd450Trunc>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.fabs").set_body(DispatchGLSLPureIntrin<GLSLstd450FAbs>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.exp").set_body(DispatchGLSLPureIntrin<GLSLstd450Exp>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.log").set_body(DispatchGLSLPureIntrin<GLSLstd450Log>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.sqrt").set_body(DispatchGLSLPureIntrin<GLSLstd450Sqrt>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.pow").set_body(DispatchGLSLPureIntrin<GLSLstd450Pow>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.webgpu.tanh").set_body(DispatchGLSLPureIntrin<GLSLstd450Tanh>);

}  // namespace spirv
}  // namespace codegen
}  // namespace tvm
