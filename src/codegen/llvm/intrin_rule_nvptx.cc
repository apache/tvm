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
 * \file intrin_rule_nvptx.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/tir/expr.h>
#include <tvm/tir/expr.h>
#include <tvm/runtime/registry.h>
#include <sstream>

namespace tvm {
namespace codegen {

inline void DispatchExternLibDevice(const TVMArgs& args, TVMRetValue* rv) {
  PrimExpr e = args[0];
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  CHECK(call != nullptr);
  CHECK(call->dtype.bits() == 32 || call->dtype.bits() == 64) << "Only support float32 or float64.";
  std::ostringstream intrinsic_name;
  intrinsic_name << "__nv_" << call->name;
  if (call->dtype.bits() == 32) intrinsic_name << "f";
  *rv = CallNode::make(call->dtype, intrinsic_name.str(), call->args,
                   CallNode::PureExtern);
}

namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.floor")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.ceil")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.round")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.trunc")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.fabs")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.exp")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.erf")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.fma")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.log")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.sqrt")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.pow")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.tanh")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.cos")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.sin")
.set_body(DispatchExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.atan")
.set_body(DispatchExternLibDevice);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
