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

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <sstream>

namespace tvm {
namespace codegen {

inline void DispatchPureExternLibDevice(const TVMArgs& args, TVMRetValue* rv) {
  PrimExpr e = args[0];
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  CHECK(call != nullptr);
  CHECK(call->dtype.bits() == 32 || call->dtype.bits() == 64) << "Only support float32 or float64.";

  const OpNode* op = call->op.as<OpNode>();
  CHECK(op != nullptr);
  std::string name = op->name;
  CHECK_EQ(name.substr(0, 4), "tir.");

  std::ostringstream intrinsic_name;
  intrinsic_name << "__nv_" << name.substr(4);
  if (call->dtype.bits() == 32) intrinsic_name << "f";

  Array<PrimExpr> new_args = {StringImm(intrinsic_name.str())};
  for (auto arg : call->args) {
    new_args.push_back(arg);
  }
  *rv = Call(call->dtype, builtin::call_pure_extern(), new_args);
}

namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.floor").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.ceil").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.round").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.trunc").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.fabs").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.exp").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.exp2").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.exp10").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.erf").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.fma").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.log").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.log2").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.log10").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.sqrt").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.pow").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.tanh").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.tan").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.cos").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.cosh").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.sin").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.sinh").set_body(DispatchPureExternLibDevice);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.nvptx.atan").set_body(DispatchPureExternLibDevice);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
