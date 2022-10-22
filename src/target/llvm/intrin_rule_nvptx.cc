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
#include <tvm/tir/op_attr_types.h>

#include <sstream>

namespace tvm {
namespace codegen {

inline PrimExpr DispatchPureExternLibDevice(const PrimExpr& e) {
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK(call->dtype.bits() == 32 || call->dtype.bits() == 64)
      << "Only support float32 or float64.";

  const OpNode* op = call->op.as<OpNode>();
  ICHECK(op != nullptr);
  std::string name = op->name;
  ICHECK_EQ(name.substr(0, 4), "tir.");

  std::ostringstream intrinsic_name;
  intrinsic_name << "__nv_" << name.substr(4);
  if (call->dtype.bits() == 32) intrinsic_name << "f";

  Array<PrimExpr> new_args = {StringImm(intrinsic_name.str())};
  for (auto arg : call->args) {
    new_args.push_back(arg);
  }
  return Call(call->dtype, builtin::call_pure_extern(), new_args);
}

namespace llvm {
using tir::FLowerIntrinsic;

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.exp10")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.fma").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.tan").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic",
                                                     DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tir.atan")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
