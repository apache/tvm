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

#include <tvm/ffi/function.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include <sstream>

namespace tvm {
namespace codegen {

inline PrimExpr DispatchPureExternLibDevice(const PrimExpr& e) {
  using namespace tirx;
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK(call->dtype.bits() == 32 || call->dtype.bits() == 64)
      << "Only support float32 or float64.";

  const OpNode* op = call->op.as<OpNode>();
  TVM_FFI_ICHECK(op != nullptr);
  std::string name = op->name;
  TVM_FFI_ICHECK_EQ(name.substr(0, 5), "tirx.");

  std::ostringstream intrinsic_name;
  intrinsic_name << "__nv_" << name.substr(5);
  if (call->dtype.bits() == 32) intrinsic_name << "f";

  ffi::Array<PrimExpr> new_args = {StringImm(intrinsic_name.str())};
  for (auto arg : call->args) {
    new_args.push_back(arg);
  }
  return Call(call->dtype, builtin::call_pure_extern(), new_args);
}

namespace llvm {
using tirx::FLowerIntrinsic;

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", [](const PrimExpr& e) -> PrimExpr {
      // Redirect to nearbyint (ties-to-even) to match constant-folding semantics.
      using namespace tirx;
      const CallNode* call = e.as<CallNode>();
      TVM_FFI_ICHECK(call != nullptr);
      auto nearbyint_op = Op::Get("tirx.nearbyint");
      auto new_call = Call(call->dtype, nearbyint_op, call->args);
      return DispatchPureExternLibDevice(new_call);
    });

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.exp")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.exp10")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.erf")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.fma")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.log")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.log10")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.pow")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.tan")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.cos")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.sin")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
