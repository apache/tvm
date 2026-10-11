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
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op_attr_types.h>

#include <sstream>

namespace tvm {
namespace codegen {

inline PrimExpr DispatchPureExternLibDevice(const PrimExpr& e) {
  using namespace tirx;
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  PrimType call_ty = call->ty.as_or_throw<PrimType>();
  TVM_FFI_ICHECK(call_ty.bits() == 32 || call_ty.bits() == 64)
      << "Only support float32 or float64.";

  const OpNode* op = call->op.as<OpNode>();
  TVM_FFI_ICHECK(op != nullptr);
  std::string name = op->name;
  TVM_FFI_ICHECK(name.substr(0, 5) == "tirx." || name.substr(0, 5) == "prim.")
      << "Unexpected intrinsic name: " << name;

  std::ostringstream intrinsic_name;
  intrinsic_name << "__nv_" << name.substr(5);
  if (call_ty.bits() == 32) intrinsic_name << "f";

  ffi::Array<Expr> new_args = {StringImm(intrinsic_name.str())};
  new_args.insert(new_args.end(), call->args.begin(), call->args.end());
  return Call(call_ty, tirx::call_pure_extern_op(), new_args).as_or_throw<PrimExpr>();
}

namespace llvm {
using tirx::FLowerIntrinsic;

TVM_FFI_STATIC_INIT_BLOCK() {
  // clang-format off
  OpDef("prim.floor")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.ceil")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.round")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", [](const PrimExpr& e) -> PrimExpr {
        // Redirect to nearbyint (ties-to-even) to match constant-folding semantics.
        using namespace tirx;
        const CallNode* call = e.as<CallNode>();
        TVM_FFI_ICHECK(call != nullptr);
        static const Op nearbyint_op = Op::Get("prim.nearbyint");
        auto new_call = Call(call->ty.as_or_throw<PrimType>(), nearbyint_op, call->args)
                            .as_or_throw<PrimExpr>();
        return DispatchPureExternLibDevice(new_call);
      });

  OpDef("prim.nearbyint")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.trunc")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.fabs")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.exp")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.exp2")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.exp10")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.erf")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.fma")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.log")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.log2")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.log10")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.sqrt")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.pow")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.tanh")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.tan")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.cos")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.cosh")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.sin")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.sinh")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);

  OpDef("prim.atan")
      .set_attr<FLowerIntrinsic>("nvptx.FLowerIntrinsic", DispatchPureExternLibDevice);
  // clang-format on
}

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
