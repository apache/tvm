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
 * \file intrin_rule_llvm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/tir/op.h>
#include "intrin_rule_llvm.h"

namespace tvm {
namespace codegen {
namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.prefetch")
.set_body(DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 4>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.exp")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.exp2")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::exp2, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.exp10")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& x = call->args[0];
  PrimExpr ln10 = make_const(x.dtype(), 2.302585093);
  PrimExpr ret = tir::CallNode::make(
      x.dtype(), "exp", {x * ln10}, tir::CallNode::PureIntrinsic);
  *rv = ret;
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fma")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.log")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.log2")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::log2, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.log10")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::log10, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sqrt")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.floor")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.ceil")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.trunc")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fabs")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.round")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.nearbyint")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.tanh")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& x = call->args[0];
  PrimExpr one = make_const(x.dtype(), 1);
  PrimExpr two = make_const(x.dtype(), 2);
  PrimExpr neg_two = make_const(x.dtype(), -2);

  PrimExpr exp_neg2x = tir::CallNode::make(
      x.dtype(), "exp", {neg_two * x}, tir::CallNode::PureIntrinsic);
  PrimExpr exp_pos2x = tir::CallNode::make(
      x.dtype(), "exp", {two * x}, tir::CallNode::PureIntrinsic);

  PrimExpr tanh_pos = (one - exp_neg2x) / (one + exp_neg2x);
  PrimExpr tanh_neg = (exp_pos2x - one) / (exp_pos2x + one);
  *rv = tir::SelectNode::make(
      x >= make_zero(x.dtype()), tanh_pos, tanh_neg);
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.pow")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 2>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.popcount")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.tan")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& x = call->args[0];
  PrimExpr sin_x = tir::CallNode::make(
      x.dtype(), "sin", {x}, tir::CallNode::PureIntrinsic);
  PrimExpr cos_x = tir::CallNode::make(
      x.dtype(), "cos", {x}, tir::CallNode::PureIntrinsic);
  PrimExpr tan_x = sin_x / cos_x;
  *rv = tan_x;
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.cos")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.cosh")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& x = call->args[0];
  PrimExpr two = make_const(x.dtype(), 2);
  PrimExpr neg_one = make_const(x.dtype(), -1);
  PrimExpr exp_negx = tir::CallNode::make(
      x.dtype(), "exp", {neg_one * x}, tir::CallNode::PureIntrinsic);
  PrimExpr exp_posx = tir::CallNode::make(
      x.dtype(), "exp", {x}, tir::CallNode::PureIntrinsic);
  PrimExpr ret = (exp_posx + exp_negx) / two;
  *rv = ret;
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sin")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sinh")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  using tir::make_const;
  using tir::make_zero;
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  CHECK(call != nullptr);
  const PrimExpr& x = call->args[0];
  PrimExpr two = make_const(x.dtype(), 2);
  PrimExpr neg_one = make_const(x.dtype(), -1);
  PrimExpr exp_negx = tir::CallNode::make(
      x.dtype(), "exp", {neg_one * x}, tir::CallNode::PureIntrinsic);
  PrimExpr exp_posx = tir::CallNode::make(
      x.dtype(), "exp", {x}, tir::CallNode::PureIntrinsic);
  PrimExpr ret = (exp_posx - exp_negx) / two;
  *rv = ret;
});

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
