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
 * \file intrin_rule_rocm.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <sstream>

namespace tvm {
namespace codegen {

inline void DispatchExternOCML(const TVMArgs& args, TVMRetValue* rv) {
  PrimExpr e = args[0];
  using namespace tir;
  const CallNode* call = e.as<CallNode>();
  CHECK(call != nullptr);
  std::ostringstream intrinsic_name;
  intrinsic_name << "__ocml_" << call->name << "_f" << call->dtype.bits();
  *rv = Call(call->dtype, intrinsic_name.str(), call->args, CallNode::PureExtern);
}

inline void DispatchShuffle(const TVMArgs& targs, TVMRetValue* rv) {
  PrimExpr e_call = targs[0];
  using namespace tir;
  const CallNode* call = e_call.as<CallNode>();
  CHECK(call != nullptr);
  CHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  PrimExpr var = call->args[1];
  CHECK_EQ(var.dtype().bits(), 32);

  // get own lane in self (__lane_id)
  PrimExpr minus_one = tir::make_const(DataType::Int(32), -1);
  PrimExpr zero = tir::make_zero(DataType::Int(32));
  PrimExpr lo =
      Call(DataType::Int(32), "llvm.amdgcn.mbcnt.lo", {minus_one, zero}, CallNode::PureExtern);
  PrimExpr self =
      Call(DataType::Int(32), "llvm.amdgcn.mbcnt.hi", {minus_one, lo}, CallNode::PureExtern);

  // compute lane to get from
  PrimExpr width = call->args[3];
  PrimExpr index;
  if (call->name == "tvm_warp_shuffle") {
    PrimExpr src_lane = call->args[2];
    index = src_lane + (self & ~(width - 1));
  } else if (call->name == "tvm_warp_shuffle_up") {
    PrimExpr delta = call->args[2];
    index = self - delta;
    index = Select(index < (self & ~(width - 1)), self, index);
  } else {
    CHECK_EQ(call->name, "tvm_warp_shuffle_down");
    PrimExpr delta = call->args[2];
    index = self + delta;
    index = Select((self & (width - 1)) + delta >= width, self, index);
  }
  PrimExpr res =
      Call(var.dtype(), "llvm.amdgcn.ds.bpermute", {index << 2, var}, CallNode::PureExtern);
  *rv = res;
}

namespace llvm {

// dummy because we don't have the activemask
TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tvm_warp_activemask")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      PrimExpr zero = tir::make_zero(DataType::Int(32));
      *rv = zero;
    });

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tvm_warp_shuffle").set_body(DispatchShuffle);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tvm_warp_shuffle_up").set_body(DispatchShuffle);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tvm_warp_shuffle_down").set_body(DispatchShuffle);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.floor").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.ceil").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.round").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.trunc").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fabs").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.exp").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.exp2").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.exp10").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.erf").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fma").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.log").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.log2").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.log10").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sqrt").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.pow").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tanh").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tan").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.cos").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.cosh").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sin").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sinh").set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.atan").set_body(DispatchExternOCML);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
