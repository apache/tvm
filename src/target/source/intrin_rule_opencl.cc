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
 * \file intrin_rule_opencl.cc
 * \brief OpenCL intrinsic rules.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tir/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
using tir::FLowerIntrinsic;

TVM_REGISTER_OP("tir.clz").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.nearbyint")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp10")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log10")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.popcount")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fmod")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

// There is no warp shuffle instruction in standard OpenCL
// When shuffle is used, we assume it is intel's shuffle extension
static PrimExpr DispatchIntelShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  arith::Analyzer analyzer;
  ICHECK(analyzer.CanProve(call->args[3] == call->args[4]))
      << "Intel warp shuffle dose not support width != warp_size";
  Array<PrimExpr> opencl_args{{StringImm("intel_sub_group_shuffle"), call->args[1], call->args[2]}};
  return Call(call->dtype, builtin::call_pure_extern(), opencl_args);
}

TVM_REGISTER_OP("tir.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchIntelShuffle);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
