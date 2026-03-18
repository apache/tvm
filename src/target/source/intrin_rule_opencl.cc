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
#include <tvm/tirx/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
using tirx::FLowerIntrinsic;

TVM_REGISTER_OP("tirx.clz").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.exp").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.erf").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.exp10")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.log").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.log10")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.pow").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.popcount")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.fmod")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sin").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.cos").set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchPureExtern<Direct>);

// There is no warp shuffle instruction in standard OpenCL
// When shuffle is used, we assume it is intel's shuffle extension
static PrimExpr DispatchIntelShuffle(const PrimExpr& e) {
  const CallNode* call = e.as<CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  TVM_FFI_ICHECK_EQ(call->args.size(), 5);  // mask, value, warp_id, width, warp_size
  arith::Analyzer analyzer;
  TVM_FFI_ICHECK(analyzer.CanProve(call->args[3] == call->args[4]))
      << "Intel warp shuffle dose not support width != warp_size";
  ffi::Array<PrimExpr> opencl_args{
      {StringImm("intel_sub_group_shuffle"), call->args[1], call->args[2]}};
  return Call(call->dtype, builtin::call_pure_extern(), opencl_args);
}

TVM_REGISTER_OP("tirx.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("opencl.FLowerIntrinsic", DispatchIntelShuffle);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
