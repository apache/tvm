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
 * \file intrin_rule_webgpu.cc
 * \brief WebGPU intrinsic rules.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;

// See full list of builtin: https://www.w3.org/TR/WGSL/#builtin-functions

struct ReturnAbs {
  std::string operator()(DataType t, std::string name) const { return "abs"; }
};

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<ReturnAbs>);

TVM_REGISTER_OP("tirx.acos")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.acosh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.asin")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.asinh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.atan2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.cos").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.exp").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.fma").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.log").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.pow").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sin").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.tan").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchNumericalStableTanh);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

// extra dispatch
TVM_REGISTER_OP("tirx.erf").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchFastErf);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
