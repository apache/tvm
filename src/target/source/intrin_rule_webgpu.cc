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
#include <tvm/tir/op_attr_types.h>

#include "../intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tir::FLowerIntrinsic;

// See full list of builtin: https://www.w3.org/TR/WGSL/#builtin-functions

struct ReturnAbs {
  std::string operator()(DataType t, std::string name) const { return "abs"; }
};

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<ReturnAbs>);

TVM_REGISTER_OP("tir.acos")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.acosh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.asin")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.asinh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.atan")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.atan2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cos").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.cosh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.exp2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.fma").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.log2")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sin").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sinh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.tan").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic",
                                                     DispatchPureExtern<Direct>);

TVM_REGISTER_OP("tir.tanh")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchNumericalStableTanh);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchPureExtern<Direct>);

// extra dispatch
TVM_REGISTER_OP("tir.erf").set_attr<FLowerIntrinsic>("webgpu.FLowerIntrinsic", DispatchFastErf);

}  // namespace intrin
}  // namespace codegen
}  // namespace tvm
