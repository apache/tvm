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

#ifdef TVM_LLVM_VERSION

#include <tvm/tir/op_attr_types.h>

#include "intrin_rule_llvm.h"

namespace tvm {
namespace codegen {
namespace llvm {
using tir::FLowerIntrinsic;

TVM_REGISTER_OP("tir.exp").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_OP("tir.fma").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

TVM_REGISTER_OP("tir.log").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_OP("tir.sqrt")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_OP("tir.floor")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_OP("tir.ceil")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_OP("tir.trunc")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_OP("tir.fabs")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_OP("tir.round")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_OP("tir.pow").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 1>);

TVM_REGISTER_OP("tir.ctpop")
    .set_attr<FLowerIntrinsic>("hexagon.FLowerIntrinsic",
                               DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
