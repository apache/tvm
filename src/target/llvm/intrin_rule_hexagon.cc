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

#include "intrin_rule_llvm.h"

namespace tvm {
namespace codegen {
namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.exp")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.fma")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 3>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.log")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.sqrt")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.floor")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.ceil")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.trunc")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.fabs")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.round")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.pow")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.hexagon.popcount")
    .set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
