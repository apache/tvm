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
 *  Copyright (c) 2017 by Contributors
 * \file intrin_rule_rocm.cc
 */
#ifdef TVM_LLVM_VERSION

#include "intrin_rule_llvm.h"
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/api_registry.h>
#include <sstream>

namespace tvm {
namespace codegen {

inline void DispatchExternOCML(const TVMArgs& args, TVMRetValue* rv) {
  Expr e = args[0];
  using namespace ir;
  const Call* call = e.as<Call>();
  CHECK(call != nullptr);
  std::ostringstream intrinsic_name;
  intrinsic_name << "__ocml_" << call->name << "_f" << call->type.bits();
  *rv = Call::make(call->type, intrinsic_name.str(), call->args,
                   Call::PureExtern);
}

namespace llvm {

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.floor")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.ceil")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.round")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.trunc")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fabs")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.exp")
.set_body(DispatchExternOCML);

// On AMD GPU, fma is slower than mac
// removing fma dispatch allows backend to generate faster mac instruction
TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.fma")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.log")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.sqrt")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.pow")
.set_body(DispatchExternOCML);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.rocm.tanh")
.set_body(DispatchExternOCML);
}  // namespace llvm
}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
