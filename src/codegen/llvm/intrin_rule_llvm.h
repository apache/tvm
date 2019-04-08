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
 * \file intrin_rule_llvm.h
 * \brief Common utilities for llvm intrinsics.
 */
#ifndef TVM_CODEGEN_LLVM_INTRIN_RULE_LLVM_H_
#define TVM_CODEGEN_LLVM_INTRIN_RULE_LLVM_H_
#ifdef TVM_LLVM_VERSION

#include <tvm/ir.h>
#include <tvm/api_registry.h>
#include <tvm/codegen.h>
#include <string>
#include "llvm_common.h"

namespace tvm {
namespace codegen {
// num_signature means number of arguments used to query signature
template<unsigned id, int num_signature>
inline void DispatchLLVMPureIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const ir::Call* call = e.as<ir::Call>();
  CHECK(call != nullptr);
  Array<Expr> cargs;
  // intrin id.
  cargs.push_back(ir::UIntImm::make(UInt(32), id));
  cargs.push_back(ir::UIntImm::make(UInt(32), num_signature));

  for (Expr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = ir::Call::make(
      call->type, "llvm_intrin", cargs, ir::Call::PureIntrinsic);
}

template<unsigned id, int num_signature>
inline void DispatchLLVMIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const ir::Call* call = e.as<ir::Call>();
  CHECK(call != nullptr);
  Array<Expr> cargs;
  // intrin id.
  cargs.push_back(ir::UIntImm::make(UInt(32), id));
  cargs.push_back(ir::UIntImm::make(UInt(32), num_signature));
  for (Expr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = ir::Call::make(
      call->type, "llvm_intrin", cargs, ir::Call::Intrinsic);
}

}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
#endif  // TVM_CODEGEN_LLVM_INTRIN_RULE_LLVM_H_
