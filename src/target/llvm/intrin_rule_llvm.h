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
 * \file intrin_rule_llvm.h
 * \brief Common utilities for llvm intrinsics.
 */
#ifndef TVM_TARGET_LLVM_INTRIN_RULE_LLVM_H_
#define TVM_TARGET_LLVM_INTRIN_RULE_LLVM_H_

#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace codegen {
// num_signature means number of arguments used to query signature
template <unsigned id, int num_signature>
inline PrimExpr DispatchLLVMPureIntrin(const PrimExpr& e) {
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));
  cargs.push_back(IntImm(DataType::UInt(32), num_signature));

  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  return tir::Call(call->dtype, tir::builtin::call_llvm_pure_intrin(), cargs);
}

template <unsigned id, int num_signature>
inline PrimExpr DispatchLLVMIntrin(const PrimExpr& e) {
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));
  cargs.push_back(IntImm(DataType::UInt(32), num_signature));
  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  return tir::Call(call->dtype, tir::builtin::call_llvm_intrin(), cargs);
}

}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
#endif  // TVM_TARGET_LLVM_INTRIN_RULE_LLVM_H_
