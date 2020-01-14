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
#ifndef TVM_CODEGEN_LLVM_INTRIN_RULE_LLVM_H_
#define TVM_CODEGEN_LLVM_INTRIN_RULE_LLVM_H_
#ifdef TVM_LLVM_VERSION

#include <tvm/ir.h>
#include <tvm/runtime/registry.h>
#include <tvm/packed_func_ext.h>

#include <tvm/codegen.h>
#include <string>
#include "llvm_common.h"

namespace tvm {
namespace codegen {
// num_signature means number of arguments used to query signature
template<unsigned id, int num_signature>
inline void DispatchLLVMPureIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  PrimExpr e = targs[0];
  const ir::CallNode* call = e.as<ir::CallNode>();
  CHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(ir::UIntImmNode::make(DataType::UInt(32), id));
  cargs.push_back(ir::UIntImmNode::make(DataType::UInt(32), num_signature));

  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = ir::CallNode::make(
      call->dtype, "llvm_intrin", cargs, ir::CallNode::PureIntrinsic);
}

template<unsigned id, int num_signature>
inline void DispatchLLVMIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  PrimExpr e = targs[0];
  const ir::CallNode* call = e.as<ir::CallNode>();
  CHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(ir::UIntImmNode::make(DataType::UInt(32), id));
  cargs.push_back(ir::UIntImmNode::make(DataType::UInt(32), num_signature));
  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = ir::CallNode::make(
      call->dtype, "llvm_intrin", cargs, ir::CallNode::Intrinsic);
}

}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
#endif  // TVM_CODEGEN_LLVM_INTRIN_RULE_LLVM_H_
