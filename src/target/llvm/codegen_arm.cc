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
 * \file codegen_arm.cc
 * \brief ARM specific code generator
 */
#ifdef TVM_LLVM_VERSION

#include <llvm/IR/Intrinsics.h>
#include <tvm/runtime/registry.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsARM.h>
#endif
#include <llvm/Target/TargetMachine.h>

#include "codegen_cpu.h"

namespace tvm {
namespace codegen {

// ARM specific code generator, this is used as an example on
// how to override behavior llvm code generator for specific target
class CodeGenARM final : public CodeGenCPU {
 public:
  CodeGenARM() = default;
  virtual ~CodeGenARM() = default;

  void InitTarget() final {
    // set native vector bits.
    native_vector_bits_ = 16 * 8;
    CodeGenCPU::InitTarget();
  }
  llvm::Value* CreateIntrinsic(const CallNode* op) override;

 private:
  PrimExpr ARMPopcount(const CallNode* op);
};

llvm::Value* CodeGenARM::CreateIntrinsic(const CallNode* op) {
  if (op->op.same_as(builtin_call_llvm_intrin_) || op->op.same_as(builtin_call_llvm_pure_intrin_)) {
    llvm::Intrinsic::ID id = static_cast<llvm::Intrinsic::ID>(Downcast<IntImm>(op->args[0])->value);
    if (id == llvm::Intrinsic::ctpop) {
      PrimExpr e = ARMPopcount(op);
      return CodeGenCPU::CreateIntrinsic(e.as<CallNode>());
    }
  }
  return CodeGenCPU::CreateIntrinsic(op);
}

PrimExpr CodeGenARM::ARMPopcount(const CallNode* call) {
  using namespace tir;
  const PrimExpr& e = call->args[2];
  llvm::Intrinsic::ID ctpop_id = llvm::Intrinsic::ctpop;
  llvm::Intrinsic::ID vpaddlu_id = llvm::Intrinsic::arm_neon_vpaddlu;

  // Fallback to default llvm lowering rule if input type not a full vector or half vector length
  int total_size = call->dtype.bits() * call->dtype.lanes();
  if (!call->dtype.is_vector() || call->dtype.bits() == 8 ||
      (total_size != 128 && total_size != 64)) {
    Array<PrimExpr> vcnt_args;
    vcnt_args.push_back(IntImm(DataType::UInt(32), ctpop_id));
    vcnt_args.push_back(IntImm(DataType::UInt(32), 1));
    vcnt_args.push_back(e);
    return tir::Call(call->dtype, builtin_call_llvm_pure_intrin_, vcnt_args);
  }

  // Popcount lowering rule:
  // Reinterpret input vector as a vector of 8bit values and preform popcount
  // Pairwise add between adjacent elements and double width with vpaddlu
  // to return back to original input type

  // Dvisions are always divisible (number of bits = 64 or 128)
  DataType uint8_type = DataType(e.dtype().code(), 8, e.dtype().bits() * e.dtype().lanes() / 8);
  DataType uint16_type =
      DataType(uint8_type.code(), 16, uint8_type.bits() * uint8_type.lanes() / 16);
  DataType uint32_type =
      DataType(uint16_type.code(), 32, uint8_type.bits() * uint8_type.lanes() / 32);

  // Interpret input as vector of 8bit values
  PrimExpr input8 = reinterpret(uint8_type, e);
  // Popcount 8bit->8bit
  const CallNode* c0 = input8.as<CallNode>();
  ICHECK(c0 != nullptr);
  Array<PrimExpr> vcnt8_args;
  vcnt8_args.push_back(IntImm(DataType::UInt(32), ctpop_id));
  vcnt8_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt8_args.push_back(input8);
  PrimExpr vcnt8 = tir::Call(uint8_type, builtin_call_llvm_pure_intrin_, vcnt8_args);

  // Accumulation 8->16bit
  Array<PrimExpr> vcnt16_args;
  vcnt16_args.push_back(IntImm(DataType::UInt(32), vpaddlu_id));
  vcnt16_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt16_args.push_back(vcnt8);
  PrimExpr vcnt16 = tir::Call(uint16_type, builtin_call_llvm_pure_intrin_, vcnt16_args);
  if (call->dtype.bits() == 16) {
    return vcnt16;
  }

  // Accumulation 16->32bit
  Array<PrimExpr> vcnt32_args;
  vcnt32_args.push_back(IntImm(DataType::UInt(32), vpaddlu_id));
  vcnt32_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt32_args.push_back(vcnt16);
  PrimExpr vcnt32 = tir::Call(uint32_type, builtin_call_llvm_pure_intrin_, vcnt32_args);
  if (call->dtype.bits() == 32) {
    return vcnt32;
  }

  // Accumulation 32->64bit
  Array<PrimExpr> vcnt64_args;
  vcnt64_args.push_back(IntImm(DataType::UInt(32), vpaddlu_id));
  vcnt64_args.push_back(IntImm(DataType::UInt(32), 1));
  vcnt64_args.push_back(vcnt32);
  return tir::Call(call->dtype, builtin_call_llvm_pure_intrin_, vcnt64_args);
}

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_arm")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      *rv = static_cast<void*>(new CodeGenARM());
    });

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
