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
 * \file codegen_x86_64.cc
 * \brief X86-64 specific code generator
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/registry.h>
#include "codegen_cpu.h"

#include "llvm/MC/MCSubtargetInfo.h"

namespace tvm {
namespace codegen {

namespace {
bool TargetHasFeature(const llvm::TargetMachine& tm, const std::string& feature) {
  // MCSubTargetInfo::checkFeatures was added in LLVM 6.0
#if TVM_LLVM_VERSION >= 60
  const auto* MCInfo = tm.getMCSubtargetInfo();
  return MCInfo->checkFeatures(std::string("+") + feature);
#else
  return false;
  // TODO(tulloch) - enable this block, need to figure out how to reimplement
  // this given visibility constraints, similar to
  // https://github.com/rust-lang/rust/pull/31709

  // Copied from
  // https://github.com/llvm-mirror/llvm/blob/5136df4/lib/MC/MCSubtargetInfo.cpp#L78-L88.

  // auto checkFeatures = [&](const std::string FS) {
  //   llvm::SubtargetFeatures T(FS);
  //   llvm::FeatureBitset Set, All;
  //   for (std::string F : T.getFeatures()) {
  //     llvm::SubtargetFeatures::ApplyFeatureFlag(Set, F, MCInfo->ProcFeatures);
  //     if (F[0] == '-') {
  //       F[0] = '+';
  //     }
  //     llvm::SubtargetFeatures::ApplyFeatureFlag(All, F, MCInfo->ProcFeatures);
  //   }
  //   return (MCInfo->getFeatureBits() & All) == Set;
  // };
  // return checkFeatures(MCInfo, std::string("+") + feature);
#endif
}
}  // namespace

class CodeGenX86_64 final : public CodeGenCPU {
 public:
  llvm::Value* VisitExpr_(const CastNode* op) override;

 private:
  llvm::Value* CallVectorIntrin(llvm::Intrinsic::ID id, size_t intrin_lanes, llvm::Type* result_ty,
                                const std::vector<llvm::Value*>& args);
};

llvm::Value* CodeGenX86_64::VisitExpr_(const CastNode* op) {
  // LLVM does not automatically generate the correct instruction sequences for
  // half -> float conversion (i.e. using AVX2/AVX-512 vectorized variants of
  // vcvtph2ps), so we explicitly generate them ourselves.
  const auto from = op->value.dtype();
  const auto to = op->dtype;
  if (from.is_float() && to.is_float() && from.bits() == 16 && to.bits() == 32) {
    CHECK_EQ(from.lanes(), to.lanes());
    CHECK_NOTNULL(target_machine_);

    const auto has_avx512 = TargetHasFeature(*target_machine_, "avx512f");

    if (from.lanes() >= 16 && has_avx512) {
      return CallVectorIntrin(
          ::llvm::Intrinsic::x86_avx512_mask_vcvtph2ps_512, 16,
          DTypeToLLVMType(DataType::Float(32, from.lanes())),
          {
            MakeValue(tir::CallNode::make(
                DataType::Int(16, from.lanes()), tir::CallNode::reinterpret, {op->value},
                tir::CallNode::PureIntrinsic)),
                MakeValue(
                    tir::BroadcastNode::make(
                      FloatImm(DataType::Float(32), 0), from.lanes())),
                /*mask=*/MakeValue(IntImm(DataType::Int(16), -1)),
                /*rounding-mode=*/MakeValue(IntImm(DataType::Int(32), 4)),
          });
    }

#if TVM_LLVM_VERSION <= 100
    // The intrinsic x86_vcvtph2ps_256 was removed in LLVM 11.
    const auto has_f16c = TargetHasFeature(*target_machine_, "f16c");

    if (from.lanes() >= 8 && has_f16c) {
      return CallVectorIntrin(
          ::llvm::Intrinsic::x86_vcvtph2ps_256, 8,
          DTypeToLLVMType(DataType::Float(32, from.lanes())),
          {MakeValue(tir::CallNode::make(
              DataType::Int(16, from.lanes()), tir::CallNode::reinterpret, {op->value},
              tir::CallNode::PureIntrinsic))});
    }
#endif
  }

  return CodeGenCPU::VisitExpr_(op);
}

llvm::Value* CodeGenX86_64::CallVectorIntrin(llvm::Intrinsic::ID id, size_t intrin_lanes,
                                             llvm::Type* result_ty,

                                             const std::vector<llvm::Value*>& args) {
  llvm::Function* f = llvm::Intrinsic::getDeclaration(module_.get(), id, {});
  size_t num_elems = llvm::cast<llvm::VectorType>(result_ty)->getNumElements();
  if (intrin_lanes == num_elems) {
    return builder_->CreateCall(f, args);
  }

  // Otherwise, we split the vector into intrin_lanes sized elements (widening where necessary),
  // compute each result, and then concatenate the vectors (slicing the result if necessary).
  CHECK_LT(intrin_lanes, num_elems);
  std::vector<llvm::Value*> split_results;
  for (size_t i = 0; i < num_elems; i += intrin_lanes) {
    std::vector<llvm::Value*> split_args;
    for (const auto& v : args) {
      if (v->getType()->isVectorTy()) {
        CHECK_EQ(llvm::cast<llvm::VectorType>(v->getType())->getNumElements(), num_elems);
        split_args.push_back(CreateVecSlice(v, i, intrin_lanes));
      } else {
        split_args.push_back(v);
      }
    }
    split_results.push_back(CallVectorIntrin(
        id, intrin_lanes, llvm::VectorType::get(result_ty->getScalarType(), intrin_lanes),
        split_args));
  }
  return CreateVecSlice(CreateVecConcat(split_results), 0, num_elems);
}

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_x86-64")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
    CodeGenLLVM* cg = new CodeGenX86_64();
    *rv = static_cast<void*>(cg);
  });

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
