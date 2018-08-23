/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_arm.cc
 * \brief ARM specific code generator
 */
#ifdef TVM_LLVM_VERSION
#include "codegen_cpu.h"

namespace tvm {
namespace codegen {

// ARM specific code generator, this is used as an example on
// how to override behavior llvm code generator for specific target
class CodeGenARM final : public CodeGenCPU {
 public:
  void InitTarget(llvm::TargetMachine* tm) final {
    // set native vector bits.
    native_vector_bits_ = 16 * 8;
    CodeGenCPU::InitTarget(tm);
  }
  llvm::Value* CreateIntrinsic(const Call* op) override;

 private:
  Expr ARMPopcount(const Call* op);
};

llvm::Value* CodeGenARM::CreateIntrinsic(const Call* op) {
  if (op->is_intrinsic("llvm_intrin")) {
    llvm::Intrinsic::ID id = static_cast<llvm::Intrinsic::ID>(
        op->args[0].as<UIntImm>()->value);
    if (id == ::llvm::Intrinsic::ctpop) {
      Expr e = ARMPopcount(op);
      return CodeGenCPU::CreateIntrinsic(e.as<Call>());
    }
  }
  return CodeGenCPU::CreateIntrinsic(op);
}

Expr CodeGenARM::ARMPopcount(const Call *call) {
  using namespace ir;
  const Expr& e = call->args[2];
  ::llvm::Intrinsic::ID ctpop_id = ::llvm::Intrinsic::ctpop;
  ::llvm::Intrinsic::ID vpaddlu_id = ::llvm::Intrinsic::arm_neon_vpaddlu;

  // Fallback to default llvm lowering rule if input type not a full vector or half vector length
  int total_size =  call->type.bits() * call->type.lanes();
  if (!call->type.is_vector() || call->type.bits() == 8 ||
     (total_size != 128 && total_size != 64)) {
    Array<Expr> vcnt_args;
    vcnt_args.push_back(ir::UIntImm::make(UInt(32), ctpop_id));
    vcnt_args.push_back(ir::UIntImm::make(UInt(32), 1));
    vcnt_args.push_back(e);
    return ir::Call::make(call->type,  "llvm_intrin", vcnt_args, Call::PureIntrinsic);
  }

  // Popcount lowering rule:
  // Reinterpret input vector as a vector of 8bit values and preform popcount
  // Pairwise add between adjacent elements and double width with vpaddlu
  // to return back to original input type

  // Dvisions are always divisible (number of bits = 64 or 128)
  Type uint8_type = Type(e.type().code(), 8, e.type().bits() * e.type().lanes() / 8);
  Type uint16_type = Type(uint8_type.code(), 16, uint8_type.bits() * uint8_type.lanes() / 16);
  Type uint32_type = Type(uint16_type.code(), 32, uint8_type.bits() * uint8_type.lanes() / 32);

  // Interpret input as vector of 8bit values
  Expr input8 = reinterpret(uint8_type, e);
  // Popcount 8bit->8bit
  const Call* c0 = input8.as<Call>();
  CHECK(c0 != nullptr);
  Array<Expr> vcnt8_args;
  vcnt8_args.push_back(ir::UIntImm::make(UInt(32), ctpop_id));
  vcnt8_args.push_back(ir::UIntImm::make(UInt(32), 1));
  vcnt8_args.push_back(input8);
  Expr vcnt8 = ir::Call::make(uint8_type,  "llvm_intrin", vcnt8_args, Call::PureIntrinsic);

  // Accumulation 8->16bit
  Array<Expr> vcnt16_args;
  vcnt16_args.push_back(ir::UIntImm::make(UInt(32), vpaddlu_id));
  vcnt16_args.push_back(ir::UIntImm::make(UInt(32), 1));
  vcnt16_args.push_back(vcnt8);
  Expr vcnt16 = ir::Call::make(uint16_type, "llvm_intrin", vcnt16_args, Call::PureIntrinsic);
  if (call->type.bits() == 16) {
    return vcnt16;
  }

  // Accumulation 16->32bit
  Array<Expr> vcnt32_args;
  vcnt32_args.push_back(ir::UIntImm::make(UInt(32), vpaddlu_id));
  vcnt32_args.push_back(ir::UIntImm::make(UInt(32), 1));
  vcnt32_args.push_back(vcnt16);
  Expr vcnt32 = ir::Call::make(uint32_type,  "llvm_intrin", vcnt32_args, Call::PureIntrinsic);
  if (call->type.bits() == 32) {
    return vcnt32;
  }

  // Accumulation 32->64bit
  Array<Expr> vcnt64_args;
  vcnt64_args.push_back(ir::UIntImm::make(UInt(32), vpaddlu_id));
  vcnt64_args.push_back(ir::UIntImm::make(UInt(32), 1));
  vcnt64_args.push_back(vcnt32);
  return ir::Call::make(call->type,  "llvm_intrin", vcnt64_args, Call::PureIntrinsic);
}

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_arm")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
    CodeGenLLVM* cg = new CodeGenARM();
    *rv = static_cast<void*>(cg);
  });

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
