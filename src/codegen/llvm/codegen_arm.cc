/*!
 *  Copyright (c) 2017 by Contributors
 * \file codegen_arm.cc
 * \brief ARM specific code generator
 */
#ifdef TVM_LLVM_VERSION
#include "./codegen_cpu.h"

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
};

TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_arm")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
    CodeGenLLVM* cg = new CodeGenARM();
    *rv = static_cast<void*>(cg);
  });

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
