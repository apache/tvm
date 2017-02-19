/*!
 *  Copyright (c) 2017 by Contributors
 * \file llvm_common.cc
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/base.h>
#include <mutex>
#include "./llvm_common.h"

namespace tvm {
namespace codegen {

struct LLVMEnv {
  std::mutex mu;
  bool native_initialized{false};

  static LLVMEnv* Global() {
    static LLVMEnv inst;
    return &inst;
  }
};

void InitializeLLVM() {
  LLVMEnv* e = LLVMEnv::Global();
  if (!e->native_initialized) {
    std::lock_guard<std::mutex>(e->mu);
    if (!e->native_initialized) {
      e->native_initialized = true;
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      llvm::InitializeNativeTargetAsmParser();
    }
  }
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
