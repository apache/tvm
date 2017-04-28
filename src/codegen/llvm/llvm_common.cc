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
  bool all_initialized{false};

  static LLVMEnv* Global() {
    static LLVMEnv inst;
    return &inst;
  }
};

void InitializeLLVM() {
  LLVMEnv* e = LLVMEnv::Global();
  if (!e->all_initialized) {
    std::lock_guard<std::mutex>(e->mu);
    if (!e->all_initialized) {
      e->all_initialized = true;
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
    }
  }
}

std::pair<llvm::TargetMachine*, std::string>
GetLLVMTarget(const std::string& target_str) {
  // setup target triple
  std::string target_triple;
  CHECK_EQ(target_str.substr(0, 4), "llvm");
  if (target_str.length() > 4) {
    target_triple = target_str.substr(5, target_str.length() - 5);
  } else {
    target_triple = "";
  }
  if (target_triple.length() == 0 ||
      target_triple == "default") {
    target_triple = llvm::sys::getDefaultTargetTriple();
  }

  std::string err;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(target_triple, err);
  CHECK(target) << err << " target_triple=" << target_triple;
  std::string cpu = "generic";
  std::string features = "";
  llvm::TargetOptions opt;
  auto rmodel = llvm::Reloc::PIC_;
  llvm::TargetMachine* tm =
      target->createTargetMachine(target_triple, cpu, features, opt, rmodel);
  return {tm, target_triple};
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
