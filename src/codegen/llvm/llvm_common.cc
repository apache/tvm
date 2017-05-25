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

llvm::TargetMachine*
GetLLVMTargetMachine(const std::string& target_str) {
  // setup target triple
  CHECK(target_str.length() >= 4 &&
        target_str.substr(0, 4) == "llvm")
      << "llvm target must starts with llvm";
  // simple parser
  std::string target_triple = "";
  std::string cpu = "generic";
  std::string features = "";
  std::string key, value;
  if (target_str.length() > 5) {
    std::istringstream is(target_str.substr(5, target_str.length() - 5));
    while (is >> key) {
      size_t pos = key.find('=');
      if (pos != std::string::npos) {
        CHECK_GE(key.length(), pos + 1)
            << "inavlid argument " << key;
        value = key.substr(pos + 1, key.length() - 1);
        key = key.substr(0, pos);
      } else {
        CHECK(is >> value)
            << "Unspecified value for option " << key;
      }
      if (key == "-target" ||
          key == "-mtriple") {
        target_triple = value;
      } else if (key == "-mcpu") {
        cpu = value;
      } else if (key == "-features") {
        features = value;
      } else {
        LOG(FATAL) << "unknown option " << key;
      }
    }
  }
  if (target_triple.length() == 0 ||
      target_triple == "default") {
    target_triple = llvm::sys::getDefaultTargetTriple();
  }
  std::string err;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(target_triple, err);
  CHECK(target) << err << " target_triple=" << target_triple;
  llvm::TargetOptions opt;
  auto rmodel = llvm::Reloc::PIC_;
  llvm::TargetMachine* tm =
      target->createTargetMachine(target_triple, cpu, features, opt, rmodel);
  return tm;
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
