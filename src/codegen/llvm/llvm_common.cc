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
    std::lock_guard<std::mutex> lock(e->mu);
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

void ParseLLVMTargetOptions(const std::string& target_str,
                            std::string* triple,
                            std::string* mcpu,
                            std::string* mattr,
                            llvm::TargetOptions* options) {
  // setup target triple
  size_t start = 0;
  if (target_str.length() >= 4 &&
      target_str.substr(0, 4) == "llvm") {
    start = 4;
  }
  // simple parser
  triple->resize(0);
  mcpu->resize(0);
  mattr->resize(0);

  bool soft_float_abi = false;
  std::string key, value;
  std::istringstream is(target_str.substr(start, target_str.length() - start));

  while (is >> key) {
    if (key == "--system-lib" || key == "-system-lib") {
      continue;
    }
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
      *triple = value;
    } else if (key == "-mcpu") {
      *mcpu = value;
    } else if (key == "-mattr") {
      *mattr = value;
    } else if (key == "-mfloat-abi") {
      if (value == "hard") {
        soft_float_abi = false;
      } else if (value == "soft") {
        soft_float_abi = true;
      } else {
        LOG(FATAL) << "invalid -mfloat-abi option " << value;
      }
    } else if (key == "-device" || key == "-libs" || key == "-model") {
      // pass
    } else {
      LOG(FATAL) << "unknown option " << key;
    }
  }

  if (triple->length() == 0 ||
      *triple == "default") {
    *triple = llvm::sys::getDefaultTargetTriple();
  }
  // set target option
  llvm::TargetOptions& opt = *options;
  opt = llvm::TargetOptions();
  #if TVM_LLVM_VERSION < 50
  opt.LessPreciseFPMADOption = true;
  #endif
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = true;
  opt.NoInfsFPMath = true;
  opt.NoNaNsFPMath = true;
  if (soft_float_abi) {
    opt.FloatABIType = llvm::FloatABI::Soft;
  } else {
    opt.FloatABIType = llvm::FloatABI::Hard;
  }
}


llvm::TargetMachine*
GetLLVMTargetMachine(const std::string& target_str,
                     bool allow_null) {
  std::string target_triple, mcpu, mattr;
  llvm::TargetOptions opt;

  ParseLLVMTargetOptions(target_str,
                         &target_triple,
                         &mcpu,
                         &mattr,
                         &opt);

  if (target_triple.length() == 0 ||
      target_triple == "default") {
    target_triple = llvm::sys::getDefaultTargetTriple();
  }
  if (mcpu.length() == 0) {
    mcpu = "generic";
  }

  std::string err;
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(target_triple, err);
  if (target == nullptr) {
    CHECK(allow_null) << err << " target_triple=" << target_triple;
    return nullptr;
  }
  llvm::TargetMachine* tm = target->createTargetMachine(
      target_triple, mcpu, mattr, opt, llvm::Reloc::PIC_);
  return tm;
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
