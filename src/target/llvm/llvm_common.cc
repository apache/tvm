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
 * \file llvm_common.cc
 */
#ifdef TVM_LLVM_VERSION

#include "llvm_common.h"

#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>

#include <atomic>
#include <memory>
#include <mutex>

namespace tvm {
namespace codegen {

struct LLVMEnv {
  std::mutex mu;
  std::atomic<bool> all_initialized{false};

  static LLVMEnv* Global() {
    static LLVMEnv inst;
    return &inst;
  }
};

void InitializeLLVM() {
  LLVMEnv* e = LLVMEnv::Global();
  if (!e->all_initialized.load(std::memory_order::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(e->mu);
    if (!e->all_initialized.load(std::memory_order::memory_order_acquire)) {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
      e->all_initialized.store(true, std::memory_order::memory_order_release);
    }
  }
}

void ParseLLVMTargetOptions(const Target& target, std::string* triple, std::string* mcpu,
                            std::string* mattr, llvm::TargetOptions* options) {
  // simple parser
  triple->resize(0);
  mcpu->resize(0);
  mattr->resize(0);
  bool soft_float_abi = false;
  if (const Optional<String>& v = target->GetAttr<String>("mtriple")) {
    *triple = v.value();
  }
  if (const Optional<String>& v = target->GetAttr<String>("mcpu")) {
    *mcpu = v.value();
  }
  if (const Optional<Array<String>>& v = target->GetAttr<Array<String>>("mattr")) {
    std::ostringstream os;
    bool is_first = true;
    for (const String& s : v.value()) {
      if (!is_first) {
        os << ',';
      }
      is_first = false;
      os << s;
    }
    *mattr = os.str();
  }
  if (const Optional<String>& v = target->GetAttr<String>("mfloat-abi")) {
    String value = v.value();
    if (value == "hard") {
#if TVM_LLVM_VERSION < 60
      LOG(FATAL) << "-mfloat-abi hard is only supported for LLVM > 6.0";
#endif
      soft_float_abi = false;
    } else if (value == "soft") {
      soft_float_abi = true;
    } else {
      LOG(FATAL) << "invalid -mfloat-abi option " << value;
    }
  }
  if (triple->length() == 0 || *triple == "default") {
    *triple = llvm::sys::getDefaultTargetTriple();
  }
  // set target option
  llvm::TargetOptions& opt = *options;
  opt = llvm::TargetOptions();
#if TVM_LLVM_VERSION < 50
  opt.LessPreciseFPMADOption = true;
#endif
  // We depend on generating IR with proper fast math flags to control fast math
  // semantics. These just enable these optimizations if the proper IR flags
  // are set.
  opt.UnsafeFPMath = true;
  opt.NoInfsFPMath = true;
  opt.NoNaNsFPMath = true;

#if TVM_LLVM_VERSION >= 50
  opt.NoSignedZerosFPMath = true;
#endif

  // Assume no generated code ever needs to handle floating point exceptions.
  opt.NoTrappingFPMath = true;

  // TODO(AndrewZhaoLuo): Look into control of setting this flag.
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;

  if (soft_float_abi) {
    opt.FloatABIType = llvm::FloatABI::Soft;
  } else {
    opt.FloatABIType = llvm::FloatABI::Hard;
  }
  if (const Optional<String>& v = target->GetAttr<String>("mabi")) {
    opt.MCOptions.ABIName = v.value();
  }
}

std::unique_ptr<llvm::TargetMachine> GetLLVMTargetMachine(const Target& target, bool allow_null) {
  std::string target_triple, mcpu, mattr;
  llvm::TargetOptions opt;

  ParseLLVMTargetOptions(target, &target_triple, &mcpu, &mattr, &opt);

  if (target_triple.length() == 0 || target_triple == "default") {
    target_triple = llvm::sys::getDefaultTargetTriple();
  }
  if (mcpu.length() == 0) {
    mcpu = "generic";
  }

  std::string err;
  const llvm::Target* llvm_target = llvm::TargetRegistry::lookupTarget(target_triple, err);
  if (llvm_target == nullptr) {
    ICHECK(allow_null) << err << " target_triple=" << target_triple;
    return nullptr;
  }

  Integer llvm_opt_level = target->GetAttr<Integer>("O").value_or(Integer(2));
  llvm::CodeGenOpt::Level llvm_opt;
  if (llvm_opt_level <= 0) {
    llvm_opt = llvm::CodeGenOpt::None;
  } else if (llvm_opt_level == 1) {
    llvm_opt = llvm::CodeGenOpt::Less;
  } else if (llvm_opt_level == 2) {
    llvm_opt = llvm::CodeGenOpt::Default;
  } else {
    // llvm_opt_level >= 3
    llvm_opt = llvm::CodeGenOpt::Aggressive;
  }

  llvm::TargetMachine* tm = llvm_target->createTargetMachine(
      target_triple, mcpu, mattr, opt, llvm::Reloc::PIC_, llvm::CodeModel::Small, llvm_opt);
  return std::unique_ptr<llvm::TargetMachine>(tm);
}

std::string LLVMTargetToString(const Target& target) {
  std::ostringstream os;
  os << "llvm";
  if (Optional<String> mtriple = target->GetAttr<String>("mtriple")) {
    os << " -mtriple=" << mtriple.value();
  }
  if (Optional<String> mcpu = target->GetAttr<String>("mcpu")) {
    os << " -mcpu=" << mcpu.value();
  }
  if (Optional<Array<String>> mattr = target->GetAttr<Array<String>>("mattr")) {
    bool is_first = true;
    os << " -mattr=";
    for (const String& attr : mattr.value()) {
      if (!is_first) {
        os << ",";
      }
      is_first = false;
      os << attr;
    }
  }
  if (Optional<String> mfloat_abo = target->GetAttr<String>("mfloat-abi")) {
    os << " -mfloat-abi=" << mfloat_abo.value();
  }
  if (Optional<String> mabi = target->GetAttr<String>("mabi")) {
    os << " -mabi=" << mabi.value();
  }
  return os.str();
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
