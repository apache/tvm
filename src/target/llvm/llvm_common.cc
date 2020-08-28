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

#include <dmlc/logging.h>
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
  opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  if (soft_float_abi) {
    opt.FloatABIType = llvm::FloatABI::Soft;
  } else {
    opt.FloatABIType = llvm::FloatABI::Hard;
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
    CHECK(allow_null) << err << " target_triple=" << target_triple;
    return nullptr;
  }
  llvm::TargetMachine* tm =
      llvm_target->createTargetMachine(target_triple, mcpu, mattr, opt, llvm::Reloc::PIC_);
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
  return os.str();
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
