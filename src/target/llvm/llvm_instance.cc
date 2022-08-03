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

#ifdef TVM_LLVM_VERSION

#include "llvm_instance.h"

#include <dmlc/base.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#if TVM_LLVM_VERSION >= 150
#include <llvm/IR/FMF.h>
#else
#include <llvm/IR/Operator.h>
#endif
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#if TVM_LLVM_VERSION >= 140
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/target/target.h>

#include <atomic>
#include <sstream>
#include <string>
#include <system_error>

namespace tvm {
namespace codegen {

namespace {
namespace defaults {
static const char* cpu = "generic";
static const llvm::CodeGenOpt::Level opt_level = llvm::CodeGenOpt::Aggressive;
}  // namespace defaults
}  // namespace

namespace {
bool InitializeLLVM() {
  static std::atomic_flag initialized = ATOMIC_FLAG_INIT;
  if (!initialized.test_and_set()) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  }
  return true;
}

std::string Join(std::string sep, llvm::ArrayRef<std::string> strings) {
  std::string result;
  bool is_first = true;
  for (const std::string& s : strings) {
    if (!is_first) {
      result += sep;
    }
    result += s;
    is_first = false;
  }
  return result;
}

}  // namespace

// LLVMInstance

LLVMInstance::LLVMInstance() {
  // Call InitializeLLVM before anything else.
  static const bool DMLC_ATTRIBUTE_UNUSED init_llvm = InitializeLLVM();
  ctx_ = std::make_shared<llvm::LLVMContext>();
}

LLVMInstance::~LLVMInstance() = default;

std::unique_ptr<llvm::Module> LLVMInstance::ParseIR(const std::string& llvm_ir) const {
  auto buffer = llvm::MemoryBuffer::getMemBuffer(llvm_ir, /*BufferName=*/"",
                                                 /*RequiresNullTerminator=*/false);
  return ParseBuffer(*buffer);
}

std::unique_ptr<llvm::Module> LLVMInstance::LoadIR(const std::string& file_name) const {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> maybe_buffer =
      llvm::MemoryBuffer::getFileAsStream(file_name);
  if (std::error_code ec = maybe_buffer.getError()) {
    LOG(FATAL) << ec.message();
  }
  return ParseBuffer(**maybe_buffer);
}

std::unique_ptr<llvm::Module> LLVMInstance::ParseBuffer(const llvm::MemoryBuffer& buffer) const {
  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> module = llvm::parseIR(buffer.getMemBufferRef(), error, *ctx_);
  if (module == nullptr) {
    std::string message;
    llvm::raw_string_ostream ostream(message);
    error.print(/*ProgName=*/nullptr, ostream, /*ShowColors=*/false, /*ShowKindLabel=*/true);
    LOG(FATAL) << ostream.str();
  }

  return module;
}

// LLVMTarget

LLVMTarget::LLVMTarget(LLVMInstance& instance, const Target& target)
    : instance_(instance), ctx_(instance.GetContext()) {
  triple_ = target->GetAttr<String>("mtriple").value_or("default");

  if (triple_.empty() || triple_ == "default") {
    triple_ = llvm::sys::getDefaultTargetTriple();
  }
  cpu_ = target->GetAttr<String>("mcpu").value_or(defaults::cpu);

  if (const Optional<Array<String>>& v = target->GetAttr<Array<String>>("mattr")) {
    for (const String& s : v.value()) {
      attrs_.push_back(s);
    }
  }

  llvm::FloatABI::ABIType float_abi = llvm::FloatABI::Default;
  if (const Optional<String>& v = target->GetAttr<String>("mfloat-abi")) {
    String value = v.value();
    if (value == "hard") {
      float_abi = llvm::FloatABI::Hard;
    } else if (value == "soft") {
      float_abi = llvm::FloatABI::Soft;
    } else {
      LOG(FATAL) << "invalid -mfloat-abi option " << value;
    }
  }

  // Target options

#if TVM_LLVM_VERSION < 50
  target_options_.LessPreciseFPMADOption = true;
#endif
  // In clang, these are fed from LangOpts which describe language specific features
  // TODO(AndrewZhaoLuo): figure out how these relate to fast math flags
  target_options_.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  target_options_.UnsafeFPMath = false;
  target_options_.NoInfsFPMath = false;
  target_options_.NoNaNsFPMath = true;
  target_options_.FloatABIType = float_abi;
  if (const Optional<String>& v = target->GetAttr<String>("mabi")) {
    target_options_.MCOptions.ABIName = v.value();
  }

  auto maybe_level = target->GetAttr<Integer>("opt-level");

  if (maybe_level.defined()) {
    int level = maybe_level.value()->value;
    if (level <= 0) {
      opt_level_ = llvm::CodeGenOpt::None;
    } else if (level == 1) {
      opt_level_ = llvm::CodeGenOpt::Less;
    } else if (level == 2) {
      opt_level_ = llvm::CodeGenOpt::Default;
    } else {
      // level >= 3
      opt_level_ = llvm::CodeGenOpt::Aggressive;
    }
  } else {
    opt_level_ = defaults::opt_level;
  }

  // Fast math options

  auto GetBoolFlag = [&target](llvm::StringRef flag) -> bool {
    return target->GetAttr<Bool>(flag.str()).value_or(Bool(false));
  };
  if (GetBoolFlag("fast-math")) {
#if TVM_LLVM_VERSION >= 60
    fast_math_flags_.setFast();
#else
    fast_math_flags_.setUnsafeAlgebra();
#endif
  } else {
#if TVM_LLVM_VERSION >= 50
    // This option was added in 5.x, and has a boolean argument,
    // unlike the rest of options at the time.
    fast_math_flags_.setAllowContract(GetBoolFlag("fast-math-contract"));
#endif
#if TVM_LLVM_VERSION >= 70
    fast_math_flags_.setNoNaNs(GetBoolFlag("fast-math-nnan"));
    fast_math_flags_.setNoInfs(GetBoolFlag("fast-math-ninf"));
    fast_math_flags_.setNoSignedZeros(GetBoolFlag("fast-math-nsz"));
    fast_math_flags_.setAllowReciprocal(GetBoolFlag("fast-math-arcp"));
    fast_math_flags_.setAllowContract(GetBoolFlag("fast-math-contract"));
    fast_math_flags_.setAllowReassoc(GetBoolFlag("fast-math-reassoc"));
    fast_math_flags_.setApproxFunc(GetBoolFlag("fast-math-afn"));
#else
    // LLVM 4.x, 5.x, and 6.x
    if (GetBoolFlag("fast-math-nnan")) fast_math_flags_.setNoNaNs();
    if (GetBoolFlag("fast-math-ninf")) fast_math_flags_.setNoInfs();
    if (GetBoolFlag("fast-math-nsz")) fast_math_flags_.setNoSignedZeros();
    if (GetBoolFlag("fast-math-arcp")) fast_math_flags_.setAllowReciprocal();
#if TVM_LLVM_VERSION >= 60
    if (GetBoolFlag("fast-math-reassoc")) fast_math_flags_.setAllowReassoc();
    if (GetBoolFlag("fast-math-afn")) fast_math_flags_.setApproxFunc();
#endif
#endif
  }
}

LLVMTarget::LLVMTarget(LLVMInstance& scope, const std::string& target_str)
    : LLVMTarget(scope, Target(target_str)) {}

LLVMTarget::~LLVMTarget() = default;

llvm::LLVMContext* LLVMTarget::GetContext() const {
  ICHECK(!ctx_.expired()) << "LLVM scope has been deleted";
  return ctx_.lock().get();
}

llvm::TargetMachine* LLVMTarget::GetOrCreateTargetMachine(bool allow_missing) {
  if (target_machine_) return target_machine_.get();

  std::string error;
  if (const llvm::Target* llvm_instance = llvm::TargetRegistry::lookupTarget(triple_, error)) {
    llvm::TargetMachine* tm =
        llvm_instance->createTargetMachine(triple_, cpu_, GetTargetFeatureString(), target_options_,
                                           reloc_model_, code_model_, opt_level_);
    target_machine_ = std::unique_ptr<llvm::TargetMachine>(tm);
    if (!allow_missing) {
      ICHECK(target_machine_ != nullptr) << error;
    }
  }
  return target_machine_.get();
}

std::string LLVMTarget::GetTargetFeatureString() const {  //
  return Join(",", attrs_);
}

std::string LLVMTarget::str() const {
  std::ostringstream os;
  os << "llvm";
  if (!triple_.empty()) {
    os << " -mtriple=" << triple_;
  }
  if (!cpu_.empty() && cpu_ != defaults::cpu) {
    os << " -mcpu=" << cpu_;
  }
  if (!attrs_.empty()) {
    os << " -mattr=" << GetTargetFeatureString();
  }

  switch (target_options_.FloatABIType) {
    case llvm::FloatABI::Soft:
      os << " -mfloat-abi=soft";
      break;
    case llvm::FloatABI::Hard:
      os << " -mfloat-abi=hard";
      break;
    case llvm::FloatABI::Default:
      break;
  }
  if (!target_options_.MCOptions.ABIName.empty()) {
    os << " -mabi=" << target_options_.MCOptions.ABIName;
  }

  bool do_individual = true;
#if TVM_LLVM_VERSION >= 60
  if (fast_math_flags_.isFast()) {
    os << " -fast-math";
    do_individual = false;
  }
#else
  if (fast_math_flags_.unsafeAlgebra()) {
    os << " -fast-math";
    do_individual = false;
  }
#endif

  if (do_individual) {
    if (fast_math_flags_.noNaNs()) os << " -fast-math-nnan";
    if (fast_math_flags_.noInfs()) os << " -fast-math-ninf";
    if (fast_math_flags_.noSignedZeros()) os << " -fast-math-nsz";
    if (fast_math_flags_.allowReciprocal()) os << " -fast-math-arcp";
#if TVM_LLVM_VERSION >= 50
    if (fast_math_flags_.allowContract()) os << " -fast-math-contract";
#endif
#if TVM_LLVM_VERSION >= 60
    if (fast_math_flags_.allowReassoc()) os << " -fast-math-reassoc";
    if (fast_math_flags_.approxFunc()) os << " -fast-math-afn";
#endif
  }

  if (opt_level_ != defaults::opt_level) {
    os << " -opt-level=";
    switch (opt_level_) {
      case llvm::CodeGenOpt::None:
        os << "0";
        break;
      case llvm::CodeGenOpt::Less:
        os << "1";
        break;
      case llvm::CodeGenOpt::Default:
        os << "2";
        break;
      case llvm::CodeGenOpt::Aggressive:
        os << "3";
        break;
    }
  }

  return os.str();
}

std::string LLVMTarget::GetTargetMetadata(const llvm::Module& module) {
  if (llvm::Metadata* tvm_target = module.getModuleFlag("tvm_target")) {
    auto* mdstr = llvm::cast<llvm::MDString>(tvm_target);
    llvm::StringRef meta = mdstr->getString();
    if (meta.startswith("llvm")) {
      return meta.str();
    }
  }
  return "llvm -mtriple " + module.getTargetTriple();
}

void LLVMTarget::SetTargetMetadata(llvm::Module* module) const {
  module->addModuleFlag(llvm::Module::Warning, "tvm_target",
                        llvm::MDString::get(*GetContext(), str()));
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
