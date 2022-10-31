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
#include <llvm/Support/CommandLine.h>
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
#include <cctype>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

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

// LLVMTargetInfo

std::ostream& operator<<(std::ostream& os, const LLVMTargetInfo::Option& opt) {
  os << '-' << opt.name;
  switch (opt.type) {
    case LLVMTargetInfo::Option::OptType::Bool:
      return os << ":bool=" << (opt.value.b ? "true" : "false");
    case LLVMTargetInfo::Option::OptType::Int:
      return os << ":int=" << opt.value.i;
    case LLVMTargetInfo::Option::OptType::UInt:
      return os << ":uint=" << opt.value.u;
    case LLVMTargetInfo::Option::OptType::String:
      return os << ":string=" << opt.value.s;
    default:
      os << ":?(" << static_cast<int>(opt.type) << ")";
      break;
  }
  return os;
}

LLVMTargetInfo::LLVMTargetInfo(LLVMInstance& instance, const Target& target) {
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

  if (const Optional<Array<String>>& v = target->GetAttr<Array<String>>("cl-opt")) {
    llvm::StringMap<llvm::cl::Option*>& options = llvm::cl::getRegisteredOptions();
    bool parse_error = false;
    for (const String& s : v.value()) {
      Option opt = ParseOptionString(s);
      if (opt.type == Option::OptType::Invalid) {
        parse_error = true;
        continue;
      }
      if (options.count(opt.name)) {
        llvm_options_.push_back(opt);
      } else {
        // Flag an error, but don't abort. LLVM flags may change, and this would
        // give the code a chance to run even if the option no longer applies.
        LOG(ERROR) << "\"" << opt.name << "\" is not an LLVM option, option ignored";
      }
    }
    ICHECK(!parse_error) << "there were errors parsing command-line options";
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

  target_options_.UseInitArray = true;

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

LLVMTargetInfo::LLVMTargetInfo(LLVMInstance& scope, const std::string& target_str)
    : LLVMTargetInfo(scope, Target(target_str)) {}

LLVMTargetInfo::~LLVMTargetInfo() = default;

llvm::TargetMachine* LLVMTargetInfo::GetOrCreateTargetMachine(bool allow_missing) {
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

std::string LLVMTargetInfo::GetTargetFeatureString() const {  //
  return Join(",", attrs_);
}

std::string LLVMTargetInfo::str() const {
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

  if (size_t num = llvm_options_.size(); num > 0) {
    os << " -cl-opt=";
    std::vector<std::string> opts;
    for (const Option& opt : llvm_options_) {
      std::stringstream os;
      os << opt;
      opts.emplace_back(os.str());
    }
    auto* quote = num > 1 ? "'" : "";
    os << quote << Join(",", opts) << quote;
  }

  return os.str();
}

LLVMTargetInfo::Option LLVMTargetInfo::ParseOptionString(const std::string& str) {
  Option opt;
  opt.type = Option::OptType::Invalid;

  // Option string: "-"+ <option_name> ":" <type> "=" <value>
  //
  // Note: "-"+ means 1 or more dashes, but only "-" are "--" valid.

  // The first step is to do "lexing" of the option string, i.e. to break
  // it up into parts (like "tokens") according to the syntax above. These
  // parts will be non-overlapping substrings of the option string, and
  // concatenated together, they will be equal to the option string.
  // The literal elements are parts on their own.
  //
  // Note that the option string may be malformed, so any of the literal
  // elements in the syntax may be missing.

  std::vector<std::string> parts;

  auto find_first_of = [](const std::string& str, const std::string& chars, auto start = 0) {
    auto pos = str.find_first_of(chars, start);
    return pos != std::string::npos ? pos : str.size();
  };
  auto find_first_not_of = [](const std::string& str, const std::string& chars, auto start = 0) {
    auto pos = str.find_first_not_of(chars, start);
    return pos != std::string::npos ? pos : str.size();
  };

  // "-"+
  std::string::size_type pos_start = 0, pos_end = str.size();
  std::string::size_type pos_at = find_first_not_of(str, "-", pos_start);
  if (pos_at > 0) {
    parts.push_back(str.substr(pos_start, pos_at));
  }
  // <option_name>, always present, may be empty string
  pos_start = pos_at;
  pos_at = find_first_of(str, ":=", pos_start);
  parts.push_back(str.substr(pos_start, pos_at - pos_start));

  // ":" or "=", if any
  pos_start = pos_at;
  char c = pos_start < pos_end ? str[pos_start] : 0;
  if (c != 0) {
    parts.emplace_back(1, c);
    pos_start++;
  }
  // If the character found in the previous step wasn't '=', look for '='.
  if (c == ':') {
    // <type>
    pos_at = find_first_of(str, "=", pos_start);
    if (pos_at > pos_start) {  // if non-empty
      parts.push_back(str.substr(pos_start, pos_at - pos_start));
    }

    // "="
    if (pos_at < pos_end) {
      parts.emplace_back(1, str[pos_at]);
      pos_start = pos_at + 1;
    }
  }
  if (pos_start < pos_end) {
    // <value>
    parts.push_back(str.substr(pos_start));
  }

  // After breaking up the option string, examine and validate the individual
  // parts.

  int part_this = 0, part_end = parts.size();

  const std::string error_header = "while parsing option \"" + str + "\": ";

  // Check for "-" or "--".
  if (part_this < part_end) {
    auto& p = parts[part_this++];
    if ((p.size() != 1 && p.size() != 2) || p.find_first_not_of('-') != std::string::npos) {
      LOG(ERROR) << error_header << "option must start with \"-\" or \"--\"";
      return opt;
    }
  }

  // Validate option name.
  if (part_this < part_end) {
    auto& p = parts[part_this++];
    if (p.empty()) {
      LOG(ERROR) << error_header << "option name must not be empty";
      return opt;
    }
    opt.name = std::move(p);
  }

  // Check type, if present.
  Option::OptType type = Option::OptType::Invalid;
  if (part_this < part_end) {
    auto& p0 = parts[part_this];
    if (p0 == ":") {
      part_this++;  // Only advance if we saw ":".
      if (part_this < part_end) {
        auto& p1 = parts[part_this];
        ICHECK(!p1.empty()) << "tokenizing error";  // This shouldn't happen.
        if (p1 != "=") {
          part_this++;
          if (p1 == "bool") {
            type = Option::OptType::Bool;
          } else if (p1 == "int") {
            type = Option::OptType::Int;
          } else if (p1 == "uint") {
            type = Option::OptType::UInt;
          } else if (p1 == "string") {
            type = Option::OptType::String;
          }
        }
      }
      // If there was ":", there must be a type.
      if (type == Option::OptType::Invalid) {
        LOG(ERROR) << error_header << "invalid type";
        return opt;
      }
    }
  }

  // Check value, if present.
  std::optional<std::string> value;
  if (part_this < part_end) {
    auto& p0 = parts[part_this];
    if (p0 == "=") {
      part_this++;
      if (part_this < part_end) {
        value = std::move(parts[part_this]);
      } else {
        value = "";
      }
    } else {
      // If there are still any parts left to be processed, there must be "=".
      LOG(ERROR) << error_header << "expecting \"=\"";
      return opt;
    }
  }

  // NOLINTNEXTLINE(runtime/int)
  auto to_integer = [](const std::string& s) -> std::optional<long long> {
    // std::stoll takes "long long"
    long long number;  // NOLINT(runtime/int)
    size_t pos;
    try {
      number = std::stoll(s, &pos);
    } catch (...) {
      return std::nullopt;
    }
    if (pos == s.size()) {
      return number;
    } else {
      return std::nullopt;
    }
  };

  auto to_boolean = [&to_integer](const std::string& s) -> std::optional<bool> {
    // Return 0 or 1, if string corresponds to a valid boolean value,
    // otherwise return 2.
    auto ti = to_integer(s);
    if (ti.has_value() && (ti.value() == 0 || ti.value() == 1)) {
      return static_cast<bool>(ti.value());
    }

    std::string lower;
    std::transform(s.begin(), s.end(), std::back_inserter(lower),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower == "true") {
      return true;
    } else if (lower == "false") {
      return false;
    }
    return std::nullopt;
  };

  if (value.has_value()) {
    if (type == Option::OptType::Int || type == Option::OptType::UInt) {
      auto v = to_integer(value.value());
      if (!v.has_value()) {
        LOG(ERROR) << error_header << "invalid integer value \"" << value.value() << "\"";
        return opt;
      }
      if (type == Option::OptType::Int) {
        opt.value.i = static_cast<int>(v.value());
        if (opt.value.i != v.value()) {
          LOG(WARNING) << error_header << "value exceeds int range, assuming " << opt.value.i;
        }
      } else {
        // NOLINTNEXTLINE(runtime/int)
        opt.value.u = static_cast<unsigned>(static_cast<unsigned long long>(v.value()));
        if (opt.value.u != static_cast<unsigned long long>(v.value())) {  // NOLINT(runtime/int)
          LOG(WARNING) << error_header << "value exceeds int range, assuming " << opt.value.u;
        }
      }
    } else if (type == Option::OptType::String) {
      opt.value.s = std::move(value.value());
    } else {
      // "type" is either Bool (given explicitly) or Invalid (type not present in string)
      auto v = to_boolean(value.value());
      if (!v.has_value()) {
        LOG(ERROR) << error_header << "invalid boolean value \"" << value.value() << "\"";
        return opt;
      }
      opt.value.b = v.value();
      type = Option::OptType::Bool;
    }
  } else {
    // Value was not present in string. Assume "true" if "type" is Bool or Invalid
    if (type == Option::OptType::Bool || type == Option::OptType::Invalid) {
      opt.value.b = true;
      type = Option::OptType::Bool;
    } else {
      LOG(ERROR) << error_header << "must have a value";
      return opt;
    }
  }

  ICHECK(type != Option::OptType::Invalid);
  opt.type = type;
  return opt;
}

bool LLVMTargetInfo::MatchesGlobalState() const {
  for (const Option& opt : GetCommandLineOptions()) {
    Option current_opt = opt;
    GetOptionValue(&current_opt);
    ICHECK(current_opt.type != Option::OptType::Invalid);
    switch (current_opt.type) {
      case Option::OptType::Bool:
        if (current_opt.value.b != opt.value.b) return false;
        continue;
      case Option::OptType::Int:
        if (current_opt.value.i != opt.value.i) return false;
        continue;
      case Option::OptType::UInt:
        if (current_opt.value.u != opt.value.u) return false;
        continue;
      case Option::OptType::String:
        if (current_opt.value.s != opt.value.s) return false;
        continue;
      default:;  // NOLINT(whitespace/semicolon)
    }
  }
  return true;
}

void LLVMTargetInfo::GetOptionValue(LLVMTargetInfo::Option* opt) const {
  llvm::StringMap<llvm::cl::Option*>& options = llvm::cl::getRegisteredOptions();
  llvm::cl::Option* base_op = options[opt->name];

  if (opt->type == Option::OptType::Bool) {
    auto* bool_op = static_cast<llvm::cl::opt<bool>*>(base_op);
    opt->value.b = bool_op->getValue();
  } else if (opt->type == Option::OptType::Int) {
    auto* int_op = static_cast<llvm::cl::opt<int>*>(base_op);
    opt->value.i = int_op->getValue();
  } else if (opt->type == Option::OptType::UInt) {
    auto* uint_op = static_cast<llvm::cl::opt<unsigned>*>(base_op);
    opt->value.u = uint_op->getValue();
  } else if (opt->type == Option::OptType::String) {
    auto* str_op = static_cast<llvm::cl::opt<std::string>*>(base_op);
    opt->value.s = str_op->getValue();
  } else {
    opt->type = Option::OptType::Invalid;
  }
}

// LLVMTarget

bool LLVMTarget::modified_llvm_state_ = false;

LLVMTarget::LLVMTarget(LLVMInstance& instance, const LLVMTargetInfo& target_info)
    : LLVMTargetInfo(target_info), instance_(instance), ctx_(instance.GetContext()) {
  // Populate the list of saved options with the current values.
  for (const Option& opt : GetCommandLineOptions()) {
    GetOptionValue(&saved_llvm_options_.emplace_back(opt));
  }

  if (modified_llvm_state_) {
    ICHECK(!ApplyLLVMOptions(true));
  } else {
    modified_llvm_state_ = ApplyLLVMOptions(true);
  }
}

LLVMTarget::LLVMTarget(LLVMInstance& instance, const Target& target)
    : LLVMTarget(instance, LLVMTargetInfo(instance, target)) {}

LLVMTarget::LLVMTarget(LLVMInstance& scope, const std::string& target_str)
    : LLVMTarget(scope, Target(target_str)) {}

LLVMTarget::~LLVMTarget() {
  // Revert all applied LLVM options.
  if (ApplyLLVMOptions(false)) {
    modified_llvm_state_ = false;
  }
}

llvm::LLVMContext* LLVMTarget::GetContext() const {
  ICHECK(!ctx_.expired()) << "LLVM scope has been deleted";
  return ctx_.lock().get();
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

bool LLVMTarget::ApplyLLVMOptions(bool apply_otherwise_revert, bool dry_run) {
  llvm::StringMap<llvm::cl::Option*>& options = llvm::cl::getRegisteredOptions();
  bool changed = false;

#define HANDLE_OPTION_VALUE(option, new_val, saved_val)                  \
  do {                                                                   \
    auto current = (option)->getValue();                                 \
    auto replacement = apply_otherwise_revert ? (new_val) : (saved_val); \
    if (current != replacement) {                                        \
      changed = true;                                                    \
      if (!dry_run) {                                                    \
        (option)->setValue(replacement);                                 \
      }                                                                  \
    }                                                                    \
  } while (false);

  const auto& new_options = GetCommandLineOptions();
  for (size_t i = 0, e = saved_llvm_options_.size(); i != e; ++i) {
    const Option& new_opt = new_options[i];
    const Option& saved_opt = saved_llvm_options_[i];

    llvm::cl::Option* base_op = options[new_opt.name];

    if (new_opt.type == Option::OptType::Bool) {
      auto* bool_op = static_cast<llvm::cl::opt<bool>*>(base_op);
      HANDLE_OPTION_VALUE(bool_op, new_opt.value.b, saved_opt.value.b);
    } else if (new_opt.type == Option::OptType::Int) {
      auto* int_op = static_cast<llvm::cl::opt<int>*>(base_op);
      HANDLE_OPTION_VALUE(int_op, new_opt.value.i, saved_opt.value.i);
    } else if (new_opt.type == Option::OptType::UInt) {
      auto* uint_op = static_cast<llvm::cl::opt<unsigned>*>(base_op);
      HANDLE_OPTION_VALUE(uint_op, new_opt.value.u, saved_opt.value.u);
    } else if (new_opt.type == Option::OptType::String) {
      auto* str_op = static_cast<llvm::cl::opt<std::string>*>(base_op);
      HANDLE_OPTION_VALUE(str_op, new_opt.value.s, saved_opt.value.s);
    } else {
      LOG(FATAL) << "unexpected type in option " << new_opt;
    }

    if (dry_run && changed) {
      return true;
    }
  }

#undef HANDLE_OPTION_VALUE

  return changed;
}

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
