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

/*! \file llvm_instance.h
 */
#ifndef TVM_TARGET_LLVM_LLVM_INSTANCE_H_
#define TVM_TARGET_LLVM_LLVM_INSTANCE_H_

#ifdef TVM_LLVM_VERSION

#include <llvm/ADT/ArrayRef.h>
#if TVM_LLVM_VERSION >= 150
#include <llvm/IR/FMF.h>
#else
#include <llvm/IR/Operator.h>
#endif
#include <llvm/Support/CodeGen.h>
#include <llvm/Target/TargetOptions.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/runtime/container/string.h>
#include <tvm/target/target.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// LLVM compatibility macro
#if TVM_LLVM_VERSION >= 200
#define llvmGetPointerTo(arg, offset) (llvm::PointerType::get((arg), (offset)))
#else
#define llvmGetPointerTo(arg, offset) (arg->getPointerTo(offset))
#endif

namespace llvm {
class LLVMContext;
class MemoryBuffer;
class Module;
class TargetMachine;
}  // namespace llvm

namespace tvm {
namespace codegen {

class LLVMTarget;

/*!
 * \class LLVMInstance
 * \brief LLVMInstance is a class that (conceptually) starts and stops LLVM.
 *        All uses of LLVM should take place within a lifetime of an object
 *        of this class.
 *
 * E.g.
 * ```{.cpp}
 * {
 *   LLVMInstance llvm_instance;
 *   ...
 *   someFunctionFromLLVM(...);
 *   ...
 * }
 * // no more calls to LLVM here
 * ```
 * In addition to that, LLVMInstance provides an LLVM context (llvm::LLVMContext).
 * The context is a structure in LLVM where common IR constructs are maintained,
 * (such as types, constants, etc.) so that they can be identified by their
 * address (i.e. pointer comparison). Because of that, it's important to use
 * the same context throughout compilation.
 *
 * At the moment the "starting" of LLVM performs initialization of LLVM, but
 * "stopping" doesn't do anything. In the future, if such a need arises, this
 * functionality may be extended to perform dlopen/dlclose of the LLVM-based
 * code in TVM.
 *
 * This class provides means to deserialize an LLVM module, either from text
 * (in a string), or from a file. In either case, the serialized module can
 * be LLVM IR assembly, or binary bitcode enconding.
 */
class LLVMInstance {
 public:
  /*!
   * \brief Constructs LLVMInstance
   */
  LLVMInstance();
  /*!
   * \brief Destroys LLVMInstance object
   */
  ~LLVMInstance();  // Must not be "= default" here in the header file.

  /*!
   * \brief Get the LLVM context for this scope.
   */
  std::shared_ptr<llvm::LLVMContext> GetContext() const { return ctx_; }

  /*!
   * \brief Create `llvm::Module` from a string.
   *
   * Parse the string in \param llvm_ir, and return the `llvm::Module`.
   * At the moment this function will abort if the parsing fails.
   * \param llvm_ir string with the LLVM IR assembly or bitcode
   * \return created `llvm::Module`
   */
  std::unique_ptr<llvm::Module> ParseIR(const std::string& llvm_ir) const;
  /*!
   * \brief Load `llvm::Module` from a given file
   *
   * Read the file \param file_name, and return the `llvm::Module`.
   * At the moment this function will abort if reading of the file or creation
   * of the module fails.
   * \param file_name file with the LLVM IR assembly or bitcode
   * \return created `llvm::Module`
   */
  std::unique_ptr<llvm::Module> LoadIR(const std::string& file_name) const;

 private:
  std::unique_ptr<llvm::Module> ParseBuffer(const llvm::MemoryBuffer& buffer) const;

  std::shared_ptr<llvm::LLVMContext> ctx_;
};

/*!
 * \class LLVMTargetInfo
 * \brief Summary of information for this TVM target relevant to LLVM code
 *        generation.
 *
 * This class contains all information that LLVM needs for code generation for
 * a particular target. The purpose of this class is only to provide information
 * in an easily-accessible form (for example for querying the target properties).
 *
 * Note that objects of this class must be created within the lifetime of an
 * LLVMInstance object.
 */
class LLVMTargetInfo {
 public:
  /*!
   * \brief Constructs LLVMTargetInfo from `Target`
   * \param scope LLVMInstance object
   * \param target TVM Target object for target "llvm"
   */
  LLVMTargetInfo(LLVMInstance& scope, const Target& target);  // NOLINT(runtime/references)
  /*!
   * \brief Constructs LLVMTargetInfo from target string
   * \param scope LLVMInstance object
   * \param target TVM target string for target "llvm"
   */
  // NOLINTNEXTLINE(runtime/references)
  LLVMTargetInfo(LLVMInstance& scope, const std::string& target_str);
  /*!
   * \brief Constructs LLVMTargetInfo from `Target`
   * \param scope LLVMInstance object
   * \param target TVM JSON Target object for target "llvm"
   */
  // NOLINTNEXTLINE(runtime/references)
  LLVMTargetInfo(LLVMInstance& instance, const TargetJSON& target);

  /*!
   * \brief Destroys LLVMTargetInfo object
   */
  ~LLVMTargetInfo();

  /*!
   * \brief Returns string representation (as TVM target) of the LLVMTargetInfo
   * \return Target string
   *
   * Note: If the LLVMTargetInfo object was created from a string `s`, the string
   * returned here may not be exactly equal to `s`. For example, if the CPU
   * was "default", the returned string will have CPU set to the detected host
   * CPU.
   */
  std::string str() const;

  /*!
   * \brief Return LLVM's `TargetMachine`, or nullptr
   * \param allow_missing do not abort if the target machine cannot be created,
   *        return nullptr instead
   * \return Pointer to the `TargetMachine` object (or nullptr if it cannot be
   *         created, \see allow_missing)
   */
  llvm::TargetMachine* GetOrCreateTargetMachine(bool allow_missing = false);

  /*!
   * \brief Get the target triple
   * \return the target triple
   */
  const std::string& GetTargetTriple() const { return triple_; }
  /*!
   * \brief Get the CPU name
   * \return the CPU name: the detected host CPU if the original TVM target
   *         specified it as "default"
   */
  const std::string& GetCPU() const { return cpu_; }
  /*!
   * \brief Get the list of LLVM target features
   * \return array of individual feature strings
   */
  llvm::ArrayRef<std::string> GetTargetFeatures() const { return attrs_; }
  /*!
   * \brief Get the LLVM target feature string
   * \return comma-separated list of LLVM target features
   */
  std::string GetTargetFeatureString() const;
  /*!
   * \brief Get the LLVM target options
   * \return `llvm::TargetOptions` object for this target
   */
  const llvm::TargetOptions& GetTargetOptions() const { return target_options_; }
  /*!
   * \brief Get the LLVM target reloc model
   * \return `llvm::Reloc::Model` object for this target
   */
  const llvm::Reloc::Model& GetTargetRelocModel() const { return reloc_model_; }
  /*!
   * \brief Get the LLVM target code model
   * \return `llvm::CodeModel::Model` object for this target
   */
  const llvm::CodeModel::Model& GetTargetCodeModel() const { return code_model_; }
  /*!
   * \brief Get fast math flags
   * \return `llvm::FastMathFlags` for this target
   */
  llvm::FastMathFlags GetFastMathFlags() const { return fast_math_flags_; }
  /*!
   * \brief Get the LLVM JIT engine type
   * \return the type name of the JIT engine (default "orcjit" or "mcjit")
   */
  const std::string GetJITEngine() const { return jit_engine_; }
  /*!
   * \brief Get the LLVM optimization level
   * \return optimization level for this target
   */
#if TVM_LLVM_VERSION <= 170
  llvm::CodeGenOpt::Level GetOptLevel() const { return opt_level_; }
#else
  llvm::CodeGenOptLevel GetOptLevel() const { return opt_level_; }
#endif

  /*!
   * \class Option
   * \brief Internal representation of command-line option
   */
  struct Option {
    enum class OptType {
      Invalid = 0,  //!< placeholder, indicates parsing error
      Bool,         //!< enum value corresponding to type string "bool"
      Int,          //!< enum value corresponding to type string "int"
      UInt,         //!< enum value corresponding to type string "uint"
      String,       //!< enum value corresponding to type string "string"
    };
    std::string name;  //!< option name
    OptType type;      //!< type of the option value
    struct {
      union {
        bool b;          //!< bool option value
        int i;           //!< int option value
        unsigned u = 0;  //!< unsigned option value
      };
      std::string s;  //!< string option value
    } value;          //!< option value specified in the option string
  };

  /*!
   * \brief Get LLVM command line options
   * \return the list of LLVM command line options specified for this target
   */
  const std::vector<Option>& GetCommandLineOptions() const { return llvm_options_; }

  /*!
   * \brief Parse a string from the `cl-opt` target attribute
   * \param str the option string
   * \return parsed `Option` object, if parsing failed the type member will be
   *         set to `Option::OptType::Invalid`
   */
  static Option ParseOptionString(const std::string& str);

  /*!
   * \brief Checks if the settings in this object that describe global state
   *        match the current global state
   * \return true or false correspondingly
   * \note The global state can be modified by command line options. This
   *       function checks if the specified options differ from their current
   *       values.
   */
  bool MatchesGlobalState() const;

  /*!
   * \brief Get all supported targets from the LLVM backend
   * \return list with all valid targets
   */
  const Array<String> GetAllLLVMTargets() const;

  /*!
   * \brief Get all CPU arches from target
   * \return list with all valid cpu architectures
   * \note The arches are fetched from the LLVM backend using the target `-mtriple`.
   */
  const Array<String> GetAllLLVMTargetArches() const;

  /*!
   * \brief Get all CPU features from target
   * \return Map with all valid cpu features as keys and empty string as value. The Map
   *         is intended to be used as a Set, which TVM does not currently support.
   * \note The features are fetched from the LLVM backend using the target `-mtriple`
   *       and the `-mcpu` architecture, but also consider the `-mattr` attributes.
   */
  const Map<String, String> GetAllLLVMCpuFeatures() const;

  /*!
   * \brief Check the target if has a specific cpu feature
   * \param feature string with the feature to check
   * \return true or false
   * \note The feature is checked in the LLVM backend for the target `-mtriple`
   *       and `-mcpu` architecture, but also consider the `-mattr` attributes.
   */
  const bool TargetHasCPUFeature(const std::string& feature) const;

 protected:
  /*!
   * \brief Get the current value of given LLVM option
   * \param opt Option with "type" and "name" set
   * Fills in the "value" field in the provided Option argument, or sets the
   * "type" to Invalid if the option value cannot be obtained.
   */
  void GetOptionValue(Option* opt) const;

 private:
  std::string triple_;
  std::string cpu_;
  std::vector<std::string> attrs_;
  std::vector<Option> llvm_options_;
  llvm::TargetOptions target_options_;
  llvm::FastMathFlags fast_math_flags_;
#if TVM_LLVM_VERSION <= 170
  llvm::CodeGenOpt::Level opt_level_;
#else
  llvm::CodeGenOptLevel opt_level_;
#endif
  llvm::Reloc::Model reloc_model_ = llvm::Reloc::PIC_;
  llvm::CodeModel::Model code_model_ = llvm::CodeModel::Small;
  std::shared_ptr<llvm::TargetMachine> target_machine_;
  std::string jit_engine_ = "orcjit";
};

/*!
 * \class LLVMTarget
 * \brief Information used by LLVM for code generation for particular target
 *
 * In addition to all information that LLVM needs for code generation for
 * a particular target, objects of this class handle saving and restoring
 * global LLVM state that may be affected by these flags. This way, code
 * generation for each LLVM-based target in TVM will start with the same LLVM
 * global state.
 *
 * Note that objects of this class must be created within the lifetime of an
 * LLVMInstance object.
 */
class LLVMTarget : public LLVMTargetInfo {
 public:
  /*!
   * \brief Constructs LLVMTarget from `Target`
   * \param scope LLVMInstance object
   * \param target_info Target info object for target "llvm"
   */
  LLVMTarget(LLVMInstance& scope, const LLVMTargetInfo& target_info);  // NOLINT(runtime/references)
  /*!
   * \brief Constructs LLVMTarget from `Target`
   * \param scope LLVMInstance object
   * \param target TVM Target object for target "llvm"
   */
  LLVMTarget(LLVMInstance& scope, const Target& target);  // NOLINT(runtime/references)
  /*!
   * \brief Constructs LLVMTarget from target string
   * \param scope LLVMInstance object
   * \param target TVM target string for target "llvm"
   */
  LLVMTarget(LLVMInstance& scope, const std::string& target_str);  // NOLINT(runtime/references)
  /*!
   * \brief Destroys LLVMTarget object
   */
  ~LLVMTarget();

  /*!
   * \brief Get the LLVMInstance object from which the LLVMTarget object was
   *        created
   * \return The enclosing LLVMInstance object
   */
  const LLVMInstance& GetInstance() const { return instance_; }
  /*!
   * \brief Get the current LLVM context
   * \return the current LLVM context
   */
  llvm::LLVMContext* GetContext() const;
  /*!
   * \brief Extract the target string from given `llvm::Module`
   * \param module LLVM module with the TVM target string embedded as metadata
   * \return the target string from module's metadata
   */
  static std::string GetTargetMetadata(const llvm::Module& module);
  /*!
   * \brief Embed target string as metadata in given `llvm::Module`
   * \param module the module to insert the target string into
   */
  void SetTargetMetadata(llvm::Module* module) const;

  // Stubs to enable use with `With`.
  void EnterWithScope() {}
  void ExitWithScope() {}

 private:
  std::vector<Option> saved_llvm_options_;

  /*!
   * \brief Apply or revert command-line LLVM options
   * \param apply_otherwise_revert if true, apply the options (saving previous
   *        values, if false, then restore the saved values
   * \param dry_run if true, do not make any changes (or save anything)
   * \return true is changes were made (or would have been made in a dry run),
   *         false otherwise
   */
  bool ApplyLLVMOptions(bool apply_otherwise_revert, bool dry_run = false);

  const LLVMInstance& instance_;
  std::weak_ptr<llvm::LLVMContext> ctx_;

  /*!
   * \brief Global singleton flag indicating whether LLVM's global state has
   *        been modified or not (via command-line flags). There can only be
   *        a single such modification in effect at any given time.
   */
  static bool modified_llvm_state_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
#endif  // TVM_TARGET_LLVM_LLVM_INSTANCE_H_
