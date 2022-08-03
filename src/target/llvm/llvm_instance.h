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

#include <memory>
#include <string>
#include <utility>
#include <vector>

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
 * \brief LLVMInstance is a class that (conceptually) starts and stops LLVM. All
 * uses of LLVM should take place within a lifetime of an object of this class.
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
 * \class LLVMTarget
 * \brief Information used by LLVM for code generation for particular target
 *
 * This class contains all information that LLVM needs for code generation for
 * a particular target. Since Target in TVM will soon contain command line
 * flags for LLVM, objects of this class will handle saving and restoring
 * global LLVM state that may be affected by these flags. This way, code
 * generation for each LLVM-based target in TVM will start with the same LLVM
 * global state.
 *
 * Note that objects of this class must be created within the lifetime of an
 * LLVMInstance object.
 */
class LLVMTarget {
 public:
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
   * \brief Returns string representation (as TVM target) of the LLVMTarget
   * \return Target string
   *
   * Note: If the LLVMTarget object was created from a string `s`, the string
   * returned here may not be exactly equal to `s`. For example, if the CPU
   * was "default", the returned string will have CPU set to the detected host
   * CPU.
   */
  std::string str() const;

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
   * \brief Get fast math flags
   * \return `llvm::FastMathFlags` for this target
   */
  llvm::FastMathFlags GetFastMathFlags() const { return fast_math_flags_; }
  /*!
   * \brief Get the LLVM optimization level
   * \return optimization level for this target
   */
  llvm::CodeGenOpt::Level GetOptLevel() const { return opt_level_; }

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
  const LLVMInstance& instance_;
  std::weak_ptr<llvm::LLVMContext> ctx_;

  std::string triple_;
  std::string cpu_;
  std::vector<std::string> attrs_;
  llvm::TargetOptions target_options_;
  llvm::FastMathFlags fast_math_flags_;
  llvm::CodeGenOpt::Level opt_level_;
  llvm::Reloc::Model reloc_model_ = llvm::Reloc::PIC_;
  llvm::CodeModel::Model code_model_ = llvm::CodeModel::Small;
  std::shared_ptr<llvm::TargetMachine> target_machine_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
#endif  // TVM_TARGET_LLVM_LLVM_INSTANCE_H_
