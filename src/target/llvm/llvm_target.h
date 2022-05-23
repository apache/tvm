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

#ifndef TVM_TARGET_LLVM_LLVM_TARGET_H_
#define TVM_TARGET_LLVM_LLVM_TARGET_H_

#ifdef TVM_LLVM_VERSION

#include <llvm/ADT/ArrayRef.h>
#if TVM_LLVM_VERSION >= 150
#include <llvm/IR/FMF.h>
#else
#include <llvm/IR/Operator.h>
#endif
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <tvm/target/target.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

class LLVMTarget {
 public:
  explicit LLVMTarget(const Target& target);
  explicit LLVMTarget(const std::string& target_str);
  ~LLVMTarget();  // Must not be "= default" here in the header file.

  LLVMTarget& operator=(LLVMTarget&& llvm_target);

  std::string str() const;
  std::shared_ptr<llvm::LLVMContext> GetOrCreateContext();
  llvm::TargetMachine* GetOrCreateTargetMachine(bool allow_missing = false);

  const std::string& GetTargetTriple() const { return triple_; }
  const std::string& GetCPU() const { return cpu_; }
  llvm::ArrayRef<std::string> GetTargetFeatures() const { return attrs_; }
  std::string GetTargetFeatureString() const;
  const llvm::TargetOptions& GetTargetOptions() const { return target_options_; }
  llvm::FastMathFlags GetFastMathFlags() const { return fast_math_flags_; }
  llvm::CodeGenOpt::Level GetOptLevel() const { return opt_level_; }

  using ModuleData = std::pair<std::unique_ptr<llvm::Module>, std::unique_ptr<LLVMTarget>>;
  static ModuleData ParseIR(const std::string& llvm_ir,
                            std::shared_ptr<llvm::LLVMContext> ctx = {});
  static ModuleData LoadIR(const std::string& file_name,
                           std::shared_ptr<llvm::LLVMContext> ctx = {});

  static std::string GetTargetMetadata(const llvm::Module& module);
  void SetTargetMetadata(llvm::Module* module) const;

 protected:
  std::shared_ptr<llvm::LLVMContext> ctx_;

  std::string triple_;
  std::string cpu_;
  std::vector<std::string> attrs_;
  llvm::TargetOptions target_options_;
  llvm::FastMathFlags fast_math_flags_;
  llvm::CodeGenOpt::Level opt_level_;
  llvm::Reloc::Model reloc_model_ = llvm::Reloc::PIC_;
  llvm::CodeModel::Model code_model_ = llvm::CodeModel::Small;
  std::unique_ptr<llvm::TargetMachine> target_machine_;

 private:
  LLVMTarget(const std::string& target_str, std::shared_ptr<llvm::LLVMContext> ctx);
  static ModuleData ParseBuffer(const llvm::MemoryBuffer& buffer,
                                std::shared_ptr<llvm::LLVMContext> ctx_or_null);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
#endif  // TVM_TARGET_LLVM_LLVM_TARGET_H_
