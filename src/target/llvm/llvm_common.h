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
 * \file llvm_common.h
 * \brief Common utilities for llvm initialization.
 */
#ifndef TVM_TARGET_LLVM_LLVM_COMMON_H_
#define TVM_TARGET_LLVM_LLVM_COMMON_H_
#ifdef TVM_LLVM_VERSION

#include <llvm/ExecutionEngine/MCJIT.h>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/SourceMgr.h>

#include <llvm/IR/Value.h>
#include <llvm/IR/Intrinsics.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/IntrinsicsARM.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/IntrinsicsX86.h>
#endif
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Verifier.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>

#if TVM_LLVM_VERSION >= 100
#include <llvm/Support/Alignment.h>
#endif
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/CodeGen/TargetLoweringObjectFileImpl.h>

#include <llvm/Linker/Linker.h>

#include <utility>
#include <string>
#include <memory>

namespace tvm {
namespace codegen {

/*!
 * \brief Initialize LLVM on this process,
 *  can be called multiple times.
 */
void InitializeLLVM();

/*!
 * \brief Parse target options
 * \param target_str Target string, in format "llvm -target=xxx -mcpu=xxx"
 * \param triple Target triple
 * \param mcpu cpu info
 * \param options the options
 * \param mattr The attributes
 */
void ParseLLVMTargetOptions(const std::string& target_str,
                            std::string* triple,
                            std::string* mcpu,
                            std::string* mattr,
                            llvm::TargetOptions* options);

/*!
 * \brief Get target machine from target_str string.
 * \param target_str Target string, in format "llvm -target=xxx -mcpu=xxx"
 * \param allow_null Whether allow null to be returned.
 * \return target machine
 */
std::unique_ptr<llvm::TargetMachine>
GetLLVMTargetMachine(const std::string& target_str, bool allow_null = false);

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
#endif  // TVM_TARGET_LLVM_LLVM_COMMON_H_
