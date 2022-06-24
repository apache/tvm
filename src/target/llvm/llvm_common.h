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

#ifdef _MSC_VER
#pragma warning(disable : 4141 4291 4146 4624)
#endif
#ifdef TVM_LLVM_VERSION

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/SourceMgr.h>
#if TVM_LLVM_VERSION >= 100
#include <llvm/IR/IntrinsicsAMDGPU.h>
#include <llvm/IR/IntrinsicsARM.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/IntrinsicsX86.h>
#endif
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#if TVM_LLVM_VERSION >= 100
#include <llvm/Support/Alignment.h>
#endif
#include <llvm/CodeGen/TargetLoweringObjectFileImpl.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/MemoryBuffer.h>
#if TVM_LLVM_VERSION >= 140
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <tvm/runtime/container/string.h>

#include <memory>
#include <string>
#include <utility>

namespace tvm {

// The TVM target
class Target;

namespace codegen {

/*!
 * \brief Initialize LLVM on this process,
 *  can be called multiple times.
 */
void InitializeLLVM();

/*!
 * \brief Parse target options
 * \param target The TVM target
 * \param triple Target triple
 * \param mcpu cpu info
 * \param options the options
 * \param mattr The attributes
 */
void ParseLLVMTargetOptions(const Target& target, std::string* triple, std::string* mcpu,
                            std::string* mattr, llvm::TargetOptions* options);

/*!
 * \brief Get target machine from TVM target.
 * \param target The TVM target
 * \param allow_null Whether allow null to be returned.
 * \return target machine
 */
std::unique_ptr<llvm::TargetMachine> GetLLVMTargetMachine(const Target& target,
                                                          bool allow_null = false);

/*!
 * \brief Convert the TVM's LLVM target to string by extracting only relevant fields
 * \param target The TVM target to be extracted
 * \return The raw string format for the TVM LLVM target
 */
std::string LLVMTargetToString(const Target& target);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
#endif  // TVM_TARGET_LLVM_LLVM_COMMON_H_
