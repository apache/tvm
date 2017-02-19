/*!
 *  Copyright (c) 2017 by Contributors
 * \file llvm_common.h
 * \brief Common utilities for llvm initialization.
 */
#ifndef TVM_CODEGEN_LLVM_LLVM_COMMON_H_
#define TVM_CODEGEN_LLVM_LLVM_COMMON_H_
#ifdef TVM_LLVM_VERSION

#include <llvm/ExecutionEngine/MCJIT.h>

#include <llvm/Bitcode/BitcodeWriter.h>

#include <llvm/IR/Value.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/IPO.h>

#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include <utility>
#include <string>


namespace tvm {
namespace codegen {

/*!
 * \brief Initialize LLVM on this process,
 *  can be called multiple times.
 */
void InitializeLLVM();

/*!
 * \brief Get target machine from target_str string.
 * \param target_str Target triple string, can have llvm- prefix, can be empty.
 * \return Pair of target machine and target triple.
 */
std::pair<llvm::TargetMachine*, std::string>
LLVMGetTarget(const std::string& target_str);

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
#endif  // TVM_CODEGEN_LLVM_LLVM_COMMON_H_
