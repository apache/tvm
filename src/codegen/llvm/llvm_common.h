/*!
 *  Copyright (c) 2017 by Contributors
 * \file llvm_common.h
 * \brief Common utilities for llvm initialization.
 */
#ifndef TVM_CODEGEN_LLVM_LLVM_COMMON_H_
#define TVM_CODEGEN_LLVM_LLVM_COMMON_H_
#ifdef TVM_LLVM_VERSION

#include <llvm/ExecutionEngine/MCJIT.h>

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

#include <llvm/Support/Casting.h>
#include <llvm/Support/TargetSelect.h>

extern "C" {
// Function signature for LLVM generated packed function.
typedef int (*LLVMPackedCFunc)(void* args,
                               int* type_codes,
                               int num_args);
}  // extern "C"

namespace tvm {
namespace codegen {

/*!
 * \brief Initialize LLVM on this process,
 *  can be called multiple times.
 */
void InitializeLLVM();

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
#endif  // TVM_CODEGEN_LLVM_LLVM_COMMON_H_
