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
 * \file codegen_blob.cc
 */
#ifdef TVM_LLVM_VERSION
#include "codegen_blob.h"

#include <tvm/runtime/module.h>
#include <tvm/target/target.h>

#include <cstring>

namespace tvm {
namespace codegen {

std::pair<std::unique_ptr<llvm::Module>, std::shared_ptr<llvm::LLVMContext>> CodeGenBlob(
    const std::string& data, bool system_lib, const std::string& llvm_target_string) {
  InitializeLLVM();
  Target target(llvm_target_string);
  auto tm = GetLLVMTargetMachine(target);
  auto triple = tm->getTargetTriple();
  auto ctx = std::make_shared<llvm::LLVMContext>();
  std::string module_name = "devc";
  std::unique_ptr<llvm::Module> module(new llvm::Module(module_name, *ctx));
  module->setTargetTriple(triple.str());
  // Store full target string in metadata, because flags such as -mfloat-abi must be preserved for
  // ModulePackImportsToLLVM.
  module->addModuleFlag(llvm::Module::ModFlagBehavior::Override, "tvm_target",
                        llvm::MDString::get(*ctx, LLVMTargetToString(target)));
  module->setDataLayout(tm->createDataLayout());
  auto* blob_value = llvm::ConstantDataArray::getString(*ctx, data, false);
  auto* tvm_dev_mblob = new llvm::GlobalVariable(
      *module, blob_value->getType(), true, llvm::GlobalValue::ExternalLinkage, blob_value,
      runtime::symbol::tvm_dev_mblob, nullptr, llvm::GlobalVariable::NotThreadLocal, 0);

  // If large const data (>2GB) is saved to default .rodata section
  // then linking it to shared library will fail - relocation truncated to fit: R_X86_64_PC32.
  // The issue exists on Linux x86_64 platform.
  // GCC handles this situation by using -mcmodel=medium parameter but LLVM ignores it.
  // The workaround is to explicitly put large const data to .lrodata section.
  // Lets put const data which is larger than 1GB to .lrodata section
  const size_t large_data_threshold = 1 << 30;
  if (data.size() > large_data_threshold && triple.getArch() == llvm::Triple::x86_64 &&
      triple.isOSBinFormatELF()) {
    tvm_dev_mblob->setSection(".lrodata");
  }

#if TVM_LLVM_VERSION >= 100
  tvm_dev_mblob->setAlignment(llvm::Align(1));
#else
  tvm_dev_mblob->setAlignment(1);
#endif

  if (triple.isOSWindows()) {
    tvm_dev_mblob->setDLLStorageClass(llvm::GlobalVariable::DLLExportStorageClass);
  }

  if (system_lib) {
    // LLVM type helper
    auto void_ty = llvm::Type::getVoidTy(*ctx);
    auto int32_ty = llvm::Type::getInt32Ty(*ctx);
    auto int8_ty = llvm::Type::getInt8Ty(*ctx);
    auto int8_ptr_ty = int8_ty->getPointerTo(0);

    llvm::Constant* constant_zero = llvm::Constant::getNullValue(int32_ty);
    auto* tvm_dev_mblob_reg = new llvm::GlobalVariable(
        *module, int32_ty, false, llvm::GlobalValue::InternalLinkage, constant_zero,
        std::string(runtime::symbol::tvm_dev_mblob) + "_reg_");
    auto tvm_dev_mblob_reg_alignment = module->getDataLayout().getABITypeAlignment(int32_ty);
#if TVM_LLVM_VERSION >= 100
    tvm_dev_mblob_reg->setAlignment(llvm::Align(tvm_dev_mblob_reg_alignment));
#else
    tvm_dev_mblob_reg->setAlignment(tvm_dev_mblob_reg_alignment);
#endif

    auto* tvm_dev_mblob_string_ty =
        llvm::ArrayType::get(int8_ty, std::strlen(runtime::symbol::tvm_dev_mblob) + 1);
    auto* tvm_dev_mblob_string_value =
        llvm::ConstantDataArray::getString(*ctx, runtime::symbol::tvm_dev_mblob, true);
    auto* tvm_dev_mblob_string = new llvm::GlobalVariable(
        *module, tvm_dev_mblob_string_ty, true, llvm::GlobalValue::PrivateLinkage,
        tvm_dev_mblob_string_value, std::string(runtime::symbol::tvm_dev_mblob) + ".str");
#if TVM_LLVM_VERSION >= 100
    tvm_dev_mblob_string->setAlignment(llvm::Align(1));
#else
    tvm_dev_mblob_string->setAlignment(1);
#endif

    // Global init function
    llvm::Function* init_fn = llvm::Function::Create(
        llvm::FunctionType::get(void_ty, false), llvm::GlobalValue::InternalLinkage,
        llvm::Twine("_GLOBAL__sub_I_", module_name), module.get());

    // Create variable initialization function.
    llvm::Function* var_init_fn = llvm::Function::Create(
        llvm::FunctionType::get(void_ty, false), llvm::GlobalValue::InternalLinkage,
        llvm::Twine("__cxx_global_var_init"), module.get());

    // Create TVMBackendRegisterSystemLibSymbol function
    llvm::Function* tvm_backend_fn =
        llvm::Function::Create(llvm::FunctionType::get(int32_ty, {int8_ptr_ty, int8_ptr_ty}, false),
                               llvm::GlobalValue::ExternalLinkage,
                               llvm::Twine("TVMBackendRegisterSystemLibSymbol"), module.get());

    // Set necessary fn sections
    auto get_static_init_section_specifier = [&triple]() -> std::string {
      if (triple.isOSLinux()) {
        return ".text.startup";
      } else if (triple.isOSDarwin()) {
        return "__TEXT,__StaticInit,regular,pure_instructions";
      } else {
        return "";
      }
    };

    auto static_init_section_specifier = get_static_init_section_specifier();

    if (!static_init_section_specifier.empty()) {
      init_fn->setSection(static_init_section_specifier);
      var_init_fn->setSection(static_init_section_specifier);
    }

    // The priority is 65535 for all platforms as clang do.
    llvm::appendToGlobalCtors(*module, init_fn, 65535);

    // Define init_fn body
    llvm::IRBuilder<> ir_builder(*ctx);
    llvm::BasicBlock* init_fn_bb = llvm::BasicBlock::Create(*ctx, "entry", init_fn);
    ir_builder.SetInsertPoint(init_fn_bb);
    ir_builder.CreateCall(var_init_fn);
    ir_builder.CreateRetVoid();

    // Define var_init_fn body
    llvm::BasicBlock* var_init_fn_bb = llvm::BasicBlock::Create(*ctx, "entry", var_init_fn);
    ir_builder.SetInsertPoint(var_init_fn_bb);
    llvm::Constant* indices[] = {constant_zero, constant_zero};
    llvm::SmallVector<llvm::Value*, 2> args;
    args.push_back(llvm::ConstantExpr::getGetElementPtr(tvm_dev_mblob_string_ty,
                                                        tvm_dev_mblob_string, indices));
    args.push_back(
        llvm::ConstantExpr::getGetElementPtr(blob_value->getType(), tvm_dev_mblob, indices));
    auto* tvm_backend_fn_ret_value = ir_builder.CreateCall(tvm_backend_fn, args);
    ir_builder.CreateStore(tvm_backend_fn_ret_value, tvm_dev_mblob_reg);
    ir_builder.CreateRetVoid();
  }

  return std::make_pair(std::move(module), ctx);
}

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
