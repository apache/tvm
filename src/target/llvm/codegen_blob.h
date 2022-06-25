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
 * \file codegen_blob.h
 * \brief Code Generation of blob data
 */
#ifndef TVM_TARGET_LLVM_CODEGEN_BLOB_H_
#define TVM_TARGET_LLVM_CODEGEN_BLOB_H_

#ifdef TVM_LLVM_VERSION

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <memory>
#include <string>
#include <utility>

namespace tvm {
namespace codegen {
/**
 * \brief Code Generation of blob data
 *
 * \param data Blob data
 * \param system_lib Whether expose as system library.
 * \param target_triple LLVM target triple
 *
 * \return LLVM module and LLVM context
 */
std::pair<std::unique_ptr<llvm::Module>, std::shared_ptr<llvm::LLVMContext>> CodeGenBlob(
    const std::string& data, bool system_lib, const std::string& llvm_target_string);

}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
#endif  // TVM_TARGET_LLVM_CODEGEN_BLOB_H_
