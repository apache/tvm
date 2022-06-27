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

#include <tvm/runtime/container/string.h>

#include <memory>
#include <string>
#include <utility>

namespace llvm {
class Module;
class Target;
class TargetMachine;
class TargetOptions;
}  // namespace llvm

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
