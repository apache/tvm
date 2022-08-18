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
 * \file llvm_module.h
 * \brief Declares top-level shared functions related to the LLVM codegen.
 */

#ifndef TVM_TARGET_LLVM_LLVM_MODULE_H_
#define TVM_TARGET_LLVM_LLVM_MODULE_H_

#ifdef TVM_LLVM_VERSION

#include <tvm/relay/runtime.h>
#include <tvm/runtime/container/array.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>

namespace tvm {
namespace codegen {

runtime::Module CreateLLVMCppMetadataModule(runtime::metadata::Metadata metadata, Target target,
                                            tvm::relay::Runtime runtime);

runtime::Module CreateLLVMCrtMetadataModule(const Array<runtime::Module>& modules, Target target,
                                            tvm::relay::Runtime runtime);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_LLVM_VERSION
#endif  // TVM_TARGET_LLVM_LLVM_MODULE_H_
