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
 * \file tvm/target/codegen.h
 * \brief Translates IRModule to runtime::Module.
 */
#ifndef TVM_TARGET_CODEGEN_H_
#define TVM_TARGET_CODEGEN_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>

#include <string>

namespace tvm {
/*! \brief namespace for target translation and codegen. */
namespace codegen {
// use packed function from runtime.
using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

/*!
 * \brief Build a module from array of lowered function.
 * \param mod The Module to be built
 * \param target The target to be built.
 * \return The result runtime::Module.
 */
runtime::Module Codegen(IRModule mod, Target target);

/*!
 * \brief Pack imported device library to a C file.
 *  Compile the C file and link with the host library
 *  will allow the DSO loader to automatically discover and import
 *  the dependency from the shared library.
 *
 * \param m The host module with the imports.
 * \param system_lib Whether expose as system library.
 * \return cstr The C string representation of the file.
 */
std::string PackImportsToC(const runtime::Module& m, bool system_lib);

/*!
 * \brief Pack imported device library to a LLVM module.
 *  Compile the LLVM module and link with the host library
 *  will allow the DSO loader to automatically discover and import
 *  the dependency from the shared library.
 *
 * \param m The host module with the imports.
 * \param system_lib Whether expose as system library.
 * \param target_triple LLVM target triple
 * \return runtime::Module The generated LLVM module.
 */
runtime::Module PackImportsToLLVM(const runtime::Module& m, bool system_lib,
                                  const std::string& target_triple);
}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_CODEGEN_H_
