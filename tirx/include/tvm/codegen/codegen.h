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
 * \file tvm/codegen/codegen.h
 * \brief Shared module build dispatch and imported-module packing.
 */
#ifndef TVM_CODEGEN_CODEGEN_H_
#define TVM_CODEGEN_CODEGEN_H_

#include <tvm/ffi/extra/module.h>
#include <tvm/ir/module.h>
#include <tvm/runtime/base.h>
#include <tvm/target/target.h>

#include <string>

namespace tvm {
/*! \brief namespace for target translation and codegen. */
namespace codegen {

/*!
 * \brief Build a runtime module from an IRModule for the requested target.
 * \param mod The Module to be built
 * \param target The target to be built.
 * \return The generated module.
 */
TVM_TIRX_DLL ffi::Module Build(IRModule mod, Target target);

/*!
 * \brief Pack imported device library to a C file.
 *  Compile the C file and link with the host library
 *  will allow the DSO loader to automatically discover and import
 *  the dependency from the shared library.
 *
 * \param m The host module with the imports.
 * \param system_lib Whether expose as system library.
 * \param c_symbol_prefix Optional symbol prefix of the blob symbol.
 * \return cstr The C string representation of the file.
 */
TVM_TIRX_DLL std::string PackImportsToC(const ffi::Module& m, bool system_lib,
                                        const std::string& c_symbol_prefix = "");

/*!
 * \brief Pack imported device libraries into an object file using LLVM.
 *
 * The generated object can be linked with the host library so that the DSO
 * loader discovers its imported modules. Requires LLVM code generation support.
 *
 * \param mod The host module with imports.
 * \param system_lib Whether to expose the packed imports as a system library.
 * \param llvm_target_string Full LLVM target specification, including cross-compilation options.
 * \param file_name Output path; its extension selects the object format.
 * \param c_symbol_prefix Optional blob symbol prefix, valid only for system libraries.
 */
TVM_TIRX_DLL void PackImportsToObjectFile(const ffi::Module& mod, bool system_lib,
                                          const std::string& llvm_target_string,
                                          const std::string& file_name,
                                          const std::string& c_symbol_prefix = "");

namespace attr {
inline constexpr ffi::reflection::AttrKey<bool> kParallelStridePattern{
    "codegen.parallel_stride_pattern"};
}  // namespace attr

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_CODEGEN_CODEGEN_H_
