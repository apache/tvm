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
 * \file metadata_module.h
 * \brief Declares functions that build MetadataModules for C++ and C runtimes.
 */

#ifndef TVM_TARGET_METADATA_MODULE_H_
#define TVM_TARGET_METADATA_MODULE_H_

#include <tvm/relay/executor.h>
#include <tvm/relay/runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_map>

#include "../relay/backend/utils.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Create a metadata module wrapper. The helper is used by different
 *        codegens, such as graph executor codegen and the vm compiler.
 *
 * \param params The metadata for initialization of all modules.
 * \param target_module the internal module that is compiled by tvm.
 * \param ext_modules The external modules that needs to be imported inside the metadata
 * module(s).
 * \param target The target that all the modules are compiled for
 * \param runtime The runtime to codegen for
 * \param metadata Module metadata
 * \return The created metadata module that manages initialization of metadata.
 */
runtime::Module CreateMetadataModule(
    const std::unordered_map<std::string, runtime::NDArray>& params, runtime::Module target_module,
    const Array<runtime::Module>& ext_modules, Target target, tvm::relay::Runtime runtime,
    tvm::relay::Executor executor, relay::backend::ExecutorCodegenMetadata metadata);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_METADATA_MODULE_H_
