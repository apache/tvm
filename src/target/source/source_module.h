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
 * \file source_module.h
 * \brief Source code module
 */

#ifndef TVM_TARGET_SOURCE_SOURCE_MODULE_H_
#define TVM_TARGET_SOURCE_SOURCE_MODULE_H_

#include <tvm/relay/runtime.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/module.h>
#include <tvm/target/target.h>

#include "../../relay/backend/utils.h"
#include "../../runtime/meta_data.h"

namespace tvm {
namespace codegen {

/*!

 * \brief Wrap the submodules that are to be wrapped in a c-source metadata module for C runtime.
 * \param modules The modules to be wrapped.
 * \param target the target the modules are compiled for.
 * \param runtime the runtime to code generate against
 * \param metadata Compiler-generated metadata exported to runtime.
 * \param aot_metadata If supplied, metadata for the AOTExecutor module.
 * \return The wrapped module.
 */
runtime::Module CreateCSourceCrtMetadataModule(const Array<runtime::Module>& modules, Target target,
                                               relay::Runtime runtime,
                                               relay::backend::ExecutorCodegenMetadata metadata,
                                               runtime::metadata::Metadata aot_metadata);

/*!
 * \brief Create C++-runtime targeted metadata module for "c" backend.
 * \param metadata Compiler-generated metadata.
 */
runtime::Module CreateCSourceCppMetadataModule(runtime::metadata::Metadata metadata);

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_SOURCE_MODULE_H_
