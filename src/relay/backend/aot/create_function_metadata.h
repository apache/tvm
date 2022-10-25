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
#ifndef TVM_RELAY_BACKEND_AOT_CREATE_FUNCTION_METADATA_H_
#define TVM_RELAY_BACKEND_AOT_CREATE_FUNCTION_METADATA_H_

#include <tvm/ir/module.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>

#include "../utils.h"

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

/*! \brief Create FunctionInfo metadata for all the PrimFuncs in a module lowered
 *  for AOT execution.
 * \param mod The module.
 * \param workspace_byte_alignment The alignment of the workspace pool.
 * \param constant_byte_alignment The alignment of the constant pool.
 * \return A map between function names and FunctionInfos.
 */
Map<String, FunctionInfo> CreateFunctionMetadata(const IRModule& mod,
                                                 Integer workspace_byte_alignment,
                                                 Integer constant_byte_alignment);

}  // namespace aot
}  // namespace backend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_AOT_CREATE_FUNCTION_METADATA_H_
