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
#ifndef TVM_RELAY_BACKEND_AOT_AOT_LOWER_MAIN_H_
#define TVM_RELAY_BACKEND_AOT_AOT_LOWER_MAIN_H_

#include <tvm/ir/transform.h>
#include <tvm/target/compilation_config.h>

#include <tuple>
#include <unordered_map>
#include <vector>

#include "../utils.h"

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

using StorageMap =
    std::unordered_map<Expr, StorageInfo, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

/*! \brief Exposed for testing, part of the implementation of AOTLowerMain */
std::tuple<StorageMap, std::vector<int>> CreateStorage(const Function& func);

/*! \brief Lower the Relay main function into TIR for use with the AOT executor.
 *
 * This pass expects that all operators have already been lowered to TIR and
 * so only Calls to 'call_lowered' are present in main.
 *
 * \param mod_name The name of the module.
 * \param config The compilation config.
 * \param call_type The call type to use when calling functions.
 */
transform::Pass AOTLowerMain(String mod_name, tvm::CompilationConfig config, CallType call_type);

}  // namespace aot
}  // namespace backend
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_AOT_AOT_LOWER_MAIN_H_
