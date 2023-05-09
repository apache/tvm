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
#ifndef TVM_TARGET_SPIRV_SPIRV_UTILS_H_
#define TVM_TARGET_SPIRV_SPIRV_UTILS_H_

#include <tvm/ir/module.h>
#include <tvm/target/target.h>

#include <string>
#include <unordered_map>
#include <utility>

#include "../../runtime/spirv/spirv_shader.h"

namespace tvm {
namespace codegen {
/*!
 * \brief Lower an IRModule to SPIRV modules.
 *
 * \param mod The IRModule to lower.
 * \param target The target information.
 * \return The map from function names to SPIRV binaries, and the concatenated text representation
 * of the SPIRV modules.
 */
std::pair<std::unordered_map<std::string, runtime::SPIRVShader>, std::string> LowerToSPIRV(
    IRModule mod, Target target);

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_TARGET_SPIRV_SPIRV_UTILS_H_
