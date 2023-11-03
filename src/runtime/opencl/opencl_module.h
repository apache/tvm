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
 * \file opencl_module.h
 * \brief Execution handling of OPENCL kernels
 */
#ifndef TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
#define TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_

#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../meta_data.h"
#include "../spirv/spirv_shader.h"

namespace tvm {
namespace runtime {
/*!
 * \brief Create a opencl module for GPU devices from data.
 *
 * \param data The module data.
 * \param fmt The format of the data, can be "clbin", "cl"
 * \param fmap The map function information map of each function.
 * \param source Generated OpenCL kernels.
 */
Module OpenCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source);

/*!
 * \brief Create a opencl module from SPIRV.
 *
 * \param shaders The map from function names to SPIRV binaries.
 * \param spirv_text The concatenated text representation of SPIRV modules.
 * \param fmap The map function information map of each function.
 */
Module OpenCLModuleCreate(const std::unordered_map<std::string, spirv::SPIRVShader>& shaders,
                          const std::string& spirv_text,
                          std::unordered_map<std::string, FunctionInfo> fmap);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENCL_OPENCL_MODULE_H_
