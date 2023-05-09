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
 *  Optional module when build opencl is switched to off
 */
#include "../../runtime/opencl/opencl_module.h"
#include "../source/codegen_source_base.h"

namespace tvm {
namespace runtime {

Module OpenCLModuleCreate(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap, std::string source) {
  return codegen::DeviceSourceModuleCreate(data, fmt, fmap, "opencl");
}

Module OpenCLModuleCreate(const std::unordered_map<std::string, SPIRVShader>& shaders,
                          const std::string& spirv_text,
                          std::unordered_map<std::string, FunctionInfo> fmap) {
  LOG(FATAL) << "OpenCLModuleCreate is called but OpenCL is not enabled.";
  return Module();
}

}  // namespace runtime
}  // namespace tvm
