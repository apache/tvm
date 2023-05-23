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
 * \file build_vulkan.cc
 * \brief Build SPIRV block
 */

#include "../../runtime/spirv/spirv_shader.h"
#include "../../runtime/vulkan/vulkan_module.h"
#include "../build_common.h"
#include "spirv_utils.h"

namespace tvm {
namespace codegen {

runtime::Module BuildSPIRV(IRModule mod, Target target) {
  auto [smap, spirv_text] = LowerToSPIRV(mod, target);
  return runtime::VulkanModuleCreate(smap, ExtractFuncInfo(mod), spirv_text);
}

TVM_REGISTER_GLOBAL("target.build.vulkan").set_body_typed([](IRModule mod, Target target) {
  return BuildSPIRV(mod, target);
});

}  // namespace codegen
}  // namespace tvm
