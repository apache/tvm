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

#ifndef TVM_BACKEND_HEXAGON_CODEGEN_MEMORY_SCOPE_H_
#define TVM_BACKEND_HEXAGON_CODEGEN_MEMORY_SCOPE_H_

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/runtime/memory_scope.h>

namespace tvm {
namespace backend {
namespace hexagon {

inline void RegisterMemoryScopes() {
  using runtime::MemoryAllocProperty;
  if (!runtime::MemoryScope::IsDefined("global.ddr")) {
    ffi::reflection::EnumDef<runtime::MemoryScopeObj>("global.ddr")
        .set_attr(runtime::attr::kMemoryAllocProperty,
                  static_cast<int>(MemoryAllocProperty::kSingleAllocation));
  }
  if (!runtime::MemoryScope::IsDefined("global.vtcm")) {
    ffi::reflection::EnumDef<runtime::MemoryScopeObj>("global.vtcm")
        .set_attr(runtime::attr::kMemoryAllocProperty,
                  static_cast<int>(MemoryAllocProperty::kNormal));
  }
}

}  // namespace hexagon
}  // namespace backend
}  // namespace tvm
#endif  // TVM_BACKEND_HEXAGON_CODEGEN_MEMORY_SCOPE_H_
