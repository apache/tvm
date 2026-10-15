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

#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/memory_scope.h>

namespace tvm {
namespace runtime {
namespace refl = ffi::reflection;

void detail::RegisterBuiltinMemoryScopes() {
  // Lookups and defaults can run in other translation units' static initializers.
  // Initialize the complete set once, before any lookup, without relying on order.
  static const bool initialized = [] {
    refl::ObjectDef<MemoryScopeObj>(refl::init(false));
    for (const char* name : {"global", "shared"}) {
      refl::EnumDef<MemoryScopeObj>{name}.set_attr(attr::kMemoryAllocProperty,
                                                   static_cast<int>(MemoryAllocProperty::kNormal));
    }
    refl::EnumDef<MemoryScopeObj>{"shared.dyn"}.set_attr(
        attr::kMemoryAllocProperty, static_cast<int>(MemoryAllocProperty::kSingleAllocation));
    for (const char* name : {"warp", "local"}) {
      refl::EnumDef<MemoryScopeObj>{name}.set_attr(
          attr::kMemoryAllocProperty, static_cast<int>(MemoryAllocProperty::kSeparateAllocations));
    }
    return true;
  }();
  (void)initialized;
}

MemoryScope MemoryScope::Get(const ffi::String& name) {
  detail::RegisterBuiltinMemoryScopes();
  return ffi::EnumObj::_GetByStrIndex<MemoryScopeObj>(name).as_or_throw<MemoryScope>();
}

const MemoryScope& MemoryScope::Global() {
  static const MemoryScope scope = Get("global");
  return scope;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  detail::RegisterBuiltinMemoryScopes();
  refl::TypeAttrDef<MemoryScopeObj>()
      .def(ffi::type_attr::kDataToJson.name,
           [](const MemoryScopeObj* node) -> ffi::String { return node->_str_index; })
      .def(ffi::type_attr::kDataFromJson.name,
           [](const ffi::String& name) { return MemoryScope::Get(name); });
  refl::GlobalDef().def("runtime.MemoryScopeRegister", &MemoryScope::Register);
}
}  // namespace runtime
}  // namespace tvm
