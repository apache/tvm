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

#ifndef TVM_BACKEND_MEMORY_SCOPE_H_
#define TVM_BACKEND_MEMORY_SCOPE_H_

#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/runtime/memory_scope.h>

#include <initializer_list>
#include <utility>

namespace tvm {
namespace backend {

// A compiler backend and its runtime sidecar may define the same owned scopes.
// Accept only matching explicit metadata when the other component loaded first.
inline void RegisterBackendMemoryScope(
    const char* name, runtime::MemoryAllocProperty property,
    std::initializer_list<std::pair<const char*, ffi::Any>> attributes = {}) {
  runtime::MemoryScope::Global();
  static ffi::reflection::TypeAttrColumn state_column(ffi::reflection::type_attr::kEnumState);
  auto state =
      state_column[runtime::MemoryScopeObj::_GetOrAllocRuntimeTypeIndex()].cast<ffi::EnumState>();
  if (auto existing = state->indexes.Get(ffi::String(name))) {
    auto check_attribute = [&](const char* key, ffi::Any expected) {
      auto column = state->attrs.Get(key);
      TVM_FFI_CHECK(column.has_value(), ValueError)
          << "Memory scope " << name << " is missing backend attribute " << key;
      auto actual = column->Get(*existing);
      TVM_FFI_CHECK(actual.has_value() && actual->type_index() == expected.type_index() &&
                        ffi::AnyEqual()(*actual, expected),
                    ValueError)
          << "Memory scope " << name << " has incompatible backend attribute " << key;
    };
    check_attribute(runtime::attr::kMemoryAllocProperty, static_cast<int>(property));
    for (const auto& [key, value] : attributes) check_attribute(key, value);
    return;
  }
  ffi::reflection::EnumDef<runtime::MemoryScopeObj> definition(name);
  definition.set_attr(runtime::attr::kMemoryAllocProperty, static_cast<int>(property));
  for (const auto& [key, value] : attributes) definition.set_attr(key, value);
}

}  // namespace backend
}  // namespace tvm
#endif  // TVM_BACKEND_MEMORY_SCOPE_H_
