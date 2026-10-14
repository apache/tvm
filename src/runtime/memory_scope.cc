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

#include <string_view>

namespace tvm {
namespace runtime {
namespace {
namespace refl = ffi::reflection;

void RegisterBuiltinScopes() {
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
}  // namespace

MemoryAllocProperty MemoryScope::alloc_property() const {
  static ffi::reflection::TypeAttrColumn state_column(ffi::type_attr::kEnumState);
  auto state = state_column[MemoryScopeObj::RuntimeTypeIndex()].cast<ffi::EnumState>();
  if (auto column = state->attrs.Get(attr::kMemoryAllocProperty)) {
    if (auto value = column->Get(*this)) {
      int property = value->cast<int>();
      TVM_FFI_CHECK(property >= static_cast<int>(MemoryAllocProperty::kNormal) &&
                        property <= static_cast<int>(MemoryAllocProperty::kSeparateAllocations),
                    ValueError)
          << "Invalid memory allocation property for " << name();
      return static_cast<MemoryAllocProperty>(property);
    }
  }
  return MemoryAllocProperty::kNormal;
}

bool MemoryScope::IsDefined(const ffi::String& name) {
  constexpr auto key = refl::AsByteArray(ffi::type_attr::kEnumState.name);
  if (TVMFFIGetTypeAttrColumn(&key) == nullptr) return false;
  auto value = refl::TypeAttrColumn(
      ffi::type_attr::kEnumState)[MemoryScopeObj::_GetOrAllocRuntimeTypeIndex()];
  if (value == nullptr) return false;
  TVM_FFI_CHECK(value.type_index() == ffi::EnumStateObj::_GetOrAllocRuntimeTypeIndex(), TypeError)
      << "Expected enum state for memory scopes";
  // The derived EnumState type index may still be deferred during static initialization.
  auto* state = static_cast<const ffi::EnumStateObj*>(value.cast<ffi::ObjectRef>().get());
  return state->indexes.count(name) != 0;
}

MemoryScope MemoryScope::Get(const ffi::String& name) {
  RegisterBuiltinScopes();
  return ffi::EnumObj::_GetByStrIndex<MemoryScopeObj>(name).as_or_throw<MemoryScope>();
}

MemoryScope MemoryScope::Register(const ffi::String& name) {
  RegisterBuiltinScopes();
  TVM_FFI_CHECK(!name.empty(), ValueError) << "Memory scope name must not be empty";
  TVM_FFI_CHECK(std::string_view(name.data(), name.size()).find('\0') == std::string_view::npos,
                ValueError)
      << "Memory scope name must not contain a null character";
  refl::EnumDef<MemoryScopeObj>(name.c_str());
  return Get(name);
}

const MemoryScope& MemoryScope::Global() {
  static const MemoryScope scope = Get("global");
  return scope;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterBuiltinScopes();
  refl::TypeAttrDef<MemoryScopeObj>()
      .def(ffi::type_attr::kDataToJson.name,
           [](const MemoryScopeObj* node) -> ffi::String { return node->_str_index; })
      .def(ffi::type_attr::kDataFromJson.name,
           [](const ffi::String& name) { return MemoryScope::Get(name); });
  refl::GlobalDef().def("runtime.MemoryScopeRegister", &MemoryScope::Register);
}
}  // namespace runtime
}  // namespace tvm
