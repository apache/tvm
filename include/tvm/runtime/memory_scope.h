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

#ifndef TVM_RUNTIME_MEMORY_SCOPE_H_
#define TVM_RUNTIME_MEMORY_SCOPE_H_

#include <tvm/ffi/enum.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/runtime/base.h>

#include <ostream>
#include <string_view>

namespace tvm {
namespace runtime {
namespace detail {
TVM_RUNTIME_DLL void RegisterBuiltinMemoryScopes();
}  // namespace detail

/*! \brief Allocation behavior associated with a memory scope. */
enum class MemoryAllocProperty : int {
  kNormal = 0,
  kSingleAllocation = 1,
  kSeparateAllocations = 2,
};

namespace attr {
/*! \brief Integer-valued MemoryAllocProperty; absence denotes kNormal. */
inline constexpr ffi::reflection::AttrKey<MemoryAllocProperty> kMemoryAllocProperty{
    "runtime.memory_alloc_property"};
}  // namespace attr

/*! \brief Canonical identity of an extensible memory space. */
class MemoryScopeObj : public ffi::EnumObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.MemoryScope", MemoryScopeObj, ffi::EnumObj);
};

/*!
 * \brief A registered memory space, independent of backend allocation policy.
 *
 * Register extensions with ffi::reflection::EnumDef<MemoryScopeObj>("my.scope").
 * Registration order has no semantic meaning.  A backend must explicitly support
 * a scope before it can allocate storage in that space.
 */
class MemoryScope : public ffi::Enum {
 public:
  /*! \brief Default to the global memory space. */
  MemoryScope() : ffi::Enum(Global()) {}
  /*! \brief Look up a registered name; never implicitly register it. */
  explicit MemoryScope(const ffi::String& name) : ffi::Enum(Get(name)) {}
  /*! \brief Return the canonical spelling. */
  const ffi::String& name() const { return get()->_str_index; }
  /*! \brief Test whether a name is registered, without initializing memory scopes. */
  static bool IsDefined(const ffi::String& name) {
    namespace refl = ffi::reflection;
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
  /*! \brief Look up a registered name. Unknown names raise ValueError. */
  TVM_RUNTIME_DLL static MemoryScope Get(const ffi::String& name);
  /*! \brief Register a new name. Duplicate names raise ValueError. */
  static MemoryScope Register(const ffi::String& name) {
    namespace refl = ffi::reflection;
    detail::RegisterBuiltinMemoryScopes();
    TVM_FFI_CHECK(!name.empty(), ValueError) << "Memory scope name must not be empty";
    TVM_FFI_CHECK(std::string_view(name.data(), name.size()).find('\0') == std::string_view::npos,
                  ValueError)
        << "Memory scope name must not contain a null character";
    refl::EnumDef<MemoryScopeObj>(name.c_str());
    return Get(name);
  }
  /*! \brief Return the canonical global singleton. */
  TVM_RUNTIME_DLL static const MemoryScope& Global();
  bool operator==(const MemoryScope& other) const { return same_as(other); }
  bool operator!=(const MemoryScope& other) const { return !same_as(other); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MemoryScope, ffi::Enum, MemoryScopeObj);
};

inline std::ostream& operator<<(std::ostream& os, const MemoryScope& scope) {
  return os << scope.name();
}

}  // namespace runtime
namespace ffi {
template <>
inline constexpr bool use_default_type_traits_v<runtime::MemoryScope> = false;
template <>
struct TypeTraits<runtime::MemoryScope>
    : public ObjectRefWithFallbackTraitsBase<runtime::MemoryScope, ffi::String> {
  static runtime::MemoryScope ConvertFallbackValue(ffi::String name) {
    return runtime::MemoryScope::Get(name);
  }
};
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_RUNTIME_MEMORY_SCOPE_H_
