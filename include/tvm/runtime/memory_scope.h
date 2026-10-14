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
#include <tvm/runtime/base.h>

#include <ostream>

namespace tvm {
namespace runtime {

/*! \brief Allocation behavior associated with a memory scope. */
enum class MemoryAllocProperty : int {
  kNormal = 0,
  kSingleAllocation = 1,
  kSeparateAllocations = 2,
};

namespace attr {
/*! \brief Integer-valued MemoryAllocProperty; absence denotes kNormal. */
inline constexpr const char* kMemoryAllocProperty = "runtime.memory_alloc_property";
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
  /*! \brief Return the allocation property assigned by the scope owner. */
  TVM_RUNTIME_DLL MemoryAllocProperty alloc_property() const;
  /*! \brief Test whether a name is registered, without initializing memory scopes. */
  TVM_RUNTIME_DLL static bool IsDefined(const ffi::String& name);
  /*! \brief Look up a registered name. Unknown names raise ValueError. */
  TVM_RUNTIME_DLL static MemoryScope Get(const ffi::String& name);
  /*! \brief Register a new name. Duplicate names raise ValueError. */
  TVM_RUNTIME_DLL static MemoryScope Register(const ffi::String& name);
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
