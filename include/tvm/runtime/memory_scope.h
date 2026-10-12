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

/*! \brief Canonical identity of an extensible memory space. */
class MemoryScopeNode : public ffi::EnumObj {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("runtime.MemoryScope", MemoryScopeNode, ffi::EnumObj);
};

/*!
 * \brief A registered memory space, independent of backend allocation policy.
 *
 * Register extensions with ffi::reflection::EnumDef<MemoryScopeNode>("my.scope").
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
  /*! \brief Look up a registered name. Unknown names raise ValueError. */
  TVM_RUNTIME_DLL static MemoryScope Get(const ffi::String& name);
  /*! \brief Register a new name. Duplicate names raise ValueError. */
  TVM_RUNTIME_DLL static MemoryScope Register(const ffi::String& name);
  /*! \brief Return the canonical global singleton. */
  TVM_RUNTIME_DLL static MemoryScope Global();
  /*! \brief Normalize legacy string input at an API boundary. */
  static MemoryScope FromString(const ffi::String& name) {
    return name.empty() ? Global() : Get(name);
  }
  bool operator==(const MemoryScope& other) const { return same_as(other); }
  bool operator!=(const MemoryScope& other) const { return !same_as(other); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(MemoryScope, ffi::Enum, MemoryScopeNode);
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
