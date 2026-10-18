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
 * \file tvm/tirx/tile/thread_scope.h
 * \brief Enum identities for backend-defined thread hierarchies.
 */
#ifndef TVM_TIRX_TILE_THREAD_SCOPE_H_
#define TVM_TIRX_TILE_THREAD_SCOPE_H_
#include <tvm/ffi/enum.h>
#include <tvm/ffi/reflection/enum_def.h>
#include <tvm/runtime/base.h>

#include <ostream>

namespace tvm::tirx {

namespace thread_scope_attr {
/*! \brief Backend owning a thread scope. */
inline constexpr ffi::reflection::AttrKey<ffi::String> kBackend{"backend"};
/*! \brief Explicit hierarchy rank. Smaller ranks denote wider scopes. */
inline constexpr ffi::reflection::AttrKey<int64_t> kRank{"rank"};
}  // namespace thread_scope_attr

class ThreadScopeObj : public ffi::EnumObj {
 public:
  const ffi::String& name() const { return _str_index; }
  TVM_TIRX_DLL ffi::String backend() const;
  TVM_TIRX_DLL ffi::String local_name() const;
  TVM_TIRX_DLL int64_t rank() const;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.ThreadScope", ThreadScopeObj, ffi::EnumObj);
};

/*!
 * \brief Canonical thread-scope identity, named as backend.scope.
 *
 * Backends register tags with EnumDef<ThreadScopeObj> and explicit backend and
 * rank attributes. Registration order has no hierarchy semantics.
 */
class ThreadScope : public ffi::Enum {
 public:
  explicit ThreadScope(const ffi::String& name) : ffi::Enum(Get(name)) {}
  TVM_TIRX_DLL static ThreadScope Get(const ffi::String& name);
  const ffi::String& name() const { return get()->name(); }
  bool operator==(const ThreadScope& other) const { return same_as(other); }
  bool operator!=(const ThreadScope& other) const { return !same_as(other); }
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(ThreadScope, ffi::Enum, ThreadScopeObj);
};

/*! \brief Compare explicit ranks within one backend's hierarchy. */
TVM_TIRX_DLL bool ScopeHigher(const ThreadScope& a, const ThreadScope& b);

inline std::ostream& operator<<(std::ostream& os, const ThreadScope& scope) {
  return os << scope.name();
}

}  // namespace tvm::tirx

namespace tvm::ffi {
template <>
inline constexpr bool use_default_type_traits_v<tirx::ThreadScope> = false;
template <>
struct TypeTraits<tirx::ThreadScope>
    : public ObjectRefWithFallbackTraitsBase<tirx::ThreadScope, ffi::String> {
  static tirx::ThreadScope ConvertFallbackValue(ffi::String name) {
    return tirx::ThreadScope::Get(name);
  }
};
}  // namespace tvm::ffi
#endif  // TVM_TIRX_TILE_THREAD_SCOPE_H_
