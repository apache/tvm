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
 * \file tvm/runtime/packed_func.h
 * \brief Type-erased function used across TVM API.
 */
#ifndef TVM_RUNTIME_PACKED_FUNC_H_
#define TVM_RUNTIME_PACKED_FUNC_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

using ffi::Any;
using ffi::AnyView;

namespace details {

template <typename T>
struct ModuleVTableEntryHelper {};

template <typename T, typename R, typename... Args>
struct ModuleVTableEntryHelper<R (T::*)(Args...) const> {
  using MemFnType = R (T::*)(Args...) const;
  static TVM_ALWAYS_INLINE void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> R { return (self->*f)(std::forward<Args>(args)...); };
    ffi::details::unpack_call<R>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                 args.data(), args.size(), rv);
  }
};

template <typename T, typename R, typename... Args>
struct ModuleVTableEntryHelper<R (T::*)(Args...)> {
  using MemFnType = R (T::*)(Args...);
  static TVM_ALWAYS_INLINE void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> R { return (self->*f)(std::forward<Args>(args)...); };
    ffi::details::unpack_call<R>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                 args.data(), args.size(), rv);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...) const> {
  using MemFnType = void (T::*)(Args...) const;
  static TVM_ALWAYS_INLINE void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> void { (self->*f)(std::forward<Args>(args)...); };
    ffi::details::unpack_call<void>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                    args.data(), args.size(), rv);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...)> {
  using MemFnType = void (T::*)(Args...);
  static TVM_ALWAYS_INLINE void Call(ffi::Any* rv, T* self, MemFnType f, ffi::PackedArgs args) {
    auto wrapped = [self, f](Args... args) -> void { (self->*f)(std::forward<Args>(args)...); };
    ffi::details::unpack_call<void>(std::make_index_sequence<sizeof...(Args)>{}, nullptr, wrapped,
                                    args.data(), args.size(), rv);
  }
};
}  // namespace details

#define TVM_MODULE_VTABLE_BEGIN(TypeKey)                                                    \
  const char* type_key() const final { return TypeKey; }                                    \
  ffi::Function GetFunction(const String& _name, const ObjectPtr<Object>& _self) override { \
    using SelfPtr = std::remove_cv_t<decltype(this)>;
#define TVM_MODULE_VTABLE_END()  \
  return ffi::Function(nullptr); \
  }
#define TVM_MODULE_VTABLE_END_WITH_DEFAULT(MemFunc) \
  {                                                 \
    auto f = (MemFunc);                             \
    return (this->*f)(_name);                       \
  }                                                 \
  }  // NOLINT(*)
#define TVM_MODULE_VTABLE_ENTRY(Name, MemFunc)                                            \
  if (_name == Name) {                                                                    \
    return ffi::Function::FromPacked([_self](ffi::PackedArgs args, Any* rv) -> void {     \
      using Helper = ::tvm::runtime::details::ModuleVTableEntryHelper<decltype(MemFunc)>; \
      SelfPtr self = static_cast<SelfPtr>(_self.get());                                   \
      Helper::Call(rv, self, MemFunc, args);                                              \
    });                                                                                   \
  }
#define TVM_MODULE_VTABLE_ENTRY_PACKED(Name, MemFunc)                     \
  if (_name == Name) {                                                    \
    return ffi::Function([_self](ffi::PackedArgs args, Any* rv) -> void { \
      (static_cast<SelfPtr>(_self.get())->*(MemFunc))(args, rv);          \
    });                                                                   \
  }

/*!
 * \brief Export typed function as a ffi::Function
 *        that can be loaded by LibraryModule.
 *
 * \param ExportName The symbol name to be exported.
 * \param Function The typed function.
 * \note ExportName and Function must be different,
 *       see code examples below.
 *
 * \sa ffi::TypedFunction
 *
 * \code
 *
 * int AddOne_(int x) {
 *   return x + 1;
 * }
 *
 * // Expose the function as "AddOne"
 * TVM_DLL_EXPORT_TYPED_FUNC(AddOne, AddOne_);
 *
 * // Expose the function as "SubOne"
 * TVM_DLL_EXPORT_TYPED_FUNC(SubOne, [](int x) {
 *   return x - 1;
 * });
 *
 * // The following code will cause compilation error.
 * // Because the same Function and ExportName
 * // TVM_DLL_EXPORT_TYPED_FUNC(AddOne_, AddOne_);
 *
 * // The following code is OK, assuming the macro
 * // is in a different namespace from xyz
 * // TVM_DLL_EXPORT_TYPED_FUNC(AddOne_, xyz::AddOne_);
 *
 * \endcode
 */
#define TVM_DLL_EXPORT_TYPED_FUNC(ExportName, Function)                                      \
  extern "C" {                                                                               \
  TVM_DLL int ExportName(void* self, TVMFFIAny* args, int32_t num_args, TVMFFIAny* result) { \
    TVM_FFI_SAFE_CALL_BEGIN();                                                               \
    using FuncInfo = ::tvm::ffi::details::FunctionInfo<decltype(Function)>;                  \
    static std::string name = #ExportName;                                                   \
    ::tvm::ffi::details::unpack_call<typename FuncInfo::RetType>(                            \
        std::make_index_sequence<FuncInfo::num_args>{}, &name, Function,                     \
        reinterpret_cast<const ::tvm::ffi::AnyView*>(args), num_args,                        \
        reinterpret_cast<::tvm::ffi::Any*>(result));                                         \
    TVM_FFI_SAFE_CALL_END();                                                                 \
  }                                                                                          \
  }
}  // namespace runtime  // NOLINT(*)
using ffi::Any;
using ffi::AnyView;
}  // namespace tvm
#endif  // TVM_RUNTIME_PACKED_FUNC_H_
