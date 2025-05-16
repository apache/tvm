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
#include <tvm/runtime/c_runtime_api.h>
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

/*!
 * \brief Utility function to convert legacy ffi::AnyView to AnyView
 * \note This routine is not fastest, but serves purpose to do transition of ABI.
 */
inline TVMFFIAny LegacyTVMArgValueToFFIAny(TVMValue value, int type_code) {
  TVMFFIAny res;
  // clear first to ensure consistent hash
  res.v_uint64 = 0;
  switch (type_code) {
    case kTVMArgInt: {
      res.type_index = ffi::TypeIndex::kTVMFFIInt;
      res.v_int64 = value.v_int64;
      return res;
    }
    case kTVMArgFloat: {
      res.type_index = ffi::TypeIndex::kTVMFFIFloat;
      res.v_float64 = value.v_float64;
      return res;
    }
    case kTVMOpaqueHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIOpaquePtr;
      res.v_ptr = value.v_handle;
      return res;
    }
    case kTVMNullptr: {
      res.type_index = ffi::TypeIndex::kTVMFFINone;
      return res;
    }
    case kTVMDataType: {
      res.type_index = ffi::TypeIndex::kTVMFFIDataType;
      res.v_dtype = value.v_type;
      return res;
    }
    case kDLDevice: {
      res.type_index = ffi::TypeIndex::kTVMFFIDevice;
      res.v_device = value.v_device;
      return res;
    }
    case kTVMDLTensorHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIDLTensorPtr;
      res.v_ptr = value.v_handle;
      return res;
    }
    case kTVMObjectHandle: {
      res.v_obj = static_cast<TVMFFIObject*>(value.v_handle);
      res.type_index = res.v_obj->type_index;
      return res;
    }
    case kTVMModuleHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIModule;
      res.v_obj = static_cast<TVMFFIObject*>(value.v_handle);
      return res;
    }
    case kTVMPackedFuncHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIFunction;
      res.v_obj = static_cast<TVMFFIObject*>(value.v_handle);
      return res;
    }
    case kTVMStr: {
      res.type_index = ffi::TypeIndex::kTVMFFIRawStr;
      res.v_c_str = value.v_str;
      return res;
    }
    case kTVMBytes: {
      res.type_index = ffi::TypeIndex::kTVMFFIByteArrayPtr;
      res.v_ptr = value.v_handle;
      return res;
    }
    case kTVMNDArrayHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFINDArray;
      res.v_obj = reinterpret_cast<TVMFFIObject*>(TVMArrayHandleToObjectHandle(value.v_handle));
      return res;
    }
    case kTVMArgBool: {
      res.type_index = ffi::TypeIndex::kTVMFFIBool;
      res.v_int64 = value.v_int64;
      return res;
    }
    case kTVMObjectRValueRefArg: {
      res.type_index = ffi::TypeIndex::kTVMFFIObjectRValueRef;
      res.v_ptr = value.v_handle;
      return res;
    }
    default: {
      LOG(FATAL) << "Unsupported type code: " << type_code;
      TVM_FFI_UNREACHABLE();
    }
  }
}

/*!
 * \brief Utility function to convert legacy ffi::AnyView to AnyView
 * \note This routine is not fastest, but serves purpose to do transition of ABI.
 */
inline AnyView LegacyTVMArgValueToAnyView(TVMValue value, int type_code) {
  return AnyView::CopyFromTVMFFIAny(LegacyTVMArgValueToFFIAny(value, type_code));
}

/*!
 * \brief Utility function to convert legacy ffi::AnyView to Any
 * \note This routine is not fastest, but serves purpose to do transition of ABI.
 */
inline Any MoveLegacyTVMArgValueToAny(TVMValue value, int type_code) {
  return ffi::details::AnyUnsafe::MoveTVMFFIAnyToAny(LegacyTVMArgValueToFFIAny(value, type_code));
}

/*
 * \brief Convert AnyView to legacy TVMValue and type_code
 * \param src The AnyView to convert
 * \param value The TVMValue to store the result
 * \param type_code The type code to store the result
 * \note This routine is not fastest, but serves purpose to do transition of ABI.
 */
inline void AnyViewToLegacyTVMArgValue(TVMFFIAny src, TVMValue* value, int* type_code) {
  switch (src.type_index) {
    case ffi::TypeIndex::kTVMFFIBool: {
      type_code[0] = kTVMArgBool;
      value[0].v_int64 = src.v_int64;
      break;
    }
    case ffi::TypeIndex::kTVMFFIInt: {
      type_code[0] = kDLInt;
      value[0].v_int64 = src.v_int64;
      break;
    }
    case ffi::TypeIndex::kTVMFFIFloat: {
      type_code[0] = kDLFloat;
      value[0].v_float64 = src.v_float64;
      break;
    }
    case ffi::TypeIndex::kTVMFFIOpaquePtr: {
      type_code[0] = kTVMOpaqueHandle;
      value[0].v_handle = src.v_ptr;
      break;
    }
    case ffi::TypeIndex::kTVMFFINone: {
      type_code[0] = kTVMNullptr;
      break;
    }
    case ffi::TypeIndex::kTVMFFIDataType: {
      type_code[0] = kTVMDataType;
      value[0].v_type = src.v_dtype;
      break;
    }
    case ffi::TypeIndex::kTVMFFIDevice: {
      type_code[0] = kDLDevice;
      value[0].v_device = src.v_device;
      break;
    }
    case ffi::TypeIndex::kTVMFFIDLTensorPtr: {
      type_code[0] = kTVMDLTensorHandle;
      value[0].v_handle = src.v_ptr;
      break;
    }
    case ffi::TypeIndex::kTVMFFIRawStr: {
      type_code[0] = kTVMStr;
      value[0].v_str = src.v_c_str;
      break;
    }
    case ffi::TypeIndex::kTVMFFIByteArrayPtr: {
      type_code[0] = kTVMBytes;
      value[0].v_handle = src.v_ptr;
      break;
    }
    case ffi::TypeIndex::kTVMFFINDArray: {
      type_code[0] = kTVMNDArrayHandle;
      value[0].v_handle = ObjectHandleToTVMArrayHandle(reinterpret_cast<Object*>(src.v_obj));
      break;
    }
    case ffi::TypeIndex::kTVMFFIModule: {
      type_code[0] = kTVMModuleHandle;
      value[0].v_handle = src.v_obj;
      break;
    }
    case ffi::TypeIndex::kTVMFFIFunction: {
      type_code[0] = kTVMPackedFuncHandle;
      value[0].v_handle = src.v_obj;
      break;
    }
    case ffi::TypeIndex::kTVMFFIObjectRValueRef: {
      type_code[0] = kTVMObjectRValueRefArg;
      value[0].v_handle = src.v_ptr;
      break;
    }
    default: {
      if (src.type_index >= ffi::TypeIndex::kTVMFFIStaticObjectBegin) {
        type_code[0] = kTVMObjectHandle;
        value[0].v_handle = src.v_obj;
        break;
      }
      LOG(FATAL) << "Unsupported type index: " << src.type_index;
    }
  }
}

/*
 * \brief Move Any to legacy TVMValue and type_code
 * \param src The Any to move
 * \param value The TVMValue to store the result
 * \param type_code The type code to store the result
 */
inline void MoveAnyToLegacyTVMValue(Any&& src, TVMValue* value, int* type_code) {
  TVMFFIAny val = ffi::details::AnyUnsafe::MoveAnyToTVMFFIAny(std::move(src));
  // NOTE: conversion rule is the same as AnyViewToLegacyTVMArgValue
  AnyViewToLegacyTVMArgValue(val, value, type_code);
}

/*!
 * \brief Translate legacy ffi::PackedArgs to PackedArgs
 * \param value The TVMValue array
 * \param type_code The type code array
 * \param num_args The number of arguments
 * \param dst The destination AnyView array
 */
inline void LegacyTVMArgsToPackedArgs(const TVMValue* value, const int* type_code, int num_args,
                                      AnyView* dst) {
  for (int i = 0; i < num_args; ++i) {
    dst[i] = LegacyTVMArgValueToAnyView(value[i], type_code[i]);
  }
}

/*!
 * \brief Translate legacy ffi::PackedArgs to PackedArgs
 * \param args The AnyView array
 * \param num_args The number of arguments
 * \param value The TVMValue array
 * \param type_code The type code array
 */
inline void PackedArgsToLegacyTVMArgs(const AnyView* args, int num_args, TVMValue* value,
                                      int* type_code) {
  for (int i = 0; i < num_args; ++i) {
    AnyViewToLegacyTVMArgValue(args[i].CopyToTVMFFIAny(), value + i, type_code + i);
  }
}

/*!
 * \brief Convert argument type code to string.
 * \param type_code The input type code.
 * \return The corresponding string repr.
 */
inline const char* ArgTypeCode2Str(int type_code) {
  switch (type_code) {
    case kDLInt:
      return "int";
    case kTVMArgBool:
      return "bool";
    case kDLUInt:
      return "uint";
    case kDLFloat:
      return "float";
    case kTVMStr:
      return "str";
    case kTVMBytes:
      return "bytes";
    case kTVMOpaqueHandle:
      return "handle";
    case kTVMNullptr:
      return "NULL";
    case kTVMDLTensorHandle:
      return "ArrayHandle";
    case kTVMDataType:
      return "DLDataType";
    case kDLDevice:
      return "DLDevice";
    case kTVMPackedFuncHandle:
      return "FunctionHandle";
    case kTVMModuleHandle:
      return "ModuleHandle";
    case kTVMNDArrayHandle:
      return "NDArrayContainer";
    case kTVMObjectHandle:
      return "Object";
    case kTVMObjectRValueRefArg:
      return "ObjectRValueRefArg";
    default:
      LOG(FATAL) << "unknown type_code=" << static_cast<int>(type_code);
  }
  throw;
}

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
