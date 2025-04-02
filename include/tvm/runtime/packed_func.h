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
#include <tvm/runtime/container/boxed_primitive.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>

namespace tvm {
namespace runtime {

using ffi::Any;
using ffi::AnyView;

/*!
 * \brief Utility function to convert legacy TVMArgValue to AnyView
 * \note This routine is not fastest, but serves purpose to do transition of ABI.
 */
inline TVMFFIAny LegacyTVMArgValueToFFIAny(TVMValue value, int type_code) {
  TVMFFIAny res;
  switch (type_code) {
    case kTVMArgInt: {
      res.type_index = ffi::TypeIndex::kTVMFFIInt;
      res.v_int64 = value.v_int64;
      break;
    }
    case kTVMArgFloat: {
      res.type_index = ffi::TypeIndex::kTVMFFIFloat;
      res.v_float64 = value.v_float64;
      break;
    }
    case kTVMOpaqueHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIOpaquePtr;
      res.v_ptr = value.v_handle;
      break;
    }
    case kTVMNullptr: {
      res.type_index = ffi::TypeIndex::kTVMFFINone;
      break;
    }
    case kTVMDataType: {
      res.type_index = ffi::TypeIndex::kTVMFFIDataType;
      res.v_dtype = value.v_type;
      break;
    }
    case kDLDevice: {
      res.type_index = ffi::TypeIndex::kTVMFFIDevice;
      res.v_device = value.v_device;
      break;
    }
    case kTVMDLTensorHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIDLTensorPtr;
      res.v_ptr = value.v_handle;
      break;
    }
    case kTVMObjectHandle: {
      res.v_obj = static_cast<TVMFFIObject*>(value.v_handle);
      res.type_index = res.v_obj->type_index;
      break;
    }
    case kTVMModuleHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIRuntimeModule;
      res.v_obj = static_cast<TVMFFIObject*>(value.v_handle);
      break;
    }
    case kTVMPackedFuncHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFIFunc;
      res.v_obj = static_cast<TVMFFIObject*>(value.v_handle);
      break;
    }
    case kTVMStr: {
      res.type_index = ffi::TypeIndex::kTVMFFIRawStr;
      res.v_c_str = value.v_str;
      break;
    }
    case kTVMBytes: {
      res.type_index = ffi::TypeIndex::kTVMFFIByteArrayPtr;
      res.v_ptr = value.v_handle;
      break;
    }
    case kTVMNDArrayHandle: {
      res.type_index = ffi::TypeIndex::kTVMFFINDArray;
      res.v_obj = reinterpret_cast<TVMFFIObject*>(TVMArrayHandleToObjectHandle(value.v_handle));
      break;
    }
    case kTVMArgBool: {
      res.type_index = ffi::TypeIndex::kTVMFFIBool;
      res.v_int64 = value.v_int64;
      break;
    }
    case kTVMObjectRValueRefArg: {
      res.type_index = ffi::TypeIndex::kTVMFFIObjectRValueRef;
      res.v_ptr = value.v_handle;
      break;
    }
    default: {
      LOG(FATAL) << "Unsupported type code: " << type_code;
    }
  }
  return res;
}

/*!
 * \brief Utility function to convert legacy TVMArgValue to AnyView
 * \note This routine is not fastest, but serves purpose to do transition of ABI.
 */
inline AnyView LegacyTVMArgValueToAnyView(TVMValue value, int type_code) {
  return AnyView::CopyFromTVMFFIAny(LegacyTVMArgValueToFFIAny(value, type_code));
}

/*!
 * \brief Utility function to convert legacy TVMArgValue to Any
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
    case ffi::TypeIndex::kTVMFFIRuntimeModule: {
      type_code[0] = kTVMModuleHandle;
      value[0].v_handle = src.v_obj;
      break;
    }
    case ffi::TypeIndex::kTVMFFIFunc: {
      type_code[0] = kTVMPackedFuncHandle;
      value[0].v_handle = src.v_obj;
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
 * \brief Legacy TVM args kept for backward compact
 */
class LegacyTVMArgs {
 public:
  const TVMValue* values;
  const int* type_codes;
  int num_args;
  /*!
   * \brief constructor
   * \param values The argument values
   * \param type_codes The argument type codes
   * \param num_args number of arguments.
   */
  LegacyTVMArgs(const TVMValue* values, const int* type_codes, int num_args)
      : values(values), type_codes(type_codes), num_args(num_args) {}
};

/*!
 * \brief Translate legacy TVMArgs to PackedArgs
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
 * \brief Translate legacy TVMArgs to PackedArgs
 * \param value The TVMValue array
 * \param type_code The type code array
 * \param num_args The number of arguments
 * \param dst The destination AnyView array
 */
inline void PackedArgsToLegacyTVMArgs(const AnyView* args, int num_args, TVMValue* value,
                                      int* type_code) {
  for (int i = 0; i < num_args; ++i) {
    AnyViewToLegacyTVMArgValue(args[i].CopyToTVMFFIAny(), value + i, type_code + i);
  }
}

/*!
 * \brief Call the function in legacy packed format.
 * \param ffi_func The function object
 * \param args The arguments
 * \param rv The return value.
 */
inline void LegacyCallPacked(ffi::FunctionObj* ffi_func, const TVMValue* value,
                             const int* type_code, int num_args, Any* rv) {
  std::vector<ffi::AnyView> args_vec(num_args);
  LegacyTVMArgsToPackedArgs(value, type_code, num_args, args_vec.data());
  // redirect to the normal call packed.
  ffi_func->CallPacked(args_vec.data(), args_vec.size(), rv);
}

// redirect to ffi::PackedArgs
using TVMArgs = ffi::PackedArgs;
// redirect to ffi::AnyView and ffi::Any for ArgValue and RetValue
using TVMArgValue = ffi::AnyView;
using TVMRetValue = ffi::Any;

// redirect to ffi::Function
using PackedFunc = ffi::Function;

template <typename FType>
using TypedPackedFunc = ffi::TypedFunction<FType>;

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
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    *rv = (self->*f)(args[Is]...);
  }
};

template <typename T, typename R, typename... Args>
struct ModuleVTableEntryHelper<R (T::*)(Args...)> {
  using MemFnType = R (T::*)(Args...);
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    *rv = (self->*f)(args[Is]...);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...) const> {
  using MemFnType = void (T::*)(Args...) const;
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    (self->*f)(args[Is]...);
  }
};

template <typename T, typename... Args>
struct ModuleVTableEntryHelper<void (T::*)(Args...)> {
  using MemFnType = void (T::*)(Args...);
  using IndexSeq = std::index_sequence_for<Args...>;
  static constexpr const std::size_t LenArgs = sizeof...(Args);

  template <std::size_t... Is>
  static TVM_ALWAYS_INLINE void Call(TVMRetValue* rv, T* self, MemFnType f, TVMArgs args,
                                     std::index_sequence<Is...>) {
    (self->*f)(args[Is]...);
  }
};
}  // namespace details

#define TVM_MODULE_VTABLE_BEGIN(TypeKey)                                                 \
  const char* type_key() const final { return TypeKey; }                                 \
  PackedFunc GetFunction(const String& _name, const ObjectPtr<Object>& _self) override { \
    using SelfPtr = std::remove_cv_t<decltype(this)>;
#define TVM_MODULE_VTABLE_END() \
  return PackedFunc(nullptr);   \
  }
#define TVM_MODULE_VTABLE_END_WITH_DEFAULT(MemFunc) \
  {                                                 \
    auto f = (MemFunc);                             \
    return (this->*f)(_name);                       \
  }                                                 \
  }  // NOLINT(*)
#define TVM_MODULE_VTABLE_ENTRY(Name, MemFunc)                                                    \
  if (_name == Name) {                                                                            \
    return ffi::Function::FromPacked([_self](ffi::PackedArgs args, Any* rv) -> void {             \
      using Helper = ::tvm::runtime::details::ModuleVTableEntryHelper<decltype(MemFunc)>;         \
      SelfPtr self = static_cast<SelfPtr>(_self.get());                                           \
      CHECK_EQ(args.size(), Helper::LenArgs)                                                      \
          << "Function `" << self->type_key() << "::" << Name << "` requires " << Helper::LenArgs \
          << " arguments, but got " << args.size();                                               \
      Helper::Call(rv, self, MemFunc, args, Helper::IndexSeq{});                                  \
    });                                                                                           \
  }
#define TVM_MODULE_VTABLE_ENTRY_PACKED(Name, MemFunc)                  \
  if (_name == Name) {                                                 \
    return PackedFunc([_self](ffi::PackedArgs args, Any* rv) -> void { \
      (static_cast<SelfPtr>(_self.get())->*(MemFunc))(args, rv);       \
    });                                                                \
  }
}  // namespace runtime

using ffi::Any;
using ffi::AnyView;

}  // namespace tvm
#endif  // TVM_RUNTIME_PACKED_FUNC_H_
