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
      res.v_int64 = value.v_float64;
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
      LOG(FATAL) << "Do not support kTVMObjectRValueRefArg";
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
  return Any::MoveFromTVMFFIAny(LegacyTVMArgValueToFFIAny(value, type_code));
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
      if (src.type_index >= ffi::TypeIndex::kTVMFFIDynObjectBegin) {
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
  TVMFFIAny val;
  src.MoveToTVMFFIAny(&val);
  // NOTE: conversion rule is the same as AnyViewToLegacyTVMArgValue
  AnyViewToLegacyTVMArgValue(val, value, type_code);
}

/*!
 * \brief Legacy TVM args kept for backward compact
 */
class TVMArgs {
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
  TVMArgs(const TVMValue* values, const int* type_codes, int num_args)
      : values(values), type_codes(type_codes), num_args(num_args) {}
  /*! \return size of the arguments */
  int size() const { return num_args; }
  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  AnyView operator[](int i) const { return LegacyTVMArgValueToAnyView(values[i], type_codes[i]); }
};

/* \brief argument settter to PackedFunc */
class TVMArgsSetter {
 public:
  TVMArgsSetter(TVMValue* values, int* type_codes) : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  TVM_ALWAYS_INLINE void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDLInt;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, bool value) const {
    values_[i].v_int64 = value;
    type_codes_[i] = kTVMArgBool;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    ICHECK_LE(value, static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDLInt;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kTVMNullptr;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kTVMOpaqueHandle;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, DLTensor* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kTVMDLTensorHandle;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, Device value) const {
    values_[i].v_device = value;
    type_codes_[i] = kDLDevice;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, DLDataType value) const {
    values_[i].v_type = value;
    type_codes_[i] = kTVMDataType;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, DataType dtype) const {
    operator()(i, dtype.operator DLDataType());
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, const char* value) const {
    values_[i].v_str = value;
    type_codes_[i] = kTVMStr;
  }
  // setters for container types
  TVM_ALWAYS_INLINE void operator()(size_t i, const std::string& value) const {
    values_[i].v_str = value.c_str();
    type_codes_[i] = kTVMStr;
  }
  TVM_ALWAYS_INLINE void operator()(size_t i, const TVMByteArray& value) const {
    values_[i].v_handle = const_cast<TVMByteArray*>(&value);
    type_codes_[i] = kTVMBytes;
  }
  void operator()(size_t i, const ffi::AnyView& value) const {
    AnyViewToLegacyTVMArgValue(value.CopyToTVMFFIAny(), &values_[i], &type_codes_[i]);
  }
  void operator()(size_t i, const ffi::Any& value) const {
    AnyViewToLegacyTVMArgValue(value.operator AnyView().CopyToTVMFFIAny(), &values_[i],
                               &type_codes_[i]);
  }
  // ObjectRef handling
  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  TVM_ALWAYS_INLINE void operator()(size_t i, const TObjectRef& value) const {
    this->SetObject(i, value);
  }

  template <typename TObjectRef,
            typename = typename std::enable_if<std::is_base_of<
                ObjectRef, typename std::remove_reference<TObjectRef>::type>::value>::type>
  TVM_ALWAYS_INLINE void operator()(size_t i, TObjectRef&& value) const {
    this->SetObject(i, std::forward<TObjectRef>(value));
  }

 private:
  template <typename TObjectRef>
  inline void SetObject(size_t i, TObjectRef&& value) const;
  /*! \brief The values fields */
  TVMValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

// redirect to ffi::AnyView and ffi::Any for ArgValue and RetValue
using TVMArgValue = ffi::AnyView;
using TVMRetValue = ffi::Any;

// Adapter class to keep common behavior of PackedFunc
class PackedFunc : public ffi::Function {
 public:
  using ffi::Function::operator();
  using ffi::Function::FromPacked;
  using ffi::Function::operator==;
  using ffi::Function::operator!=;
  using ffi::Function::CallPacked;
  // default construction from nullptr
  PackedFunc(std::nullptr_t) : ffi::Function(nullptr) {}
  PackedFunc(ObjectPtr<Object> data) : ffi::Function(data) {}
  PackedFunc() = default;
  PackedFunc(const PackedFunc& other) = default;
  PackedFunc(PackedFunc&& other) = default;
  PackedFunc(const ffi::Function& other) : ffi::Function(other) {}
  PackedFunc& operator=(PackedFunc&& other) = default;
  PackedFunc& operator=(const PackedFunc& other) = default;

  // explicit interchange with ffi::Function
  PackedFunc(ffi::Function&& other) : ffi::Function(std::move(other)) {}
  PackedFunc& operator=(const ffi::Function& other) {
    ffi::Function::operator=(other);
    return *this;
  }
  /*!
   * \brief Constructing a packed function from a callable type
   *        whose signature is consistent with `PackedFunc`
   * \param data the internal container of packed function.
   */
  template <typename TCallable,
            typename = std::enable_if_t<
                std::is_convertible<TCallable, std::function<void(TVMArgs, Any*)>>::value &&
                !std::is_base_of<TCallable, PackedFunc>::value>>
  explicit PackedFunc(TCallable legacy_packed) {
    PackedFunc ret;
    auto f = [legacy_packed](int num_args, const ffi::AnyView* args, ffi::Any* rv) {
      std::vector<TVMValue> values(num_args);
      std::vector<int> type_codes(num_args);
      for (int i = 0; i < num_args; ++i) {
        AnyViewToLegacyTVMArgValue(args[i].CopyToTVMFFIAny(), &values[i], &type_codes[i]);
      }
      legacy_packed(TVMArgs(values.data(), type_codes.data(), num_args), rv);
    };
    *this = ffi::Function::FromPacked(f);
  }
  /*!
   * \brief Call the function in legacy packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  static TVM_ALWAYS_INLINE void LegacyCallPacked(ffi::FunctionObj* ffi_func, TVMArgs args,
                                                 Any* rv) {
    std::vector<ffi::AnyView> args_vec(args.size());
    for (int i = 0; i < args.size(); ++i) {
      args_vec[i] = args[i];
    }
    // redirect to the normal call packed.
    ffi_func->CallPacked(args.size(), args_vec.data(), rv);
  }
  /*!
   * \brief Call the function in legacy packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  TVM_ALWAYS_INLINE void CallPacked(TVMArgs args, Any* rv) const {
    LegacyCallPacked(static_cast<ffi::FunctionObj*>(data_.get()), args, rv);
  }
};

template <typename FType>
using TypedPackedFunc = ffi::TypedFunction<FType>;

// ObjectRef related conversion handling
// Object can have three possible type codes:
//      kTVMNDArrayHandle, kTVMModuleHandle, kTVMObjectHandle
//
// We use type traits to eliminate un-necessary checks.
template <typename T>
inline void TVMArgsSetter::SetObject(size_t i, T&& value) const {
  using ContainerType = typename std::remove_reference<T>::type::ContainerType;
  if (!value.defined()) {
    type_codes_[i] = kTVMNullptr;
    values_[i].v_handle = nullptr;
    return;
  }

  Object* ptr = ffi::details::ObjectUnsafe::GetRawObjectPtrFromObjectRef(value);
  if constexpr (std::is_base_of_v<NDArray::ContainerType, ContainerType> ||
                std::is_base_of_v<ContainerType, NDArray::ContainerType>) {
    if (std::is_base_of_v<NDArray::ContainerType, ContainerType> ||
        ptr->IsInstance<NDArray::ContainerType>()) {
      values_[i].v_handle = NDArray::FFIGetHandle(value);
      type_codes_[i] = kTVMNDArrayHandle;
      return;
    }
  }

  if constexpr (std::is_base_of_v<Module::ContainerType, ContainerType> ||
                std::is_base_of_v<ContainerType, Module::ContainerType>) {
    if (std::is_base_of_v<Module::ContainerType, ContainerType> ||
        ptr->IsInstance<Module::ContainerType>()) {
      values_[i].v_handle = ptr;
      type_codes_[i] = kTVMModuleHandle;
      return;
    }
  }

  if constexpr (std::is_base_of_v<PackedFunc::ContainerType, ContainerType> ||
                std::is_base_of_v<ContainerType, PackedFunc::ContainerType>) {
    if (std::is_base_of_v<PackedFunc::ContainerType, ContainerType> ||
        ptr->IsInstance<PackedFunc::ContainerType>()) {
      values_[i].v_handle = ptr;
      type_codes_[i] = kTVMPackedFuncHandle;
      return;
    }
  }

  // Like with BoxInt, unwrap any BoxBool instances.  See the BoxInt
  // explanation for more detail.
  if constexpr (std::is_base_of_v<Bool::ContainerType, ContainerType> ||
                std::is_base_of_v<ContainerType, Bool::ContainerType>) {
    if (std::is_base_of_v<Bool::ContainerType, ContainerType> ||
        ptr->IsInstance<Bool::ContainerType>()) {
      values_[i].v_int64 = static_cast<Bool::ContainerType*>(ptr)->value;
      type_codes_[i] = kTVMArgBool;
      return;
    }
  }

  // If a boxed integer is being returned, always unbox it to the
  // primitive type.  This must be checked at the PackedFunc level to
  // ensure that a boxed primitive argument is round-tripped correctly
  // when the boxing is no longer required.
  //
  // For example, consider a PackedFunc with signature `ObjectRef
  // func(Array<ObjectRef>)`, and returns the first element of that
  // array.  When passing a Python array `[5, 17.5, "hello"]`, the
  // items are converted to `[Box<i64>(5), Box<double>(17.5),
  // String("hello")]` in order to provide an `Array<ObjectRef>`.
  //
  // If we had no additional conversions, the caller would receive the
  // return value as a `Box<i64>(5)`, which would be unexpected and
  // require additional unwrapping.  We could perform this check
  // inside the PackedFunc, but that would require a large amount of
  // duplicated checked, and would require explicit handling of
  // `TVMRetValue`.  Instead, this conversion is checked in the FFI
  // return value, to ensure that boxing/unboxing is applied
  // consistently.
  if constexpr (std::is_base_of_v<Int::ContainerType, ContainerType> ||
                std::is_base_of_v<ContainerType, Int::ContainerType>) {
    if (std::is_base_of_v<Int::ContainerType, ContainerType> ||
        ptr->IsInstance<Int::ContainerType>()) {
      values_[i].v_int64 = static_cast<Int::ContainerType*>(ptr)->value;
      type_codes_[i] = kTVMArgInt;
      return;
    }
  }

  // Like with BoxInt, unwrap any BoxFloat instances.  See the BoxInt
  // explanation for more detail.
  if constexpr (std::is_base_of_v<Float::ContainerType, ContainerType> ||
                std::is_base_of_v<ContainerType, Float::ContainerType>) {
    if (std::is_base_of_v<Float::ContainerType, ContainerType> ||
        ptr->IsInstance<Float::ContainerType>()) {
      values_[i].v_float64 = static_cast<Float::ContainerType*>(ptr)->value;
      type_codes_[i] = kTVMArgFloat;
      return;
    }
  }
  // Final fallback, if the ObjectRef has no special cases that must
  // be expressed within the TVMRetValue.
  // NOTE: we remove the RValueRef handling to keep things simple
  // we can add it back if needed later
  values_[i].v_handle = ffi::details::ObjectUnsafe::GetTVMFFIObjectPtrFromObjectRef(value);
  type_codes_[i] = kTVMObjectHandle;
}

inline PackedFunc Module::GetFunction(const String& name, bool query_imports) {
  return (*this)->GetFunction(name, query_imports);
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

namespace parameter_pack {
template <typename... EnumArgs>
struct EnumeratedParamPack {
  struct InvokeWithoutArg {
    template <template <size_t i, typename TArgument> class Functor, typename ExtraParams>
    static void F(ExtraParams&& extra_params) {
      using TExpander = int[];
      (void)TExpander{
          0,
          (Functor<EnumArgs::i, typename EnumArgs::T>::F(std::forward<ExtraParams>(extra_params)),
           0)...,
      };
    }
  };
  struct InvokeWithArg {
    template <template <size_t i, typename TArgument> class Functor, typename ExtraParams,
              typename... Params>
    static void F(ExtraParams&& extra_params, Params&&... params) {
      using TExpander = int[];
      (void)TExpander{
          0,
          (Functor<EnumArgs::i, typename EnumArgs::T>::F(std::forward<ExtraParams>(extra_params),
                                                         std::forward<Params>(params)),
           0)...,
      };
    }
  };
};

template <typename... Args>
struct EnumerateImpl {
 private:
  template <size_t _i, typename _T>
  struct Item {
    static const constexpr size_t i = _i;
    using T = _T;
  };

  template <typename...>
  struct Zipper;

  template <std::size_t... id>
  struct Zipper<std::integer_sequence<std::size_t, id...>> {
    using WithoutArg = typename EnumeratedParamPack<Item<id, Args>...>::InvokeWithoutArg;
    using WithArg = typename EnumeratedParamPack<Item<id, Args>...>::InvokeWithArg;
  };

 public:
  using WithoutArg = typename Zipper<std::index_sequence_for<Args...>>::WithoutArg;
  using WithArg = typename Zipper<std::index_sequence_for<Args...>>::WithArg;
};

template <typename... Args>
using EnumerateWithoutArg = typename EnumerateImpl<Args...>::WithoutArg;

template <typename... Args>
using EnumerateWithArg = typename EnumerateImpl<Args...>::WithArg;

template <typename... Args>
struct ParamPack {
  template <template <size_t i, typename TArgument> class Functor, typename ExtraParams>
  static void InvokeWithoutArg(ExtraParams&& extra_params) {
    EnumerateWithoutArg<Args...>::template F<Functor, ExtraParams>(
        std::forward<ExtraParams>(extra_params));
  }
};
}  // namespace parameter_pack
}  // namespace details

template <size_t i, typename T>
struct TVMArgsSetterApply {
  static TVM_ALWAYS_INLINE void F(TVMArgsSetter* setter, T&& value) {
    (*setter)(i, std::forward<T>(value));
  }
};

template <typename... Args>
void TVM_ALWAYS_INLINE PackArgs(TVMValue* values, int* type_codes, Args&&... args) {
  TVMArgsSetter setter(values, type_codes);
  details::parameter_pack::EnumerateWithArg<Args...>::template F<TVMArgsSetterApply>(
      &setter, std::forward<Args>(args)...);
}



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
    return PackedFunc([_self](TVMArgs args, TVMRetValue* rv) -> void {                            \
      using Helper = ::tvm::runtime::details::ModuleVTableEntryHelper<decltype(MemFunc)>;          \
      SelfPtr self = static_cast<SelfPtr>(_self.get());                                           \
      CHECK_EQ(args.size(), Helper::LenArgs)                                                      \
          << "Function `" << self->type_key() << "::" << Name << "` requires " << Helper::LenArgs \
          << " arguments, but got " << args.size();                                               \
      Helper::Call(rv, self, MemFunc, args, Helper::IndexSeq{});                                  \
    });                                                                                           \
  }
#define TVM_MODULE_VTABLE_ENTRY_PACKED(Name, MemFunc)                  \
  if (_name == Name) {                                                 \
    return PackedFunc([_self](TVMArgs args, TVMRetValue* rv) -> void { \
      (static_cast<SelfPtr>(_self.get())->*(MemFunc))(args, rv);       \
    });                                                                \
  }


}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_PACKED_FUNC_H_
