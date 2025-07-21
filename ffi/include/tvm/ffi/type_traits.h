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
 * \file tvm/ffi/object.h
 * \brief A managed object in the TVM FFI.
 */
#ifndef TVM_FFI_TYPE_TRAITS_H_
#define TVM_FFI_TYPE_TRAITS_H_

#include <tvm/ffi/base_details.h>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <string>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief TypeTraits that specifies the conversion behavior from/to FFI Any.
 *
 * The function specifications of TypeTraits<T>
 *
 * - CopyToAnyView: Convert a value T to AnyView
 * - MoveToAny: Move a value to Any
 * - CheckAnyStrict: Check if a Any stores a result of CopyToAnyView of current T.
 * - CopyFromAnyViewAfterCheck: Copy a value T from Any view after we pass CheckAnyStrict.
 * - MoveFromAnyAfterCheck: Move a value T from Any storage after we pass CheckAnyStrict.
 * - TryCastFromAnyView: Convert a AnyView to a T, we may apply type conversion.
 * - GetMismatchTypeInfo: Get the type key of a type when TryCastFromAnyView fails.
 * - TypeStr: Get the type key of a type
 *
 * It is possible that CheckAnyStrict is false but TryCastFromAnyView still works.
 *
 * For example, when Any x stores int, TypeTraits<float>::CheckAnyStrict(x) will be false,
 * but TypeTraits<float>::TryCastFromAnyView(x) will return a corresponding float value
 * via type conversion.
 *
 * CheckAnyStrict is mainly used in recursive container such as Array<T> to
 * decide if a new Array needed to be created via recursive conversion,
 * or we can use the current container as is when converting to Array<T>.
 *
 * A container array: Array<T> satisfies the following invariant:
 * - `all(TypeTraits<T>::CheckAnyStrict(x) for x in the array)`.
 */
template <typename, typename = void>
struct TypeTraits {
  /*! \brief Whether the type is enabled in FFI. */
  static constexpr bool convert_enabled = false;
  /*! \brief Whether the type can appear as a storage type in Container */
  static constexpr bool storage_enabled = false;
};

/*!
 * \brief TypeTraits that removes const and reference keywords.
 * \tparam T the original type
 */
template <typename T>
using TypeTraitsNoCR = TypeTraits<std::remove_const_t<std::remove_reference_t<T>>>;

template <typename T>
inline constexpr bool use_default_type_traits_v = true;

struct TypeTraitsBase {
  static constexpr bool convert_enabled = true;
  static constexpr bool storage_enabled = true;
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIAny;
  // get mismatched type when result mismatches the trait.
  // this function is called after TryCastFromAnyView fails
  // to get more detailed type information in runtime
  // especially when the error involves nested container type
  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* source) {
    return TypeIndexToTypeKey(source->type_index);
  }
};

template <typename T, typename = void>
struct TypeToFieldStaticTypeIndex {
  static constexpr int32_t value = TypeIndex::kTVMFFIAny;
};

template <typename T>
struct TypeToFieldStaticTypeIndex<T, std::enable_if_t<TypeTraits<T>::convert_enabled>> {
  static constexpr int32_t value = TypeTraits<T>::field_static_type_index;
};

template <typename T, typename = void>
struct TypeToRuntimeTypeIndex {
  static int32_t v() { return TypeToFieldStaticTypeIndex<T>::value; }
};

template <typename T>
struct TypeToRuntimeTypeIndex<T, std::enable_if_t<std::is_base_of_v<ObjectRef, T>>> {
  static int32_t v() { return T::ContainerType::RuntimeTypeIndex(); }
};

// None
template <>
struct TypeTraits<std::nullptr_t> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFINone;

  TVM_FFI_INLINE static void CopyToAnyView(const std::nullptr_t&, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    // invariant: the pointer field also equals nullptr
    // this will simplify same_as comparisons and hash
    result->v_int64 = 0;
  }

  TVM_FFI_INLINE static void MoveToAny(std::nullptr_t, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    // invariant: the pointer field also equals nullptr
    // this will simplify same_as comparisons and hash
    result->v_int64 = 0;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFINone;
  }

  TVM_FFI_INLINE static std::nullptr_t CopyFromAnyViewAfterCheck(const TVMFFIAny*) {
    return nullptr;
  }

  TVM_FFI_INLINE static std::nullptr_t MoveFromAnyAfterCheck(TVMFFIAny*) { return nullptr; }

  TVM_FFI_INLINE static std::optional<std::nullptr_t> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return nullptr;
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFINone; }
};

/**
 * \brief A type that forbids implicit conversion from int to bool
 *
 * This type is used to prevent implicit conversion from int to bool.
 */
class StrictBool {
 public:
  StrictBool(bool value) : value_(value) {}  // NOLINT(*)
  operator bool() const { return value_; }

 private:
  bool value_;
};

template <>
struct TypeTraits<StrictBool> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;

  TVM_FFI_INLINE static void CopyToAnyView(const StrictBool& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIBool;
    result->v_int64 = static_cast<bool>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(StrictBool src, TVMFFIAny* result) {
    CopyToAnyView(src, result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIBool;
  }

  TVM_FFI_INLINE static StrictBool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<bool>(src->v_int64);
  }

  TVM_FFI_INLINE static StrictBool MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<StrictBool> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIBool) {
      return StrictBool(static_cast<bool>(src->v_int64));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
};

// Bool type, allow implicit casting from int
template <>
struct TypeTraits<bool> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;

  TVM_FFI_INLINE static void CopyToAnyView(const bool& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIBool;
    result->v_int64 = static_cast<int64_t>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(bool src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIBool;
  }

  TVM_FFI_INLINE static bool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<bool>(src->v_int64);
  }

  TVM_FFI_INLINE static bool MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<bool> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
      return static_cast<bool>(src->v_int64);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
};

// Integer POD values
template <typename Int>
struct TypeTraits<Int, std::enable_if_t<std::is_integral_v<Int>>> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIInt;

  TVM_FFI_INLINE static void CopyToAnyView(const Int& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIInt;
    result->v_int64 = static_cast<int64_t>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(Int src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
    return src->type_index == TypeIndex::kTVMFFIInt;
  }

  TVM_FFI_INLINE static Int CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<Int>(src->v_int64);
  }

  TVM_FFI_INLINE static Int MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Int> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
      return Int(src->v_int64);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIInt; }
};

// Enum Integer POD values
template <typename IntEnum>
struct TypeTraits<IntEnum, std::enable_if_t<std::is_enum_v<IntEnum> &&
                                            std::is_integral_v<std::underlying_type_t<IntEnum>>>>
    : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIInt;

  TVM_FFI_INLINE static void CopyToAnyView(const IntEnum& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIInt;
    result->v_int64 = static_cast<int64_t>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(IntEnum src, TVMFFIAny* result) {
    CopyToAnyView(src, result);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
    return src->type_index == TypeIndex::kTVMFFIInt;
  }

  TVM_FFI_INLINE static IntEnum CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<IntEnum>(src->v_int64);
  }

  TVM_FFI_INLINE static IntEnum MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<IntEnum> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
      return static_cast<IntEnum>(src->v_int64);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIInt; }
};

// Float POD values
template <typename Float>
struct TypeTraits<Float, std::enable_if_t<std::is_floating_point_v<Float>>>
    : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFloat;

  TVM_FFI_INLINE static void CopyToAnyView(const Float& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIFloat;
    result->v_float64 = static_cast<double>(src);
  }

  TVM_FFI_INLINE static void MoveToAny(Float src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
    return src->type_index == TypeIndex::kTVMFFIFloat;
  }

  TVM_FFI_INLINE static Float CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<Float>(src->v_float64);
  }

  TVM_FFI_INLINE static Float MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Float> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIFloat) {
      return Float(src->v_float64);
    } else if (src->type_index == TypeIndex::kTVMFFIInt ||
               src->type_index == TypeIndex::kTVMFFIBool) {
      return Float(src->v_int64);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIFloat; }
};

// void*
template <>
struct TypeTraits<void*> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIOpaquePtr;

  TVM_FFI_INLINE static void CopyToAnyView(void* src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIOpaquePtr;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_ptr = src;
  }

  TVM_FFI_INLINE static void MoveToAny(void* src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    // NOTE: CheckAnyStrict is always strict and should be consistent with MoveToAny
    return src->type_index == TypeIndex::kTVMFFIOpaquePtr;
  }

  TVM_FFI_INLINE static void* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) { return src->v_ptr; }

  TVM_FFI_INLINE static void* MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<void*> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIOpaquePtr) {
      return static_cast<void*>(src->v_ptr);
    }
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return static_cast<void*>(nullptr);
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIOpaquePtr; }
};

// Device
template <>
struct TypeTraits<DLDevice> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDevice;

  TVM_FFI_INLINE static void CopyToAnyView(const DLDevice& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDevice;
    result->v_device = src;
  }

  TVM_FFI_INLINE static void MoveToAny(DLDevice src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDevice;
    result->v_device = src;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDevice;
  }

  TVM_FFI_INLINE static DLDevice CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return src->v_device;
  }

  TVM_FFI_INLINE static DLDevice MoveFromAnyAfterCheck(TVMFFIAny* src) {
    // POD type, we can just copy the value
    return CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<DLDevice> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDevice) {
      return src->v_device;
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIDevice; }
};

// DLTensor*, requirement: not nullable, do not retain ownership
template <>
struct TypeTraits<DLTensor*> : public TypeTraitsBase {
  static constexpr bool storage_enabled = false;
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDLTensorPtr;

  TVM_FFI_INLINE static void CopyToAnyView(DLTensor* src, TVMFFIAny* result) {
    TVM_FFI_ICHECK_NOTNULL(src);
    result->type_index = TypeIndex::kTVMFFIDLTensorPtr;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_ptr = src;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDLTensorPtr;
  }

  TVM_FFI_INLINE static DLTensor* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<DLTensor*>(src->v_ptr);
  }

  TVM_FFI_INLINE static void MoveToAny(DLTensor*, TVMFFIAny*) {
    TVM_FFI_THROW(RuntimeError)
        << "DLTensor* cannot be held in Any as it does not retain ownership, use NDArray instead";
  }

  TVM_FFI_INLINE static std::optional<DLTensor*> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDLTensorPtr) {
      return static_cast<DLTensor*>(src->v_ptr);
    } else if (src->type_index == TypeIndex::kTVMFFINDArray) {
      // Conversion from NDArray pointer to DLTensor
      // based on the assumption that NDArray always follows the TVMFFIObject header
      static_assert(sizeof(TVMFFIObject) == 16, "TVMFFIObject must be 8 bytes");
      return reinterpret_cast<DLTensor*>(reinterpret_cast<char*>(src->v_obj) +
                                         sizeof(TVMFFIObject));
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return "DLTensor*"; }
};

// Traits for ObjectRef, None to ObjectRef will always fail.
// use std::optional<ObjectRef> instead for nullable references.
template <typename TObjRef>
struct ObjectRefTypeTraitsBase : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIObject;
  using ContainerType = typename TObjRef::ContainerType;

  TVM_FFI_INLINE static void CopyToAnyView(const TObjRef& src, TVMFFIAny* result) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (!src.defined()) {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
        return;
      }
    }
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::TVMFFIObjectPtrFromObjectRef(src);
    result->type_index = obj_ptr->type_index;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  TVM_FFI_INLINE static void MoveToAny(TObjRef src, TVMFFIAny* result) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (!src.defined()) {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
        return;
      }
    }
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::MoveObjectRefToTVMFFIObjectPtr(std::move(src));
    result->type_index = obj_ptr->type_index;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) return true;
    }
    return (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
            details::IsObjectInstance<ContainerType>(src->type_index));
  }

  TVM_FFI_INLINE static TObjRef CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return TObjRef(ObjectPtr<Object>(nullptr));
      }
    }
    return TObjRef(details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj));
  }

  TVM_FFI_INLINE static TObjRef MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return TObjRef(ObjectPtr<Object>(nullptr));
      }
    }
    // move out the object pointer
    ObjectPtr<Object> obj_ptr = details::ObjectUnsafe::ObjectPtrFromOwned<Object>(src->v_obj);
    // reset the src to nullptr
    TypeTraits<std::nullptr_t>::MoveToAny(nullptr, src);
    return TObjRef(std::move(obj_ptr));
  }

  TVM_FFI_INLINE static std::optional<TObjRef> TryCastFromAnyView(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return TObjRef(ObjectPtr<Object>(nullptr));
      }
    }
    if (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      if (details::IsObjectInstance<ContainerType>(src->type_index)) {
        return TObjRef(details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj));
      }
    }
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return ContainerType::_type_key; }
};

template <typename TObjRef>
struct TypeTraits<TObjRef, std::enable_if_t<std::is_base_of_v<ObjectRef, TObjRef> &&
                                            use_default_type_traits_v<TObjRef>>>
    : public ObjectRefTypeTraitsBase<TObjRef> {};

/*!
 * \brief Helper class that convert to T only via the FallbackTypes
 *
 * The conversion will go through the FallbackTypes in the order
 * specified in the template parameter.
 * \tparam T The type of the target value.
 * \tparam FallbackTypes The type of the fallback value.
 * \note TypeTraits<T> must be derived from this class and define
 *     ConvertFallbackValue(FallbackType)->T for each FallbackType
 */
template <typename T, typename... FallbackTypes>
struct FallbackOnlyTraitsBase : public TypeTraitsBase {
  // disable container for FallbackOnlyTraitsBase
  static constexpr bool storage_enabled = false;

  TVM_FFI_INLINE static std::optional<T> TryCastFromAnyView(const TVMFFIAny* src) {
    return TryFallbackTypes<FallbackTypes...>(src);
  }

  template <typename FallbackType, typename... Rest>
  TVM_FFI_INLINE static std::optional<T> TryFallbackTypes(const TVMFFIAny* src) {
    static_assert(!std::is_same_v<bool, FallbackType>,
                  "Using bool as FallbackType can cause bug because int will be detected as bool, "
                  "use tvm::ffi::StrictBool instead");
    if (auto opt_fallback = TypeTraits<FallbackType>::TryCastFromAnyView(src)) {
      return TypeTraits<T>::ConvertFallbackValue(*std::move(opt_fallback));
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryFallbackTypes<Rest...>(src);
    }
    return std::nullopt;
  }
};

/*!
 * \brief Helper class to define ObjectRef that can be auto-converted from a
 *        fallback type, the Traits<ObjectRefType> must be derived from it
 *        and define a static methods named ConvertFallbackValue for each
 *        FallbackType
 *
 *        The conversion will go through the FallbackTypes in the order
 *        specified in the template parameter.
 * \tparam ObjectRefType The type of the ObjectRef.
 * \tparam FallbackTypes The type of the fallback value.
 */
template <typename ObjectRefType, typename... FallbackTypes>
struct ObjectRefWithFallbackTraitsBase : public ObjectRefTypeTraitsBase<ObjectRefType> {
  TVM_FFI_INLINE static std::optional<ObjectRefType> TryCastFromAnyView(const TVMFFIAny* src) {
    if (auto opt_obj = ObjectRefTypeTraitsBase<ObjectRefType>::TryCastFromAnyView(src)) {
      return *opt_obj;
    }
    // apply fallback types in TryCastFromAnyView
    return TryFallbackTypes<FallbackTypes...>(src);
  }

  template <typename FallbackType, typename... Rest>
  TVM_FFI_INLINE static std::optional<ObjectRefType> TryFallbackTypes(const TVMFFIAny* src) {
    static_assert(!std::is_same_v<bool, FallbackType>,
                  "Using bool as FallbackType can cause bug because int will be detected as bool, "
                  "use tvm::ffi::StrictBool instead");
    if (auto opt_fallback = TypeTraits<FallbackType>::TryCastFromAnyView(src)) {
      return TypeTraits<ObjectRefType>::ConvertFallbackValue(*std::move(opt_fallback));
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryFallbackTypes<Rest...>(src);
    }
    return std::nullopt;
  }
};

// Traits for weak pointer of object
// NOTE: we require the weak pointer cast from

template <typename TObject>
struct TypeTraits<TObject*, std::enable_if_t<std::is_base_of_v<Object, TObject>>>
    : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(TObject* src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetHeader(src);
    result->type_index = obj_ptr->type_index;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
  }

  TVM_FFI_INLINE static void MoveToAny(TObject* src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetHeader(src);
    result->type_index = obj_ptr->type_index;
    TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
    result->v_obj = obj_ptr;
    // needs to increase ref because original weak ptr do not own the code
    details::ObjectUnsafe::IncRefObjectHandle(result->v_obj);
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    return src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
           details::IsObjectInstance<TObject>(src->type_index);
  }

  TVM_FFI_INLINE static TObject* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if constexpr (!std::is_const_v<TObject>) {
      static_assert(TObject::_type_mutable, "TObject must be mutable to enable cast from Any");
    }
    return details::ObjectUnsafe::RawObjectPtrFromUnowned<TObject>(src->v_obj);
  }

  TVM_FFI_INLINE static std::optional<TObject*> TryCastFromAnyView(const TVMFFIAny* src) {
    if constexpr (!std::is_const_v<TObject>) {
      static_assert(TObject::_type_mutable, "TObject must be mutable to enable cast from Any");
    }
    if (CheckAnyStrict(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }

  TVM_FFI_INLINE static std::string TypeStr() { return TObject::_type_key; }
};

template <typename T>
inline constexpr bool use_default_type_traits_v<Optional<T>> = false;

template <typename T>
struct TypeTraits<Optional<T>> : public TypeTraitsBase {
  TVM_FFI_INLINE static void CopyToAnyView(const Optional<T>& src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::CopyToAnyView(*src, result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }

  TVM_FFI_INLINE static void MoveToAny(Optional<T> src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::MoveToAny(*std::move(src), result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return true;
    return TypeTraits<T>::CheckAnyStrict(src);
  }

  TVM_FFI_INLINE static Optional<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return Optional<T>(std::nullopt);
    }
    return TypeTraits<T>::CopyFromAnyViewAfterCheck(src);
  }

  TVM_FFI_INLINE static Optional<T> MoveFromAnyAfterCheck(TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return Optional<T>(std::nullopt);
    }
    return TypeTraits<T>::MoveFromAnyAfterCheck(src);
  }

  TVM_FFI_INLINE static std::optional<Optional<T>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Optional<T>(std::nullopt);
    if (std::optional<T> opt = TypeTraits<T>::TryCastFromAnyView(src)) {
      return Optional<T>(*std::move(opt));
    } else {
      // important to be explicit here
      // because nullopt can convert to std::optional<T>(nullopt) which indicate success
      // return std::optional<Optional<T>>(std::nullopt) to indicate failure
      return std::optional<Optional<T>>(std::nullopt);
    }
  }

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraits<T>::GetMismatchTypeInfo(src);
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "Optional<" + TypeTraits<T>::TypeStr() + ">";
  }
};
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_TYPE_TRAITS_H_
