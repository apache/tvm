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

namespace tvm {
namespace ffi {

/*!
 * \brief TypeTraits that specifies the conversion behavior from/to FFI Any.
 *
 * We need to implement the following conversion functions
 *
 * - void CopyToAnyView(const T& src, TVMFFIAny* result);
 *
 *   Convert a value to AnyView
 *
 * - std::optional<T> TryCopyFromAnyView(const TVMFFIAny* src);
 *
 *   Try convert AnyView to a value type.
 */
template <typename, typename = void>
struct TypeTraits {
  static constexpr bool enabled = false;
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
  static constexpr bool enabled = true;

  // get mismatched type when result mismatches the trait.
  // this function is called after TryCopyFromAnyView fails
  // to get more detailed type information in runtime
  // especially when the error involves nested container type
  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* source) {
    return TypeIndexToTypeKey(source->type_index);
  }
};

// None
template <>
struct TypeTraits<std::nullptr_t> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFINone;

  static TVM_FFI_INLINE void CopyToAnyView(const std::nullptr_t&, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    // invariant: the pointer field also equals nullptr
    // this will simplify same_as comparisons and hash
    result->v_int64 = 0;
  }

  static TVM_FFI_INLINE void MoveToAny(std::nullptr_t, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    // invariant: the pointer field also equals nullptr
    // this will simplify same_as comparisons and hash
    result->v_int64 = 0;
  }

  static TVM_FFI_INLINE std::optional<std::nullptr_t> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return nullptr;
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFINone;
  }

  static TVM_FFI_INLINE std::nullptr_t CopyFromAnyViewAfterCheck(const TVMFFIAny*) {
    return nullptr;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return StaticTypeKey::kTVMFFINone; }
};

/**
 * \brief A type that forbids implicit conversion from int to bool
 *
 * This type is used to prevent implicit conversion from int to bool.
 */
class StrictBool {
 public:
  StrictBool(bool value) : value_(value) {}
  operator bool() const { return value_; }

 private:
  bool value_;
};

template <>
struct TypeTraits<StrictBool> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;

  static TVM_FFI_INLINE void CopyToAnyView(const StrictBool& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIBool;
    result->v_int64 = static_cast<bool>(src);
  }

  static TVM_FFI_INLINE void MoveToAny(StrictBool src, TVMFFIAny* result) {
    CopyToAnyView(src, result);
  }

  static TVM_FFI_INLINE std::optional<StrictBool> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIBool) {
      return StrictBool(static_cast<bool>(src->v_int64));
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIBool;
  }

  static TVM_FFI_INLINE StrictBool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<bool>(src->v_int64);
  }

  static TVM_FFI_INLINE std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
};

// Bool type, allow implicit casting from int
template <>
struct TypeTraits<bool> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIBool;

  static TVM_FFI_INLINE void CopyToAnyView(const bool& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIBool;
    result->v_int64 = static_cast<int64_t>(src);
  }

  static TVM_FFI_INLINE void MoveToAny(bool src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  static TVM_FFI_INLINE std::optional<bool> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
      return static_cast<bool>(src->v_int64);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool;
  }

  static TVM_FFI_INLINE bool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<bool>(src->v_int64);
  }

  static TVM_FFI_INLINE std::string TypeStr() { return StaticTypeKey::kTVMFFIBool; }
};

// Integer POD values
template <typename Int>
struct TypeTraits<Int, std::enable_if_t<std::is_integral_v<Int>>> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIInt;

  static TVM_FFI_INLINE void CopyToAnyView(const Int& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIInt;
    result->v_int64 = static_cast<int64_t>(src);
  }

  static TVM_FFI_INLINE void MoveToAny(Int src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  static TVM_FFI_INLINE std::optional<Int> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool) {
      return Int(src->v_int64);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool;
  }

  static TVM_FFI_INLINE int CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<Int>(src->v_int64);
  }

  static TVM_FFI_INLINE std::string TypeStr() { return StaticTypeKey::kTVMFFIInt; }
};

// Float POD values
template <typename Float>
struct TypeTraits<Float, std::enable_if_t<std::is_floating_point_v<Float>>>
    : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIFloat;

  static TVM_FFI_INLINE void CopyToAnyView(const Float& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIFloat;
    result->v_float64 = static_cast<double>(src);
  }

  static TVM_FFI_INLINE void MoveToAny(Float src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  static TVM_FFI_INLINE std::optional<Float> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIFloat) {
      return Float(src->v_float64);
    } else if (src->type_index == TypeIndex::kTVMFFIInt ||
               src->type_index == TypeIndex::kTVMFFIBool) {
      return Float(src->v_int64);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIFloat || src->type_index == TypeIndex::kTVMFFIInt ||
           src->type_index == TypeIndex::kTVMFFIBool;
  }

  static TVM_FFI_INLINE Float CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIFloat) {
      return static_cast<Float>(src->v_float64);
    } else {
      return static_cast<Float>(src->v_int64);
    }
  }

  static TVM_FFI_INLINE std::string TypeStr() { return StaticTypeKey::kTVMFFIFloat; }
};

// void*
template <>
struct TypeTraits<void*> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIOpaquePtr;

  static TVM_FFI_INLINE void CopyToAnyView(void* src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIOpaquePtr;
    // maintain padding zero in 32bit platform
    if constexpr (sizeof(void*) != sizeof(int64_t)) {
      result->v_int64 = 0;
    }
    result->v_ptr = src;
  }

  static TVM_FFI_INLINE void MoveToAny(void* src, TVMFFIAny* result) { CopyToAnyView(src, result); }

  static TVM_FFI_INLINE std::optional<void*> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIOpaquePtr) {
      return static_cast<void*>(src->v_ptr);
    }
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return static_cast<void*>(nullptr);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIOpaquePtr ||
           src->type_index == TypeIndex::kTVMFFINone;
  }

  static TVM_FFI_INLINE void* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) { return src->v_ptr; }

  static TVM_FFI_INLINE std::string TypeStr() { return StaticTypeKey::kTVMFFIOpaquePtr; }
};

// Device
template <>
struct TypeTraits<DLDevice> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDevice;

  static TVM_FFI_INLINE void CopyToAnyView(const DLDevice& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDevice;
    result->v_device = src;
  }

  static TVM_FFI_INLINE void MoveToAny(DLDevice src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDevice;
    result->v_device = src;
  }

  static TVM_FFI_INLINE std::optional<DLDevice> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDevice) {
      return src->v_device;
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDevice;
  }

  static TVM_FFI_INLINE DLDevice CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return src->v_device;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return StaticTypeKey::kTVMFFIDevice; }
};

// DLTensor*, requirement: not nullable, do not retain ownership
template <>
struct TypeTraits<DLTensor*> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDLTensorPtr;

  static TVM_FFI_INLINE void CopyToAnyView(DLTensor* src, TVMFFIAny* result) {
    TVM_FFI_ICHECK_NOTNULL(src);
    result->type_index = TypeIndex::kTVMFFIDLTensorPtr;
    result->v_ptr = src;
  }

  static TVM_FFI_INLINE void MoveToAny(DLTensor* src, TVMFFIAny* result) {
    TVM_FFI_THROW(RuntimeError)
        << "DLTensor* cannot be held in Any as it does not retain ownership, use NDArray instead";
  }

  static TVM_FFI_INLINE std::optional<DLTensor*> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDLTensorPtr) {
      return static_cast<DLTensor*>(src->v_ptr);
    } else if (src->type_index == TypeIndex::kTVMFFINDArray) {
      // Conversion from NDArray pointer to DLTensor
      // we rely on the fact that NDArray DLTensor field immediately
      // follows the object header
      // TODO(tqchen): followup once we bring NDArray related containers to FFI.
      return reinterpret_cast<DLTensor*>(reinterpret_cast<char*>(src->v_obj) +
                                         sizeof(TVMFFIObject));
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDLTensorPtr ||
           src->type_index == TypeIndex::kTVMFFINDArray;
  }

  static TVM_FFI_INLINE DLTensor* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDLTensorPtr) {
      return static_cast<DLTensor*>(src->v_ptr);
    } else {
      return reinterpret_cast<DLTensor*>(reinterpret_cast<char*>(src->v_obj) +
                                         sizeof(TVMFFIObject));
    }
  }

  static TVM_FFI_INLINE std::string TypeStr() { return "DLTensor*"; }
};

template <int N>
struct TypeTraits<char[N]> : public TypeTraitsBase {
  // NOTE: only enable implicit conversion into AnyView
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIRawStr;

  static TVM_FFI_INLINE void CopyToAnyView(const char src[N], TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIRawStr;
    result->v_c_str = src;
  }
};

// Traits for ObjectRef, None to ObjectRef will always fail.
// use std::optional<ObjectRef> instead for nullable references.
template <typename TObjRef>
struct ObjectRefTypeTraitsBase : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIObject;
  using ContainerType = typename TObjRef::ContainerType;

  static TVM_FFI_INLINE void CopyToAnyView(const TObjRef& src, TVMFFIAny* result) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (!src.defined()) {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
        return;
      }
    }
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetTVMFFIObjectPtrFromObjectRef(src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE void MoveToAny(TObjRef src, TVMFFIAny* result) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (!src.defined()) {
        TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
        return;
      }
    }
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::MoveTVMFFIObjectPtrFromObjectRef(&src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) return true;
    }
    return (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
            details::IsObjectInstance<ContainerType>(src->type_index));
  }

  static TVM_FFI_INLINE TObjRef CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == TypeIndex::kTVMFFINone) {
        return TObjRef(ObjectPtr<Object>(nullptr));
      }
    }
    return TObjRef(details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj));
  }

  static TVM_FFI_INLINE std::optional<TObjRef> TryCopyFromAnyView(const TVMFFIAny* src) {
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

  static TVM_FFI_INLINE std::string TypeStr() { return ContainerType::_type_key; }
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
  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return (TypeTraits<FallbackTypes>::CheckAnyView(src) || ...);
  }
  static TVM_FFI_INLINE T CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return TryFallbackTypes<FallbackTypes...>(src).value();
  }

  static TVM_FFI_INLINE std::optional<T> TryCopyFromAnyView(const TVMFFIAny* src) {
    return TryFallbackTypes<FallbackTypes...>(src);
  }

  template <typename FallbackType, typename... Rest>
  static TVM_FFI_INLINE std::optional<T> TryFallbackTypes(const TVMFFIAny* src) {
    static_assert(!std::is_same_v<bool, FallbackType>,
                  "Using bool as FallbackType can cause bug because int will be detected as bool, "
                  "use tvm::ffi::StrictBool instead");
    if (auto opt_fallback = TypeTraits<FallbackType>::TryCopyFromAnyView(src)) {
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
  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return ObjectRefTypeTraitsBase<ObjectRefType>::CheckAnyView(src) ||
           (TypeTraits<FallbackTypes>::CheckAnyView(src) || ...);
  }

  static TVM_FFI_INLINE ObjectRefType CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (auto opt = ObjectRefTypeTraitsBase<ObjectRefType>::TryCopyFromAnyView(src)) {
      return opt.value();
    }
    return TryFallbackTypes<FallbackTypes...>(src).value();
  }

  static TVM_FFI_INLINE std::optional<ObjectRefType> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (auto opt_obj = ObjectRefTypeTraitsBase<ObjectRefType>::TryCopyFromAnyView(src)) {
      return opt_obj.value();
    }
    return TryFallbackTypes<FallbackTypes...>(src);
  }

  template <typename FallbackType, typename... Rest>
  static TVM_FFI_INLINE std::optional<ObjectRefType> TryFallbackTypes(const TVMFFIAny* src) {
    static_assert(!std::is_same_v<bool, FallbackType>,
                  "Using bool as FallbackType can cause bug because int will be detected as bool, "
                  "use tvm::ffi::StrictBool instead");
    if (auto opt_fallback = TypeTraits<FallbackType>::TryCopyFromAnyView(src)) {
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
struct TypeTraits<const TObject*, std::enable_if_t<std::is_base_of_v<Object, TObject>>>
    : public TypeTraitsBase {
  static TVM_FFI_INLINE void CopyToAnyView(const TObject* src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetHeader(src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE void MoveToAny(const TObject* src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetHeader(src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
    // needs to increase ref because original weak ptr do not own the code
    details::ObjectUnsafe::IncRefObjectInAny(result);
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
           details::IsObjectInstance<TObject>(src->type_index);
  }

  static TVM_FFI_INLINE const TObject* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return details::ObjectUnsafe::RawObjectPtrFromUnowned<TObject>(src->v_obj);
  }

  static TVM_FFI_INLINE std::optional<const TObject*> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyView(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return TObject::_type_key; }
};

template <typename T>
inline constexpr bool use_default_type_traits_v<Optional<T>> = false;

template <typename T>
struct TypeTraits<Optional<T>> : public TypeTraitsBase {
  static TVM_FFI_INLINE void CopyToAnyView(const Optional<T>& src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::CopyToAnyView(*src, result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }

  static TVM_FFI_INLINE void MoveToAny(Optional<T> src, TVMFFIAny* result) {
    if (src.has_value()) {
      TypeTraits<T>::MoveToAny(*std::move(src), result);
    } else {
      TypeTraits<std::nullptr_t>::CopyToAnyView(nullptr, result);
    }
  }

  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    return TypeTraits<T>::GetMismatchTypeInfo(src);
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return true;
    return TypeTraits<T>::CheckAnyView(src);
  }

  static TVM_FFI_INLINE Optional<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return Optional<T>(std::nullopt);
    }
    return TypeTraits<T>::CopyFromAnyViewAfterCheck(src);
  }

  static TVM_FFI_INLINE std::optional<Optional<T>> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFINone) return Optional<T>(std::nullopt);
    if (std::optional<T> opt = TypeTraits<T>::TryCopyFromAnyView(src)) {
      return Optional<T>(*std::move(opt));
    } else {
      // important to be explicit here
      // because nullopt can convert to std::optional<T>(nullopt) which indicate success
      // return std::optional<Optional<T>>(std::nullopt) to indicate failure
      return std::optional<Optional<T>>(std::nullopt);
    }
  }

  static TVM_FFI_INLINE std::string TypeStr() {
    return "Optional<" + TypeTraits<T>::TypeStr() + ">";
  }
};
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_TYPE_TRAITS_H_
