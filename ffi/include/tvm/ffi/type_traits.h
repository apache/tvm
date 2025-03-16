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

#include <optional>
#include <string>
#include <type_traits>

namespace tvm {
namespace ffi {

/*!
 * \brief Get type key from type index
 * \param type_index The input type index
 * \return the type key
 */
inline std::string TypeIndex2TypeKey(int32_t type_index) {
  switch (type_index) {
    case TypeIndex::kTVMFFINone:
      return "None";
    case TypeIndex::kTVMFFIBool:
      return "bool";
    case TypeIndex::kTVMFFIInt:
      return "int";
    case TypeIndex::kTVMFFIFloat:
      return "float";
    case TypeIndex::kTVMFFIOpaquePtr:
      return "void*";
    case TypeIndex::kTVMFFIDataType:
      return "DataType";
    case TypeIndex::kTVMFFIDevice:
      return "Device";
    case TypeIndex::kTVMFFIRawStr:
      return "const char*";
    default: {
      TVM_FFI_ICHECK_GE(type_index, TypeIndex::kTVMFFIStaticObjectBegin)
          << "Uknown type_index=" << type_index;
      const TypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
      return type_info->type_key;
    }
  }
}
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
    return TypeIndex2TypeKey(source->type_index);
  }
};

// None
template <>
struct TypeTraits<std::nullptr_t> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFINone;

  static TVM_FFI_INLINE void CopyToAnyView(const std::nullptr_t&, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    // invariant: the pointer field also equals nullptr
    // this will simplify the recovery of nullable object from the any
    result->v_int64 = 0;
  }

  static TVM_FFI_INLINE void MoveToAny(std::nullptr_t, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFINone;
    // invariant: the pointer field also equals nullptr
    // this will simplify the recovery of nullable object from the any
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

  static TVM_FFI_INLINE std::string TypeStr() { return "None"; }
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
      return std::make_optional<Int>(src->v_int64);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool;
  }

  static TVM_FFI_INLINE int CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<Int>(src->v_int64);
  }

  static TVM_FFI_INLINE std::string TypeStr() { return "int"; }
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
      return std::make_optional<bool>(static_cast<bool>(src->v_int64));
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIInt || src->type_index == TypeIndex::kTVMFFIBool;
  }

  static TVM_FFI_INLINE bool CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<bool>(src->v_int64);
  }

  static TVM_FFI_INLINE std::string TypeStr() { return "bool"; }
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
      return std::make_optional<Float>(src->v_float64);
    } else if (src->type_index == TypeIndex::kTVMFFIInt ||
               src->type_index == TypeIndex::kTVMFFIBool) {
      return std::make_optional<Float>(src->v_int64);
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

  static TVM_FFI_INLINE std::string TypeStr() { return "float"; }
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
      return std::make_optional<void*>(src->v_ptr);
    }
    if (src->type_index == TypeIndex::kTVMFFINone) {
      return std::make_optional<void*>(nullptr);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIOpaquePtr ||
           src->type_index == TypeIndex::kTVMFFINone;
  }

  static TVM_FFI_INLINE void* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) { return src->v_ptr; }

  static TVM_FFI_INLINE std::string TypeStr() { return "void*"; }
};

// DataType
template <>
struct TypeTraits<DLDataType> : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDataType;

  static TVM_FFI_INLINE void CopyToAnyView(const DLDataType& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  static TVM_FFI_INLINE void MoveToAny(DLDataType src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIDataType;
    result->v_dtype = src;
  }

  static TVM_FFI_INLINE std::optional<DLDataType> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIDataType) {
      return src->v_dtype;
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDataType;
  }

  static TVM_FFI_INLINE DLDataType CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return src->v_dtype;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return "DataType"; }
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

  static TVM_FFI_INLINE std::string TypeStr() { return "Device"; }
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
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index == TypeIndex::kTVMFFIDLTensorPtr;
  }

  static TVM_FFI_INLINE DLTensor* CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return static_cast<DLTensor*>(src->v_ptr);
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

// Traits for ObjectRef
template <typename TObjRef>
struct ObjectRefTypeTraitsBase : public TypeTraitsBase {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIObject;
  using ContainerType = typename TObjRef::ContainerType;

  static TVM_FFI_INLINE void CopyToAnyView(const TObjRef& src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetTVMFFIObjectPtrFromObjectRef(src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE void MoveToAny(TObjRef src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::MoveTVMFFIObjectPtrFromObjectRef(&src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
            details::IsObjectInstance<ContainerType>(src->type_index)) ||
           (src->type_index == kTVMFFINone && TObjRef::_type_is_nullable);
  }

  static TVM_FFI_INLINE TObjRef CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == kTVMFFINone) return TObjRef(nullptr);
    }
    return TObjRef(details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj));
  }

  static TVM_FFI_INLINE std::optional<TObjRef> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      if (details::IsObjectInstance<ContainerType>(src->type_index)) {
        return TObjRef(details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(src->v_obj));
      }
    }
    if constexpr (TObjRef::_type_is_nullable) {
      if (src->type_index == kTVMFFINone) return TObjRef(nullptr);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return ContainerType::_type_key; }
};

template <typename TObjRef>
struct TypeTraits<TObjRef, std::enable_if_t<std::is_base_of_v<ObjectRef, TObjRef> &&
                                            use_default_type_traits_v<TObjRef>>>
    : public ObjectRefTypeTraitsBase<TObjRef> {};

// Traits for ObjectPtr
template <typename T>
struct TypeTraits<ObjectPtr<T>> : public TypeTraitsBase {
  static TVM_FFI_INLINE void CopyToAnyView(const ObjectPtr<T>& src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::GetTVMFFIObjectPtrFromObjectPtr(src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE void MoveToAny(ObjectPtr<T> src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectUnsafe::MoveTVMFFIObjectPtrFromObjectPtr(&src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE bool CheckAnyView(const TVMFFIAny* src) {
    return src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin &&
           details::IsObjectInstance<T>(src->type_index);
  }

  static TVM_FFI_INLINE ObjectPtr<T> CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
    return details::ObjectUnsafe::ObjectPtrFromUnowned<T>(src->v_obj);
  }

  static TVM_FFI_INLINE std::optional<ObjectPtr<T>> TryCopyFromAnyView(const TVMFFIAny* src) {
    if (CheckAnyView(src)) return CopyFromAnyViewAfterCheck(src);
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return T::_type_key; }
};

// Traits for weak pointer of object
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

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_TYPE_TRAITS_H_
