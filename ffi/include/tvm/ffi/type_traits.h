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

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/internal_utils.h>
#include <tvm/ffi/object.h>

#include <optional>
#include <type_traits>

namespace tvm {
namespace ffi {

/*!
 * \brief TypeTraits that specifies the conversion behavior from/to FFI Any.
 *
 * We need to implement the following conversion functions
 *
 * - void ConvertToAnyView(const T& src, TVMFFIAny* result);
 *
 *   Convert a value to AnyView
 *
 * - std::optional<T> TryConvertFromAnyView(const TVMFFIAny* src);
 *
 *   Try convert AnyView to a value type.
 */
template <typename, typename = void>
struct TypeTraits {
  static constexpr bool enabled = false;
};

// Integer POD values
template <typename Int>
struct TypeTraits<Int, std::enable_if_t<std::is_integral_v<Int>>> {
  static constexpr bool enabled = true;

  static TVM_FFI_INLINE void ConvertToAnyView(const Int& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIInt;
    result->v_int64 = static_cast<int64_t>(src);
  }

  static TVM_FFI_INLINE void MoveToManagedAny(Int src, TVMFFIAny* result) {
    ConvertToAnyView(src, result);
  }

  static TVM_FFI_INLINE std::optional<Int> TryConvertFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIInt) {
      return std::make_optional<Int>(src->v_int64);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return "int"; }
};

// Float POD values
template <typename Float>
struct TypeTraits<Float, std::enable_if_t<std::is_floating_point_v<Float>>> {
  static constexpr bool enabled = true;

  static TVM_FFI_INLINE void ConvertToAnyView(const Float& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIFloat;
    result->v_float64 = static_cast<double>(src);
  }

  static TVM_FFI_INLINE void MoveToManagedAny(Float src, TVMFFIAny* result) {
    ConvertToAnyView(src, result);
  }

  static TVM_FFI_INLINE std::optional<Float> TryConvertFromAnyView(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIFloat) {
      return std::make_optional<Float>(src->v_float64);
    } else if (src->type_index == TypeIndex::kTVMFFIInt) {
      return std::make_optional<Float>(src->v_int64);
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return "float"; }
};

// Traits for object
template <typename TObjRef>
struct TypeTraits<TObjRef, std::enable_if_t<std::is_base_of_v<ObjectRef, TObjRef>>> {
  using ContainerType = typename TObjRef::ContainerType;

  static constexpr bool enabled = true;

  static TVM_FFI_INLINE void ConvertToAnyView(const TObjRef& src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectInternal::GetTVMFFIObjectPtrFromObjectRef(src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE void MoveToManagedAny(TObjRef src, TVMFFIAny* result) {
    TVMFFIObject* obj_ptr = details::ObjectInternal::MoveTVMFFIObjectPtrFromObjectRef(&src);
    result->type_index = obj_ptr->type_index;
    result->v_obj = obj_ptr;
  }

  static TVM_FFI_INLINE std::optional<TObjRef> TryConvertFromAnyView(const TVMFFIAny* src) {
    if (src->type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
#if TVM_FFI_ALLOW_DYN_TYPE
      if (details::IsObjectInstance<ContainerType>(src->type_index)) {
        return TObjRef(details::ObjectInternal::ObjectPtrFromUnowned<Object>(src->v_obj));
      }
#else
      TVM_FFI_THROW(RuntimeError)
          << "Converting to object requires `TVM_FFI_ALLOW_DYN_TYPE` to be on".
#endif
    } else if (src->type_index == kTVMFFINone) {
      if (!TObjRef::_type_is_nullable) return std::nullopt;
      return TObjRef(ObjectPtr<Object>());
    }
    return std::nullopt;
  }

  static TVM_FFI_INLINE std::string TypeStr() { return ContainerType::_type_key; }
};

/*!
 * \brief Get type key from type index
 * \param type_index The input type index
 * \return the type key
 */
inline std::string TypeIndex2TypeKey(int32_t type_index) {
  switch (type_index) {
    case TypeIndex::kTVMFFINone:
      return "None";
    case TypeIndex::kTVMFFIInt:
      return "int";
    case TypeIndex::kTVMFFIFloat:
      return "double";
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
#if TVM_FFI_ALLOW_DYN_TYPE
      const TypeInfo* type_info = details::ObjectGetTypeInfo(type_index);
      return type_info->type_key;
#else
      return "object.Object";
#endif
    }
  }
}
}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_TYPE_TRAITS_H_
