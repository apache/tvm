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
 * \file tvm/node/cast.h
 * \brief Value casting helpers
 */
#ifndef TVM_NODE_CAST_H_
#define TVM_NODE_CAST_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/cast.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <utility>

namespace tvm {

/*!
 * \brief Downcast a base reference type to a more specific type.
 *
 * \param ref The input reference
 * \return The corresponding SubRef.
 * \tparam SubRef The target specific reference type.
 * \tparam BaseRef the current reference type.
 */
template <typename SubRef, typename BaseRef,
          typename = std::enable_if_t<std::is_base_of_v<ffi::ObjectRef, BaseRef>>>
inline SubRef Downcast(BaseRef ref) {
  using ContainerType = typename SubRef::ContainerType;
  if (ref.defined()) {
    if (!ref->template IsInstance<ContainerType>()) {
      TVM_FFI_THROW(TypeError) << "Downcast from " << ref->GetTypeKey() << " to "
                               << SubRef::ContainerType::_type_key << " failed.";
    }
    return ffi::details::ObjectUnsafe::ObjectRefFromObjectPtr<SubRef>(
        ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<ffi::Object>(std::move(ref)));
  } else {
    if constexpr (ffi::is_optional_type_v<SubRef> || SubRef::_type_is_nullable) {
      return ffi::details::ObjectUnsafe::ObjectRefFromObjectPtr<SubRef>(nullptr);
    }
    TVM_FFI_THROW(TypeError) << "Downcast from undefined(nullptr) to `" << ContainerType::_type_key
                             << "` is not allowed. Use Downcast<ffi::Optional<T>> instead.";
    TVM_FFI_UNREACHABLE();
  }
}

/*!
 * \brief Downcast any to a specific type
 *
 * \param ref The input reference
 * \return The corresponding SubRef.
 * \tparam T The target specific reference type.
 */
template <typename T>
inline T Downcast(const ffi::Any& ref) {
  if constexpr (std::is_same_v<T, Any>) {
    return ref;
  } else {
    return ref.cast<T>();
  }
}

/*!
 * \brief Downcast any to a specific type
 *
 * \param ref The input reference
 * \return The corresponding SubRef.
 * \tparam T The target specific reference type.
 */
template <typename T>
inline T Downcast(ffi::Any&& ref) {
  if constexpr (std::is_same_v<T, Any>) {
    return std::move(ref);
  } else {
    return std::move(ref).cast<T>();
  }
}

/*!
 * \brief Downcast std::optional<Any> to std::optional<T>
 *
 * \param ref The input reference
 * \return The corresponding SubRef.
 * \tparam OptionalType The target optional type
 */
template <typename OptionalType, typename = std::enable_if_t<ffi::is_optional_type_v<OptionalType>>>
inline OptionalType Downcast(const std::optional<ffi::Any>& ref) {
  if (ref.has_value()) {
    if constexpr (std::is_same_v<OptionalType, ffi::Any>) {
      return *ref;
    } else {
      return (*ref).cast<OptionalType>();
    }
  } else {
    return OptionalType(std::nullopt);
  }
}
}  // namespace tvm
#endif  // TVM_NODE_CAST_H_
