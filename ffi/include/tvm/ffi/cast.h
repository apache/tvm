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
 * \file tvm/ffi/cast.h
 * \brief Extra value casting helpers
 */
#ifndef TVM_FFI_CAST_H_
#define TVM_FFI_CAST_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

namespace tvm {
namespace ffi {

/*!
 * \brief Get a reference type from a raw object ptr type
 *
 *  It is always important to get a reference type
 *  if we want to return a value as reference or keep
 *  the object alive beyond the scope of the function.
 *
 * \param ptr The object pointer
 * \tparam RefType The reference type
 * \tparam ObjectType The object type
 * \return The corresponding RefType
 */
template <typename RefType, typename ObjectType>
inline RefType GetRef(const ObjectType* ptr) {
  static_assert(std::is_base_of_v<typename RefType::ContainerType, ObjectType>,
                "Can only cast to the ref of same container type");

  if constexpr (is_optional_type_v<RefType> || RefType::_type_is_nullable) {
    if (ptr == nullptr) {
      return RefType(ObjectPtr<Object>(nullptr));
    }
  } else {
    TVM_FFI_ICHECK_NOTNULL(ptr);
  }
  return RefType(details::ObjectUnsafe::ObjectPtrFromUnowned<Object>(
      const_cast<Object*>(static_cast<const Object*>(ptr))));
}

/*!
 * \brief Get an object ptr type from a raw object ptr.
 *
 * \param ptr The object pointer
 * \tparam BaseType The reference type
 * \tparam ObjectType The object type
 * \return The corresponding RefType
 */
template <typename BaseType, typename ObjectType>
inline ObjectPtr<BaseType> GetObjectPtr(ObjectType* ptr) {
  static_assert(std::is_base_of<BaseType, ObjectType>::value,
                "Can only cast to the ref of same container type");
  return details::ObjectUnsafe::ObjectPtrFromUnowned<BaseType>(ptr);
}
}  // namespace ffi

using ffi::GetObjectPtr;
using ffi::GetRef;
}  // namespace tvm
#endif  // TVM_FFI_CAST_H_
