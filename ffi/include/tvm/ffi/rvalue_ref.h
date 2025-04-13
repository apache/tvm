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
 * \file tvm/ffi/rvalue_ref.h
 * \brief Helper class to define rvalue reference type.
 */
#ifndef TVM_FFI_RVALUE_REF_H_
#define TVM_FFI_RVALUE_REF_H_

#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>

#include <string>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Helper class to define rvalue reference type.
 *
 * By default,  FFI pass all values by lvalue reference.
 *
 * However, we do allow users to intentionally mark a function parameter
 * as RValueRef<T>. In such cases, the caller can choose to pass parameter
 * wrapped by RValueRef<T> to the function. In which case the parameter
 * can be directly moved by the callee. The caller can also choose to pass
 * a normal lvalue to the function, in such case a copy will be triggered.
 *
 * To keep FFI checking overhead minimal, we do not handle case when rvalue
 * is passed, but the callee did not declare the parameter as RValueRef<T>.
 *
 * This design allows us to still leverage move semantics for parameters that
 * need copy on write scenarios (and requires an unique copy).
 *
 * \code
 *
 * void Example() {
 *   auto append = Function::FromUnpacked([](RValueRef<Array<int>> ref, int val) -> Array<int> {
 *     Array<int> arr = *std::move(ref);
 *     assert(arr.unique());
 *     arr.push_back(val);
 *     return arr;
 *   });
 *   Array<int> a = Array<int>({1, 2});
 *   // as we use rvalue ref to move a into append
 *   // we keep a single copy of the Array without creating new copies during copy-on-write
 *   a = append(RvalueRef(std::move(a)), 3);
 *   assert(a.size() == 3);
 * }
 *
 * \endcode
 */
template <typename TObjRef, typename = std::enable_if_t<std::is_base_of_v<ObjectRef, TObjRef>>>
class RValueRef {
 public:
  /*! \brief only allow move constructor from rvalue of T */
  explicit RValueRef(TObjRef&& data)
      : data_(details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(std::move(data))) {}

  /*! \brief return the data as rvalue */
  TObjRef operator*() && { return TObjRef(std::move(data_)); }

 private:
  mutable ObjectPtr<Object> data_;

  template <typename, typename>
  friend struct TypeTraits;
};

template <typename TObjRef>
inline constexpr bool use_default_type_traits_v<RValueRef<TObjRef>> = false;

template <typename TObjRef>
struct TypeTraits<RValueRef<TObjRef>> : public TypeTraitsBase {
  static constexpr bool storage_enabled = false;

  static TVM_FFI_INLINE void CopyToAnyView(const RValueRef<TObjRef>& src, TVMFFIAny* result) {
    result->type_index = TypeIndex::kTVMFFIObjectRValueRef;
    // store the address of the ObjectPtr, which allows us to move the value
    // and set the original ObjectPtr to nullptr
    result->v_ptr = &(src.data_);
  }

  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index == TypeIndex::kTVMFFIObjectRValueRef) {
      ObjectPtr<Object>* rvalue_ref = reinterpret_cast<ObjectPtr<Object>*>(src->v_ptr);
      // object type does not match up, we need to try to convert the object
      // in this case we do not move the original rvalue ref since conversion creates a copy
      TVMFFIAny tmp_any;
      tmp_any.type_index = rvalue_ref->get()->type_index();

      tmp_any.v_obj = reinterpret_cast<TVMFFIObject*>(rvalue_ref->get());
      return "RValueRef<" + TypeTraits<TObjRef>::GetMismatchTypeInfo(&tmp_any) + ">";
    } else {
      return TypeTraits<TObjRef>::GetMismatchTypeInfo(src);
    }
  }

  static TVM_FFI_INLINE std::optional<RValueRef<TObjRef>> TryConvertFromAnyView(
      const TVMFFIAny* src) {
    // first try rvalue conversion
    if (src->type_index == TypeIndex::kTVMFFIObjectRValueRef) {
      ObjectPtr<Object>* rvalue_ref = reinterpret_cast<ObjectPtr<Object>*>(src->v_ptr);
      TVMFFIAny tmp_any;
      tmp_any.type_index = rvalue_ref->get()->type_index();
      tmp_any.v_obj = reinterpret_cast<TVMFFIObject*>(rvalue_ref->get());
      // fast path, storage type matches, direct move the rvalue ref
      if (TypeTraits<TObjRef>::CheckAnyStorage(&tmp_any)) {
        return RValueRef<TObjRef>(TObjRef(std::move(*rvalue_ref)));
      }
      if (std::optional<TObjRef> opt = TypeTraits<TObjRef>::TryConvertFromAnyView(&tmp_any)) {
        // object type does not match up, we need to try to convert the object
        // in this case we do not move the original rvalue ref since conversion creates a copy
        return RValueRef<TObjRef>(*std::move(opt));
      }
      return std::nullopt;
    }
    // try lvalue conversion
    if (std::optional<TObjRef> opt = TypeTraits<TObjRef>::TryConvertFromAnyView(src)) {
      return RValueRef<TObjRef>(*std::move(opt));
    } else {
      return std::nullopt;
    }
  }

  static TVM_FFI_INLINE std::string TypeStr() {
    return "RValueRef<" + TypeTraits<TObjRef>::TypeStr() + ">";
  }
};
}  // namespace ffi
}  // namespace tvm

#endif  // TVM_FFI_RVALUE_REF_H_
