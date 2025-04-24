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
 * \file tvm/ffi/container/tuple.h
 * \brief Typed tuple like std::tuple backed by ArrayObj container.
 */
#ifndef TVM_FFI_CONTAINER_TUPLE_H_
#define TVM_FFI_CONTAINER_TUPLE_H_

#include <tvm/ffi/container/array.h>

#include <string>
#include <tuple>
#include <utility>

namespace tvm {
namespace ffi {

/*!
 * \brief Typed tuple like std::tuple backed by ArrayObj container.
 *
 * Tuple implements in-place copy-on-write semantics.
 *
 * \tparam Types The types of the tuple elements
 */
template <typename... Types>
class Tuple : public ObjectRef {
 public:
  static constexpr bool all_storage_enabled_v = (details::storage_enabled_v<Types> && ...);
  static_assert(all_storage_enabled_v, "All types used in Tuple<...> must be compatible with Any");

  Tuple() : ObjectRef(MakeDefaultTupleNode()) {}
  Tuple(const Tuple<Types...>& other) : ObjectRef(other) {}
  Tuple(Tuple<Types...>&& other) : ObjectRef(std::move(other)) {}
  template <typename... UTypes,
            typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...), int>>
  Tuple(const Tuple<UTypes...>& other) : ObjectRef(other) {}
  template <typename... UTypes,
            typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...), int>>
  Tuple(Tuple<UTypes...>&& other) : ObjectRef(std::move(other)) {}
  template <typename... UTypes>
  explicit Tuple(UTypes&&... args) : ObjectRef(MakeTupleNode(std::forward<UTypes>(args)...)) {
    static_assert(sizeof...(Types) == sizeof...(UTypes), "Tuple size mismatch");
  }

  TVM_FFI_INLINE Tuple& operator=(const Tuple<Types...>& other) {
    data_ = other.data_;
    return *this;
  }

  TVM_FFI_INLINE Tuple& operator=(Tuple<Types...>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  template <typename... UTypes,
            typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...)>>
  TVM_FFI_INLINE Tuple& operator=(const Tuple<UTypes...>& other) {
    data_ = other.data_;
    return *this;
  }

  template <typename... UTypes,
            typename = std::enable_if_t<(details::type_contains_v<Types, UTypes> && ...)>>
  TVM_FFI_INLINE Tuple& operator=(Tuple<UTypes...>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  explicit Tuple(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*!
   * \brief Get I-th element of the tuple
   *
   * \tparam I The index of the element to get
   * \return The I-th element of the tuple
   */
  template <size_t I>
  auto Get() const {
    static_assert(I < sizeof...(Types), "Tuple index out of bounds");
    using ReturnType = std::tuple_element_t<I, std::tuple<Types...>>;
    const Any* ptr = GetArrayObj()->begin() + I;
    return details::AnyUnsafe::CopyFromAnyStorageAfterCheck<ReturnType>(*ptr);
  }

  /*!
   * \brief Set I-th element of the tuple
   *
   * \param item The item to set
   * \tparam I The index of the element to set
   * \tparam U The type of the item
   *
   * \note This function will perform copy on write if underlying
   *       container is not uniquely owned.
   */
  template <size_t I, typename U>
  void Set(U&& item) {
    static_assert(I < sizeof...(Types), "Tuple index out of bounds");
    using T = std::tuple_element_t<I, std::tuple<Types...>>;
    this->CopyIfNotUnique();
    Any* ptr = GetArrayObj()->MutableBegin() + I;
    *ptr = T(std::forward<U>(item));
  }

  /*! \brief specify container node */
  using ContainerType = ArrayObj;

 private:
  static ObjectPtr<ArrayObj> MakeDefaultTupleNode() {
    ObjectPtr<ArrayObj> p = make_inplace_array_object<ArrayObj, Any>(sizeof...(Types));
    p->capacity_ = sizeof...(Types);
    // immeidate set size to 0, to ensure exception safety
    p->size_ = 0;
    Any* itr = p->MutableBegin();
    // increase size after each new to ensure exception safety
    ((new (itr++) Any(Types()), p->size_++), ...);
    return p;
  }

  template <typename... UTypes>
  static ObjectPtr<ArrayObj> MakeTupleNode(UTypes&&... args) {
    ObjectPtr<ArrayObj> p = make_inplace_array_object<ArrayObj, Any>(sizeof...(Types));
    p->capacity_ = sizeof...(Types);
    // immeidate set size to 0, to ensure exception safety
    p->size_ = 0;
    Any* itr = p->MutableBegin();
    // increase size after each new to ensure exception safety
    ((new (itr++) Any(Types(std::forward<UTypes>(args))), p->size_++), ...);
    return p;
  }

  /*! \brief Copy on write */
  void CopyIfNotUnique() {
    if (!data_.unique()) {
      ObjectPtr<ArrayObj> p = make_inplace_array_object<ArrayObj, Any>(sizeof...(Types));
      p->capacity_ = sizeof...(Types);
      // immeidate set size to 0, to ensure exception safety
      p->size_ = 0;
      Any* itr = p->MutableBegin();
      const Any* read = GetArrayObj()->begin();
      // increase size after each new to ensure exception safety
      for (size_t i = 0; i < sizeof...(Types); ++i) {
        new (itr++) Any(*read++);
        p->size_++;
      }
      data_ = std::move(p);
    }
  }

  /*! \return The underlying ArrayObj */
  ArrayObj* GetArrayObj() const { return static_cast<ArrayObj*>(data_.get()); }

  template <typename... UTypes>
  friend class Tuple;
};

template <typename... Types>
inline constexpr bool use_default_type_traits_v<Tuple<Types...>> = false;

template <typename... Types>
struct TypeTraits<Tuple<Types...>> : public ObjectRefTypeTraitsBase<Tuple<Types...>> {
  using ObjectRefTypeTraitsBase<Tuple<Types...>>::CopyFromAnyStorageAfterCheck;

  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) {
      return TypeTraitsBase::GetMismatchTypeInfo(src);
    }
    const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
    if (n->size() != sizeof...(Types)) {
      return "Array[size=" + std::to_string(n->size()) + "]";
    }
    return GetMismatchTypeInfoHelper<0, Types...>(n->begin());
  }

  template <size_t I, typename T, typename... Rest>
  static TVM_FFI_INLINE std::string GetMismatchTypeInfoHelper(const Any* arr) {
    if constexpr (!std::is_same_v<T, Any>) {
      const Any& any_v = arr[I];
      if (!details::AnyUnsafe::CheckAnyStorage<T>(any_v) && !(any_v.as<T>().has_value())) {
        // now report the accurate mismatch information
        return "Array[index " + std::to_string(I) + ": " +
               details::AnyUnsafe::GetMismatchTypeInfo<T>(any_v) + "]";
      }
    }
    if constexpr (sizeof...(Rest) > 0) {
      return GetMismatchTypeInfoHelper<I + 1, Rest...>(arr);
    }
    TVM_FFI_THROW(InternalError) << "Cannot reach here";
    TVM_FFI_UNREACHABLE();
  }

  static TVM_FFI_INLINE bool CheckAnyStorage(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
    if (n->size() != sizeof...(Types)) return false;
    const TVMFFIAny* ffi_any_arr = reinterpret_cast<const TVMFFIAny*>(n->begin());
    return CheckAnyStorageHelper<0, Types...>(ffi_any_arr);
  }

  template <size_t I, typename T, typename... Rest>
  static TVM_FFI_INLINE bool CheckAnyStorageHelper(const TVMFFIAny* src_arr) {
    if constexpr (!std::is_same_v<T, Any>) {
      if (!TypeTraits<T>::CheckAnyStorage(src_arr + I)) {
        return false;
      }
    }
    if constexpr (sizeof...(Rest) > 0) {
      return CheckAnyStorageHelper<I + 1, Rest...>(src_arr);
    }
    return true;
  }

  static TVM_FFI_INLINE std::optional<Tuple<Types...>> TryConvertFromAnyView(
      const TVMFFIAny* src  //
  ) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return std::nullopt;
    const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
    if (n->size() != sizeof...(Types)) return std::nullopt;
    // fast path, storage is already in the right type
    if (CheckAnyStorage(src)) {
      return CopyFromAnyStorageAfterCheck(src);
    }
    // slow path, try to convert to each type to match the tuple storage need.
    Array<Any> arr = TypeTraits<Array<Any>>::CopyFromAnyStorageAfterCheck(src);
    Any* ptr = arr.CopyOnWrite()->MutableBegin();
    if (TryConvertElements<0, Types...>(ptr)) {
      return Tuple<Types...>(details::ObjectUnsafe::ObjectPtrFromObjectRef<Object>(arr));
    }
    return std::nullopt;
  }

  template <size_t I, typename T, typename... Rest>
  static TVM_FFI_INLINE bool TryConvertElements(Any* arr) {
    if constexpr (!std::is_same_v<T, Any>) {
      if (auto opt_convert = arr[I].as<T>()) {
        arr[I] = *std::move(opt_convert);
      } else {
        return false;
      }
    }
    if constexpr (sizeof...(Rest) > 0) {
      return TryConvertElements<I + 1, Rest...>(std::move(arr));
    }
    return true;
  }

  static TVM_FFI_INLINE std::string TypeStr() {
    return details::ContainerTypeStr<Types...>("Tuple");
  }
};

namespace details {
template <typename... T, typename... U>
inline constexpr bool type_contains_v<Tuple<T...>, Tuple<U...>> = (type_contains_v<T, U> && ...);
}  // namespace details

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_TUPLE_H_
