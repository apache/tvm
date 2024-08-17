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
 * \file tvm/ffi/any.h
 * \brief Any value support.
 */
#ifndef TVM_FFI_ANY_H_
#define TVM_FFI_ANY_H_

#include <tvm/ffi/c_api.h>
#include <tvm/ffi/type_traits.h>

namespace tvm {
namespace ffi {

class Any;

/*!
 * \brief AnyView allows us to take un-managed reference view of any value.
 */
class AnyView {
 protected:
  /*! \brief The underlying backing data of the any object */
  TVMFFIAny data_;
  // Any can see AnyView
  friend class Any;

 public:
  // NOTE: the following two functions uses styl style
  // since they are common functions appearing in FFI.
  /*!
   * \brief Reset any view to None
   */
  void reset() { data_.type_index = TypeIndex::kTVMFFINone; }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(AnyView& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }
  // default constructors
  AnyView() { data_.type_index = TypeIndex::kTVMFFINone; }
  ~AnyView() = default;
  // constructors from any view
  AnyView(const AnyView&) = default;
  AnyView& operator=(const AnyView&) = default;
  AnyView(AnyView&& other) : data_(other.data_) { other.data_.type_index = TypeIndex::kTVMFFINone; }
  AnyView& operator=(AnyView&& other) {
    // copy-and-swap idiom
    AnyView(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  // constructor from general types
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  AnyView(const T& other) {  // NOLINT(*)
    TypeTraits<T>::ConvertToAnyView(other, &data_);
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  AnyView& operator=(const T& other) {  // NOLINT(*)
    // copy-and-swap idiom
    AnyView(other).swap(*this);  // NOLINT(*)
    return *this;
  }

  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  std::optional<T> TryAs() const {
    return TypeTraits<T>::TryConvertFromAnyView(&data_);
  }

  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  operator T() const {
    std::optional<T> opt = TypeTraits<T>::TryConvertFromAnyView(&data_);
    if (opt.has_value()) {
      return std::move(opt.value());
    }
    TVM_FFI_THROW(TypeError) << "Cannot convert from type `" << TypeIndex2TypeKey(data_.type_index)
                             << "` to `" << TypeTraits<T>::TypeStr() << "`";
  }
  // The following functions are only used for testing purposes
  /*!
   * \return The underlying supporting data of any view
   * \note This function is used only for testing purposes.
   */
  TVMFFIAny AsTVMFFIAny() const { return data_; }
  /*!
   * \return Create an AnyView from TVMFFIAny
   * \param data the underlying ffi data.
   */
  static AnyView FromTVMFFIAny(TVMFFIAny data) {
    AnyView view;
    view.data_ = data;
    return view;
  }
};

// layout assert to ensure we can freely cast between the two types
static_assert(sizeof(AnyView) == sizeof(TVMFFIAny));

namespace details {
/*!
 * \brief Helper function to inplace convert any view to any.
 * \param data The pointer that represents the format as any view.
 * \param extra_any_bytes Indicate that the data may contain extra bytes following
 *  the TVMFFIAny data structure. This is reserved for future possible optimizations
 *  of small-string and extended any object.
 */
TVM_FFI_INLINE void InplaceConvertAnyViewToAny(TVMFFIAny* data,
                                               [[maybe_unused]] size_t extra_any_bytes = 0) {
  // TODO: string conversion.
  if (data->type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin) {
    details::ObjectInternal::IncRefObjectInAny(data);
  }
}
}  // namespace details

/*!
 * \brief
 */
class Any {
 protected:
  /*! \brief The underlying backing data of the any object */
  TVMFFIAny data_;

 public:
  /*!
   * \brief Reset any to None
   */
  void reset() {
    if (data_.type_index >= TVMFFITypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectInternal::DecRefObjectInAny(&data_);
    }
    data_.type_index = TVMFFITypeIndex::kTVMFFINone;
  }
  /*!
   * \brief Swap this array with another Object
   * \param other The other Object
   */
  void swap(Any& other) {  // NOLINT(*)
    std::swap(data_, other.data_);
  }

  // default constructors
  Any() { data_.type_index = TypeIndex::kTVMFFINone; }
  ~Any() { this->reset(); }
  // constructors from Any
  Any(const Any& other) : data_(other.data_) {
    if (data_.type_index >= TypeIndex::kTVMFFIStaticObjectBegin) {
      details::ObjectInternal::IncRefObjectInAny(&data_);
    }
  }
  Any(Any&& other) : data_(other.data_) { other.data_.type_index = TypeIndex::kTVMFFINone; }
  Any& operator=(const Any& other) {
    // copy-and-swap idiom
    Any(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  Any& operator=(Any&& other) {
    // copy-and-swap idiom
    Any(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  // convert from/to AnyView
  Any(const AnyView& other) : data_(other.data_) { details::InplaceConvertAnyViewToAny(&data_); }
  Any& operator=(const AnyView& other) {
    // copy-and-swap idiom
    Any(other).swap(*this);  // NOLINT(*)
    return *this;
  }
  /*! \brief Any can be converted to AnyView in zero cost. */
  operator AnyView() { return AnyView::FromTVMFFIAny(data_); }
  // constructor from general types
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  Any(T other) {  // NOLINT(*)
    TypeTraits<T>::MoveToManagedAny(std::move(other), &data_);
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  Any& operator=(T other) {  // NOLINT(*)
    // copy-and-swap idiom
    Any(std::move(other)).swap(*this);  // NOLINT(*)
    return *this;
  }
  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  std::optional<T> TryAs() const {
    return TypeTraits<T>::TryConvertFromAnyView(&data_);
  }

  template <typename T, typename = std::enable_if_t<TypeTraits<T>::enabled>>
  operator T() const {
    std::optional<T> opt = TypeTraits<T>::TryConvertFromAnyView(&data_);
    if (opt.has_value()) {
      return std::move(opt.value());
    }
    TVM_FFI_THROW(TypeError) << "Cannot convert from type `" << TypeIndex2TypeKey(data_.type_index)
                             << "` to `" << TypeTraits<T>::TypeStr() << "`";
  }
};

}  // namespace ffi
}  // namespace tvm
#endif  // TVM_FFI_ANY_H_
