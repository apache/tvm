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
 * \file tvm/script/printer/traced_object.h
 * Wrappers around TVM objects that also store an ObjectPath from some "root" object
 * to the wrapper object.
 */

#ifndef TVM_SCRIPT_PRINTER_TRACED_OBJECT_H_
#define TVM_SCRIPT_PRINTER_TRACED_OBJECT_H_

#include <tvm/node/object_path.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/object.h>

#include <string>
#include <utility>

namespace tvm {

template <typename RefT>
class TracedObject;
template <typename K, typename V>
class TracedMap;
template <typename T>
class TracedArray;
template <typename T>
class TracedOptional;
template <typename T>
class TracedBasicValue;

namespace detail {

template <typename T, bool IsObject = std::is_base_of<ObjectRef, T>::value>
struct TracedObjectWrapperSelector;

template <typename T>
struct TracedObjectWrapperSelector<T, false> {
  using Type = TracedBasicValue<T>;
};

template <typename T>
struct TracedObjectWrapperSelector<T, true> {
  using Type = TracedObject<T>;
};

template <typename K, typename V>
struct TracedObjectWrapperSelector<Map<K, V>, true> {
  using Type = TracedMap<K, V>;
};

template <typename T>
struct TracedObjectWrapperSelector<Array<T>, true> {
  using Type = TracedArray<T>;
};

template <typename T>
struct TracedObjectWrapperSelector<Optional<T>, true> {
  using Type = TracedOptional<T>;
};

}  // namespace detail

/*!
 * \brief Traced wrapper for regular (non-container) TVM objects.
 */
template <typename RefT>
class TracedObject {
  using ObjectType = typename RefT::ContainerType;

 public:
  // Don't use this direcly. For convenience, call MakeTraced() instead.
  explicit TracedObject(const RefT& object_ref, ObjectPath path)
      : ref_(object_ref), path_(std::move(path)) {}

  // Implicit conversion from a derived reference class
  template <typename DerivedRef>
  TracedObject(const TracedObject<DerivedRef>& derived)
      : ref_(derived.Get()), path_(derived.GetPath()) {}

  /*!
   * \brief Get a traced wrapper for an attribute of the wrapped object.
   */
  template <typename T, typename BaseType>
  typename detail::TracedObjectWrapperSelector<T>::Type GetAttr(T BaseType::*member_ptr) const {
    using WrapperType = typename detail::TracedObjectWrapperSelector<T>::Type;
    const ObjectType* node = static_cast<const ObjectType*>(ref_.get());
    const T& attr = node->*member_ptr;
    Optional<String> attr_key = ICHECK_NOTNULL(GetAttrKeyByAddress(node, &attr));
    return WrapperType(attr, path_->Attr(attr_key));
  }

  /*!
   * \brief Access the wrapped object.
   */
  const RefT& Get() const { return ref_; }

  /*!
   * \brief Check if the reference to the wrapped object can be converted to `RefU`.
   */
  template <typename RefU>
  bool IsInstance() const {
    return ref_->template IsInstance<typename RefU::ContainerType>();
  }

  /*!
   * \brief Same as Get().defined().
   */
  bool defined() const { return ref_.defined(); }

  /*!
   * \brief Convert the wrapped reference type to a subtype.
   *
   * Throws an exception if IsInstance<RefU>() is false.
   */
  template <typename RefU>
  TracedObject<RefU> Downcast() const {
    return TracedObject<RefU>(tvm::runtime::Downcast<RefU>(ref_), path_);
  }

  /*!
   * \brief Convert the wrapped reference type to a subtype.
   *
   * Returns an empty optional if IsInstance<RefU>() is false.
   */
  template <typename RefU>
  TracedOptional<RefU> TryDowncast() const {
    if (ref_->template IsInstance<typename RefU::ContainerType>()) {
      return Downcast<RefU>();
    } else {
      return TracedOptional<RefU>(NullOpt, path_);
    }
  }

  /*!
   * \brief Get the path of the wrapped object.
   */
  const ObjectPath& GetPath() const { return path_; }

 private:
  RefT ref_;
  ObjectPath path_;
};

/*!
 * \brief Iterator class for TracedMap<K, V>
 */
template <typename K, typename V>
class TracedMapIterator {
 public:
  using WrappedV = typename detail::TracedObjectWrapperSelector<V>::Type;
  using MapIter = typename Map<K, V>::iterator;

  using iterator_category = std::bidirectional_iterator_tag;
  using difference_type = ptrdiff_t;
  using value_type = const std::pair<K, WrappedV>;
  using pointer = value_type*;
  using reference = value_type;

  explicit TracedMapIterator(MapIter iter, ObjectPath map_path)
      : iter_(iter), map_path_(std::move(map_path)) {}

  bool operator==(const TracedMapIterator& other) const { return iter_ == other.iter_; }

  bool operator!=(const TracedMapIterator& other) const { return iter_ != other.iter_; }

  pointer operator->() const = delete;

  reference operator*() const {
    auto kv = *iter_;
    return std::make_pair(kv.first, WrappedV(kv.second, map_path_->MapValue(kv.first)));
  }

  TracedMapIterator& operator++() {
    ++iter_;
    return *this;
  }

  TracedMapIterator operator++(int) {
    TracedMapIterator copy = *this;
    ++(*this);
    return copy;
  }

 private:
  MapIter iter_;
  ObjectPath map_path_;
};

/*!
 * \brief Traced wrapper for Map objects.
 */
template <typename K, typename V>
class TracedMap {
 public:
  using WrappedV = typename detail::TracedObjectWrapperSelector<V>::Type;

  using iterator = TracedMapIterator<K, V>;

  // Don't use this direcly. For convenience, call MakeTraced() instead.
  explicit TracedMap(Map<K, V> map, ObjectPath path)
      : map_(std::move(map)), path_(std::move(path)) {}

  /*!
   * \brief Get a value by its key, wrapped in a traced wrapper.
   */
  WrappedV at(const K& key) const {
    auto it = map_.find(key);
    ICHECK(it != map_.end()) << "No such key in Map";
    auto kv = *it;
    return WrappedV(kv.second, path_->MapValue(kv.first));
  }

  /*!
   * \brief Access the wrapped map object.
   */
  const Map<K, V>& Get() const { return map_; }

  /*!
   * \brief Get the path of the wrapped object.
   */
  const ObjectPath& GetPath() const { return path_; }

  /*!
   * \brief Get an iterator to the first item of the map.
   */
  iterator begin() const { return iterator(map_.begin(), path_); }

  /*!
   * \brief Get an iterator to the end of the map.
   */
  iterator end() const { return iterator(map_.end(), path_); }

  /*!
   * \brief Returns true iff the wrapped map is empty.
   */
  bool empty() const { return map_.empty(); }

 private:
  Map<K, V> map_;
  ObjectPath path_;
};

/*!
 * \brief Iterator class for TracedArray<T>
 */
template <typename T>
class TracedArrayIterator {
 public:
  using WrappedT = typename detail::TracedObjectWrapperSelector<T>::Type;

  using difference_type = ptrdiff_t;
  using value_type = WrappedT;
  using pointer = WrappedT*;
  using reference = WrappedT&;
  using iterator_category = std::random_access_iterator_tag;

  explicit TracedArrayIterator(Array<T> array, size_t index, ObjectPath array_path)
      : array_(array), index_(index), array_path_(array_path) {}

  TracedArrayIterator& operator++() {
    ++index_;
    return *this;
  }
  TracedArrayIterator& operator--() {
    --index_;
    return *this;
  }
  TracedArrayIterator operator++(int) {
    TracedArrayIterator copy = *this;
    ++index_;
    return copy;
  }
  TracedArrayIterator operator--(int) {
    TracedArrayIterator copy = *this;
    --index_;
    return copy;
  }

  TracedArrayIterator operator+(difference_type offset) const {
    return TracedArrayIterator(array_, index_ + offset, array_path_);
  }

  TracedArrayIterator operator-(difference_type offset) const {
    return TracedArrayIterator(array_, index_ - offset, array_path_);
  }

  difference_type operator-(const TracedArrayIterator& rhs) const { return index_ - rhs.index_; }

  bool operator==(TracedArrayIterator other) const {
    return array_.get() == other.array_.get() && index_ == other.index_;
  }
  bool operator!=(TracedArrayIterator other) const { return !(*this == other); }
  value_type operator*() const { return WrappedT(array_[index_], array_path_->ArrayIndex(index_)); }

 private:
  Array<T> array_;
  size_t index_;
  ObjectPath array_path_;
};

/*!
 * \brief Traced wrapper for Array objects.
 */
template <typename T>
class TracedArray {
 public:
  using WrappedT = typename detail::TracedObjectWrapperSelector<T>::Type;

  using iterator = TracedArrayIterator<T>;

  // Don't use this direcly. For convenience, call MakeTraced() instead.
  explicit TracedArray(Array<T> array, ObjectPath path)
      : array_(std::move(array)), path_(std::move(path)) {}

  /*!
   * \brief Access the wrapped array object.
   */
  const Array<T>& Get() const { return array_; }

  /*!
   * \brief Get the path of the wrapped array object.
   */
  const ObjectPath& GetPath() const { return path_; }

  /*!
   * \brief Get an element by index, wrapped in a traced wrapper.
   */
  WrappedT operator[](size_t index) const {
    return WrappedT(array_[index], path_->ArrayIndex(index));
  }

  /*!
   * \brief Get an iterator to the first array element.
   *
   * The iterator's dereference operator will automatically wrap each element in a traced wrapper.
   */
  iterator begin() const { return iterator(array_, 0, path_); }

  /*!
   * \brief Get an iterator to the end of the array.
   *
   * The iterator's dereference operator will automatically wrap each element in a traced wrapper.
   */
  iterator end() const { return iterator(array_, array_.size(), path_); }

  /*!
   * \brief Returns true iff the wrapped array is empty.
   */
  bool empty() const { return array_.empty(); }

  /*!
   * \brief Get the size of the wrapped array.
   */
  size_t size() const { return array_.size(); }

 private:
  Array<T> array_;
  ObjectPath path_;
};

/*!
 * \brief Traced wrapper for Optional objects.
 */
template <typename T>
class TracedOptional {
 public:
  using WrappedT = typename detail::TracedObjectWrapperSelector<T>::Type;

  /*!
   * \brief Implicit conversion from the corresponding non-optional traced wrapper.
   */
  TracedOptional(const WrappedT& value)  // NOLINT(runtime/explicit)
      : optional_(value.Get().defined() ? value.Get() : Optional<T>(NullOpt)),
        path_(value.GetPath()) {}

  // Don't use this direcly. For convenience, call MakeTraced() instead.
  explicit TracedOptional(Optional<T> optional, ObjectPath path)
      : optional_(std::move(optional)), path_(std::move(path)) {}

  /*!
   * \brief Access the wrapped optional object.
   */
  const Optional<T>& Get() const { return optional_; }

  /*!
   * \brief Get the path of the wrapped optional object.
   */
  const ObjectPath& GetPath() const { return path_; }

  /*!
   * \brief Returns true iff the object is present.
   */
  bool defined() const { return optional_.defined(); }

  /*!
   * \brief Returns a non-optional traced wrapper, throws if defined() is false.
   */
  WrappedT value() const { return WrappedT(optional_.value(), path_); }

  /*!
   * \brief Same as defined().
   */
  explicit operator bool() const { return optional_.defined(); }

 private:
  Optional<T> optional_;
  ObjectPath path_;
};

/*!
 * \brief Traced wrapper for basic values (i.e. non-TVM objects)
 */
template <typename T>
class TracedBasicValue {
 public:
  explicit TracedBasicValue(const T& value, ObjectPath path)
      : value_(value), path_(std::move(path)) {}

  /*!
   * \brief Access the wrapped value.
   */
  const T& Get() const { return value_; }

  /*!
   * \brief Get the path of the wrapped value.
   */
  const ObjectPath& GetPath() const { return path_; }

  /*!
   * \brief Transform the wrapped value without changing its path.
   */
  template <typename F>
  typename detail::TracedObjectWrapperSelector<typename std::result_of<F(const T&)>::type>::Type
  ApplyFunc(F&& f) const {
    return MakeTraced(f(value_), path_);
  }

 private:
  T value_;
  ObjectPath path_;
};

/*!
 * \brief Wrap the given root object in an appropriate traced wrapper class.
 */
template <typename RefT>
typename detail::TracedObjectWrapperSelector<RefT>::Type MakeTraced(const RefT& object) {
  using WrappedT = typename detail::TracedObjectWrapperSelector<RefT>::Type;
  return WrappedT(object, ObjectPath::Root());
}

/*!
 * \brief Wrap the given object with the given path in an appropriate traced wrapper class.
 */
template <typename RefT>
typename detail::TracedObjectWrapperSelector<RefT>::Type MakeTraced(const RefT& object,
                                                                    ObjectPath path) {
  using WrappedT = typename detail::TracedObjectWrapperSelector<RefT>::Type;
  return WrappedT(object, std::move(path));
}

}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_TRACED_OBJECT_H_
