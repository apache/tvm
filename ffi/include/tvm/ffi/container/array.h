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
 * \file tvm/ffi/container/array.h
 * \brief Array type.
 *
 * tvm::ffi::Array<Any> is an erased type that contains list of content
 */
#ifndef TVM_FFI_CONTAINER_ARRAY_H_
#define TVM_FFI_CONTAINER_ARRAY_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/container_details.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tvm {
namespace ffi {

/*! \brief array node content in array */
class ArrayObj : public Object, public details::InplaceArrayBase<ArrayObj, Any> {
 public:
  /*! \return The size of the array */
  size_t size() const { return this->size_; }

  /*!
   * \brief Read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  const Any at(int64_t i) const { return this->operator[](i); }

  /*! \return begin constant iterator */
  const Any* begin() const { return static_cast<Any*>(InplaceArrayBase::AddressOf(0)); }

  /*! \return end constant iterator */
  const Any* end() const { return begin() + size_; }

  /*! \brief Release reference to all the elements */
  void clear() { ShrinkBy(size_); }

  /*!
   * \brief Set i-th element of the array in-place
   * \param i The index
   * \param item The value to be set
   */
  void SetItem(int64_t i, Any item) { this->operator[](i) = std::move(item); }

  /*!
   * \brief Constructs a container and copy from another
   * \param cap The capacity of the container
   * \param from Source of the copy
   * \return Ref-counted ArrayObj requested
   */
  static ObjectPtr<ArrayObj> CopyFrom(int64_t cap, ArrayObj* from) {
    int64_t size = from->size_;
    if (size > cap) {
      TVM_FFI_THROW(ValueError) << "not enough capacity";
    }
    ObjectPtr<ArrayObj> p = ArrayObj::Empty(cap);
    Any* write = p->MutableBegin();
    Any* read = from->MutableBegin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (int64_t& i = p->size_ = 0; i < size; ++i) {
      new (write++) Any(*read++);
    }
    return p;
  }

  /*!
   * \brief Constructs a container and move from another
   * \param cap The capacity of the container
   * \param from Source of the move
   * \return Ref-counted ArrayObj requested
   */
  static ObjectPtr<ArrayObj> MoveFrom(int64_t cap, ArrayObj* from) {
    int64_t size = from->size_;
    if (size > cap) {
      TVM_FFI_THROW(RuntimeError) << "not enough capacity";
    }
    ObjectPtr<ArrayObj> p = ArrayObj::Empty(cap);
    Any* write = p->MutableBegin();
    Any* read = from->MutableBegin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (int64_t& i = p->size_ = 0; i < size; ++i) {
      new (write++) Any(std::move(*read++));
    }
    from->size_ = 0;
    return p;
  }

  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   * \return Ref-counted ArrayObj requested
   */
  static ObjectPtr<ArrayObj> CreateRepeated(int64_t n, const Any& val) {
    ObjectPtr<ArrayObj> p = ArrayObj::Empty(n);
    Any* itr = p->MutableBegin();
    for (int64_t& i = p->size_ = 0; i < n; ++i) {
      new (itr++) Any(val);
    }
    return p;
  }

  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIArray;
  static constexpr const char* _type_key = "object.Array";
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_STATIC_OBJECT_INFO(ArrayObj, Object);

 private:
  /*! \return Size of initialized memory, used by InplaceArrayBase. */
  size_t GetSize() const { return this->size_; }

  /*! \return begin mutable iterator */
  Any* MutableBegin() const { return static_cast<Any*>(InplaceArrayBase::AddressOf(0)); }

  /*! \return end mutable iterator */
  Any* MutableEnd() const { return MutableBegin() + size_; }

  /*!
   * \brief Create an ArrayObj with the given capacity.
   * \param n Required capacity
   * \return Ref-counted ArrayObj requested
   */
  static ObjectPtr<ArrayObj> Empty(int64_t n = kInitSize) {
    TVM_FFI_ICHECK_GE(n, 0);
    ObjectPtr<ArrayObj> p = make_inplace_array_object<ArrayObj, Any>(n);
    p->capacity_ = n;
    p->size_ = 0;
    return p;
  }

  /*!
   * \brief Inplace-initialize the elements starting idx from [first, last)
   * \param idx The starting point
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return Self
   */
  template <typename IterType>
  ArrayObj* InitRange(int64_t idx, IterType first, IterType last) {
    Any* itr = MutableBegin() + idx;
    for (; first != last; ++first) {
      Any ref = *first;
      new (itr++) Any(std::move(ref));
    }
    return this;
  }

  /*!
   * \brief Move elements from right to left, requires src_begin > dst
   * \param dst Destination
   * \param src_begin The start point of copy (inclusive)
   * \param src_end The end point of copy (exclusive)
   * \return Self
   */
  ArrayObj* MoveElementsLeft(int64_t dst, int64_t src_begin, int64_t src_end) {
    Any* from = MutableBegin() + src_begin;
    Any* to = MutableBegin() + dst;
    while (src_begin++ != src_end) {
      *to++ = std::move(*from++);
    }
    return this;
  }

  /*!
   * \brief Move elements from left to right, requires src_begin < dst
   * \param dst Destination
   * \param src_begin The start point of move (inclusive)
   * \param src_end The end point of move (exclusive)
   * \return Self
   */
  ArrayObj* MoveElementsRight(int64_t dst, int64_t src_begin, int64_t src_end) {
    Any* from = MutableBegin() + src_end;
    Any* to = MutableBegin() + (src_end - src_begin + dst);
    while (src_begin++ != src_end) {
      *--to = std::move(*--from);
    }
    return this;
  }

  /*!
   * \brief Enlarges the size of the array
   * \param delta Size enlarged, should be positive
   * \param val Default value
   * \return Self
   */
  ArrayObj* EnlargeBy(int64_t delta, const Any& val = Any()) {
    Any* itr = MutableEnd();
    while (delta-- > 0) {
      new (itr++) Any(val);
      ++size_;
    }
    return this;
  }

  /*!
   * \brief Shrinks the size of the array
   * \param delta Size shrinked, should be positive
   * \return Self
   */
  ArrayObj* ShrinkBy(int64_t delta) {
    Any* itr = MutableEnd();
    while (delta-- > 0) {
      (--itr)->Any::~Any();
      --size_;
    }
    return this;
  }

  /*! \brief Number of elements used */
  int64_t size_;

  /*! \brief Number of elements allocated */
  int64_t capacity_;

  /*! \brief Initial size of ArrayObj */
  static constexpr int64_t kInitSize = 4;

  /*! \brief Expansion factor of the Array */
  static constexpr int64_t kIncFactor = 2;

  // CRTP parent class
  friend InplaceArrayBase<ArrayObj, Any>;

  // Reference class
  template <typename, typename>
  friend class Array;

  template <typename... Types>
  friend class Tuple;

  template <typename, typename>
  friend struct TypeTraits;

  // To specialize make_object<ArrayObj>
  friend ObjectPtr<ArrayObj> make_object<>();
};

/*! \brief Helper struct for type-checking
 *
 * is_valid_iterator<T,IterType>::value will be true if IterType can
 * be dereferenced into a type that can be stored in an Array<T>, and
 * false otherwise.
 */
template <typename T, typename IterType>
struct is_valid_iterator
    : std::bool_constant<
          std::is_same_v<
              T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<IterType>())>>> ||
          std::is_base_of_v<
              T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<IterType>())>>>> {
};

template <typename T, typename IterType>
struct is_valid_iterator<Optional<T>, IterType> : is_valid_iterator<T, IterType> {};

template <typename IterType>
struct is_valid_iterator<Any, IterType> : std::true_type {};

template <typename T, typename IterType>
inline constexpr bool is_valid_iterator_v = is_valid_iterator<T, IterType>::value;

/*!
 * \brief Array, container representing a contiguous sequence of ObjectRefs.
 *
 *  Array implements in-place copy-on-write semantics.
 *
 * As in typical copy-on-write, a method which would typically mutate the array
 * instead opaquely copies the underlying container, and then acts on its copy.
 *
 * If the array has reference count equal to one, we directly update the
 * container in place without copying. This is optimization is sound because
 * when the reference count is equal to one this reference is guranteed to be
 * the sole pointer to the container.
 *
 *
 * operator[] only provides const access, use Set to mutate the content.
 * \tparam T The content Value type, must be compatible with tvm::ffi::Any
 */
template <typename T, typename = typename std::enable_if_t<details::storage_enabled_v<T>>>
class Array : public ObjectRef {
 public:
  using value_type = T;
  // constructors
  /*!
   * \brief default constructor
   */
  Array() { data_ = ArrayObj::Empty(); }
  Array(Array<T>&& other) : ObjectRef(std::move(other.data_)) {}
  Array(const Array<T>& other) : ObjectRef(other.data_) {}
  template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
  Array(Array<U>&& other) : ObjectRef(std::move(other.data_)) {}
  template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
  Array(const Array<U>& other) : ObjectRef(other.data_) {}

  TVM_FFI_INLINE Array<T>& operator=(Array<T>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  TVM_FFI_INLINE Array<T>& operator=(const Array<T>& other) {
    data_ = other.data_;
    return *this;
  }
  template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
  TVM_FFI_INLINE Array<T>& operator=(Array<U>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  template <typename U, typename = std::enable_if_t<details::type_contains_v<T, U>>>
  TVM_FFI_INLINE Array<T>& operator=(const Array<U>& other) {
    data_ = other.data_;
    return *this;
  }

  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Array(ObjectPtr<Object> n) : ObjectRef(n) {}

  /*!
   * \brief Constructor from iterator
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Array(IterType first, IterType last) {
    static_assert(is_valid_iterator_v<T, IterType>,
                  "IterType cannot be inserted into a tvm::Array<T>");
    Assign(first, last);
  }

  /*!
   * \brief constructor from initializer list
   * \param init The initializer list
   */
  Array(std::initializer_list<T> init) {  // NOLINT(*)
    Assign(init.begin(), init.end());
  }

  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  Array(const std::vector<T>& init) {  // NOLINT(*)
    Assign(init.begin(), init.end());
  }

  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   */
  explicit Array(const size_t n, const T& val) { data_ = ArrayObj::CreateRepeated(n, val); }

 public:
  // iterators
  struct ValueConverter {
    using ResultType = T;
    static T convert(const Any& n) {
      return details::AnyUnsafe::CopyFromAnyStorageAfterCheck<T>(n);
    }
  };

  using iterator = details::IterAdapter<ValueConverter, const Any*>;
  using reverse_iterator = details::ReverseIterAdapter<ValueConverter, const Any*>;

  /*! \return begin iterator */
  iterator begin() const { return iterator(GetArrayObj()->begin()); }

  /*! \return end iterator */
  iterator end() const { return iterator(GetArrayObj()->end()); }

  /*! \return rbegin iterator */
  reverse_iterator rbegin() const {
    // ArrayObj::end() is never nullptr
    return reverse_iterator(GetArrayObj()->end() - 1);
  }

  /*! \return rend iterator */
  reverse_iterator rend() const {
    // ArrayObj::begin() is never nullptr
    return reverse_iterator(GetArrayObj()->begin() - 1);
  }

 public:
  // const methods in std::vector
  /*!
   * \brief Immutably read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  const T operator[](int64_t i) const {
    ArrayObj* p = GetArrayObj();
    if (p == nullptr) {
      TVM_FFI_THROW(IndexError) << "cannot index a null array";
    }
    if (i < 0 || i >= p->size_) {
      TVM_FFI_THROW(IndexError) << "indexing " << i << " on an array of size " << p->size_;
    }
    return details::AnyUnsafe::CopyFromAnyStorageAfterCheck<T>(*(p->begin() + i));
  }

  /*! \return The size of the array */
  size_t size() const {
    ArrayObj* p = GetArrayObj();
    return p == nullptr ? 0 : GetArrayObj()->size_;
  }

  /*! \return The capacity of the array */
  size_t capacity() const {
    ArrayObj* p = GetArrayObj();
    return p == nullptr ? 0 : GetArrayObj()->capacity_;
  }

  /*! \return Whether array is empty */
  bool empty() const { return size() == 0; }

  /*! \return The first element of the array */
  const T front() const {
    ArrayObj* p = GetArrayObj();
    if (p == nullptr || p->size_ == 0) {
      TVM_FFI_THROW(IndexError) << "cannot index a empty array";
    }
    return details::AnyUnsafe::CopyFromAnyStorageAfterCheck<T>(*(p->begin()));
  }

  /*! \return The last element of the array */
  const T back() const {
    ArrayObj* p = GetArrayObj();
    if (p == nullptr || p->size_ == 0) {
      TVM_FFI_THROW(IndexError) << "cannot index a empty array";
    }
    return details::AnyUnsafe::CopyFromAnyStorageAfterCheck<T>(*(p->end() - 1));
  }

 public:
  // mutation in std::vector, implements copy-on-write
  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  void push_back(const T& item) {
    ArrayObj* p = CopyOnWrite(1);
    p->EmplaceInit(p->size_++, item);
  }

  /*!
   * \brief Insert an element into the given position
   * \param position An iterator pointing to the insertion point
   * \param val The element to insert
   */
  void insert(iterator position, const T& val) {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "cannot insert a null array";
    }
    int64_t idx = std::distance(begin(), position);
    int64_t size = GetArrayObj()->size_;
    auto addr = CopyOnWrite(1)                               //
                    ->EnlargeBy(1)                           //
                    ->MoveElementsRight(idx + 1, idx, size)  //
                    ->MutableBegin();
    new (addr + idx) Any(val);
  }

  /*!
   * \brief Insert a range of elements into the given position
   * \param position An iterator pointing to the insertion point
   * \param first The begin iterator of the range
   * \param last The end iterator of the range
   */
  template <typename IterType>
  void insert(iterator position, IterType first, IterType last) {
    static_assert(is_valid_iterator_v<T, IterType>,
                  "IterType cannot be inserted into a tvm::Array<T>");

    if (first == last) {
      return;
    }
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "cannot insert a null array";
    }
    int64_t idx = std::distance(begin(), position);
    int64_t size = GetArrayObj()->size_;
    int64_t numel = std::distance(first, last);
    CopyOnWrite(numel)
        ->EnlargeBy(numel)
        ->MoveElementsRight(idx + numel, idx, size)
        ->InitRange(idx, first, last);
  }

  /*! \brief Remove the last item of the list */
  void pop_back() {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "cannot pop_back a null array";
    }
    int64_t size = GetArrayObj()->size_;
    if (size == 0) {
      TVM_FFI_THROW(RuntimeError) << "cannot pop_back an empty array";
    }
    CopyOnWrite()->ShrinkBy(1);
  }

  /*!
   * \brief Erase an element on the given position
   * \param position An iterator pointing to the element to be erased
   */
  void erase(iterator position) {
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "cannot erase a null array";
    }
    int64_t st = std::distance(begin(), position);
    int64_t size = GetArrayObj()->size_;
    if (st < 0 || st >= size) {
      TVM_FFI_THROW(RuntimeError) << "cannot erase at index " << st << ", because Array size is "
                                  << size;
    }
    CopyOnWrite()                             //
        ->MoveElementsLeft(st, st + 1, size)  //
        ->ShrinkBy(1);
  }

  /*!
   * \brief Erase a given range of elements
   * \param first The begin iterator of the range
   * \param last The end iterator of the range
   */
  void erase(iterator first, iterator last) {
    if (first == last) {
      return;
    }
    if (data_ == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "cannot erase a null array";
    }
    int64_t size = GetArrayObj()->size_;
    int64_t st = std::distance(begin(), first);
    int64_t ed = std::distance(begin(), last);
    if (st >= ed) {
      TVM_FFI_THROW(IndexError) << "cannot erase array in range [" << st << ", " << ed << ")";
    }
    if (st < 0 || st > size || ed < 0 || ed > size) {
      TVM_FFI_THROW(IndexError) << "cannot erase array in range [" << st << ", " << ed << ")"
                                << ", because array size is " << size;
    }
    CopyOnWrite()                         //
        ->MoveElementsLeft(st, ed, size)  //
        ->ShrinkBy(ed - st);
  }

  /*!
   * \brief Resize the array.
   * \param n The new size.
   */
  void resize(int64_t n) {
    if (n < 0) {
      TVM_FFI_THROW(ValueError) << "cannot resize an Array to negative size";
    }
    if (data_ == nullptr) {
      SwitchContainer(n);
      return;
    }
    int64_t size = GetArrayObj()->size_;
    if (size < n) {
      CopyOnWrite(n - size)->EnlargeBy(n - size);
    } else if (size > n) {
      CopyOnWrite()->ShrinkBy(size - n);
    }
  }

  /*!
   * \brief Make sure the list has the capacity of at least n
   * \param n lower bound of the capacity
   */
  void reserve(int64_t n) {
    if (data_ == nullptr || n > GetArrayObj()->capacity_) {
      SwitchContainer(n);
    }
  }

  /*! \brief Release reference to all the elements */
  void clear() {
    if (data_ != nullptr) {
      ArrayObj* p = CopyOnWrite();
      p->clear();
    }
  }

  template <typename... Args>
  static size_t CalcCapacityImpl() {
    return 0;
  }

  template <typename... Args>
  static size_t CalcCapacityImpl(Array<T> value, Args... args) {
    return value.size() + CalcCapacityImpl(args...);
  }

  template <typename... Args>
  static size_t CalcCapacityImpl(T value, Args... args) {
    return 1 + CalcCapacityImpl(args...);
  }

  template <typename... Args>
  static void AgregateImpl(Array<T>& dest) {}  // NOLINT(*)

  template <typename... Args>
  static void AgregateImpl(Array<T>& dest, Array<T> value, Args... args) {  // NOLINT(*)
    dest.insert(dest.end(), value.begin(), value.end());
    AgregateImpl(dest, args...);
  }

  template <typename... Args>
  static void AgregateImpl(Array<T>& dest, T value, Args... args) {  // NOLINT(*)
    dest.push_back(value);
    AgregateImpl(dest, args...);
  }

 public:
  // Array's own methods

  /*!
   * \brief set i-th element of the array.
   * \param i The index
   * \param value The value to be setted.
   */
  void Set(int64_t i, T value) {
    ArrayObj* p = this->CopyOnWrite();
    if (i < 0 || i >= p->size_) {
      TVM_FFI_THROW(IndexError) << "indexing " << i << " on an array of size " << p->size_;
    }
    *(p->MutableBegin() + i) = std::move(value);
  }

  /*! \return The underlying ArrayObj */
  ArrayObj* GetArrayObj() const { return static_cast<ArrayObj*>(data_.get()); }

  /*!
   * \brief Helper function to apply a map function onto the array.
   *
   * \param fmap The transformation function T -> U.
   *
   * \tparam F The type of the mutation function.
   *
   * \tparam U The type of the returned array, inferred from the
   * return type of F.  If overridden by the user, must be something
   * that is convertible from the return type of F.
   *
   * \note This function performs copy on write optimization.  If
   * `fmap` returns an object of type `T`, and all elements of the
   * array are mapped to themselves, then the returned array will be
   * the same as the original, and reference counts of the elements in
   * the array will not be incremented.
   *
   * \return The transformed array.
   */
  template <typename F, typename U = std::invoke_result_t<F, T>>
  Array<U> Map(F fmap) const {
    return Array<U>(MapHelper(data_, fmap));
  }

  /*!
   * \brief Helper function to apply fmutate to mutate an array.
   * \param fmutate The transformation function T -> T.
   * \tparam F the type of the mutation function.
   * \note This function performs copy on write optimization.
   */
  template <typename F, typename = std::enable_if_t<std::is_same_v<T, std::invoke_result_t<F, T>>>>
  void MutateByApply(F fmutate) {
    data_ = MapHelper(std::move(data_), fmutate);
  }

  /*!
   * \brief reset the array to content from iterator.
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  void Assign(IterType first, IterType last) {
    int64_t cap = std::distance(first, last);
    if (cap < 0) {
      TVM_FFI_THROW(ValueError) << "cannot construct an Array of negative size";
    }
    ArrayObj* p = GetArrayObj();
    if (p != nullptr && data_.unique() && p->capacity_ >= cap) {
      // do not have to make new space
      p->clear();
    } else {
      // create new space
      data_ = ArrayObj::Empty(cap);
      p = GetArrayObj();
    }
    // To ensure exception safety, size is only incremented after the initialization succeeds
    Any* itr = p->MutableBegin();
    for (int64_t& i = p->size_ = 0; i < cap; ++i, ++first, ++itr) {
      new (itr) Any(*first);
    }
  }

  /*!
   * \brief Copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  ArrayObj* CopyOnWrite() {
    if (data_ == nullptr) {
      return SwitchContainer(ArrayObj::kInitSize);
    }
    if (!data_.unique()) {
      return SwitchContainer(capacity());
    }
    return static_cast<ArrayObj*>(data_.get());
  }

  /*! \brief specify container node */
  using ContainerType = ArrayObj;

  /*!
   * \brief Agregate arguments into a single Array<T>
   * \param args sequence of T or Array<T> elements
   * \return Agregated Array<T>
   */
  template <typename... Args>
  static Array<T> Agregate(Args... args) {
    Array<T> result;
    result.reserve(CalcCapacityImpl(args...));
    AgregateImpl(result, args...);
    return result;
  }

 private:
  /*!
   * \brief Implement copy-on-write semantics, and ensures capacity is enough for extra elements.
   * \param reserve_extra Number of extra slots needed
   * \return ArrayObj pointer to the unique copy
   */
  ArrayObj* CopyOnWrite(int64_t reserve_extra) {
    ArrayObj* p = GetArrayObj();
    if (p == nullptr) {
      // necessary to get around the constexpr address issue before c++17
      const int64_t kInitSize = ArrayObj::kInitSize;
      return SwitchContainer(std::max(kInitSize, reserve_extra));
    }
    if (p->capacity_ >= p->size_ + reserve_extra) {
      return CopyOnWrite();
    }
    int64_t cap = p->capacity_ * ArrayObj::kIncFactor;
    cap = std::max(cap, p->size_ + reserve_extra);
    return SwitchContainer(cap);
  }

  /*!
   * \brief Move or copy the ArrayObj to new address with the given capacity
   * \param capacity The capacity requirement of the new address
   */
  ArrayObj* SwitchContainer(int64_t capacity) {
    if (data_ == nullptr) {
      data_ = ArrayObj::Empty(capacity);
    } else if (data_.unique()) {
      data_ = ArrayObj::MoveFrom(capacity, GetArrayObj());
    } else {
      data_ = ArrayObj::CopyFrom(capacity, GetArrayObj());
    }
    return static_cast<ArrayObj*>(data_.get());
  }

  /*! \brief Helper method for mutate/map
   *
   * A helper function used internally by both `Array::Map` and
   * `Array::MutateInPlace`.  Given an array of data, apply the
   * mapping function to each element, returning the collected array.
   * Applies both mutate-in-place and copy-on-write optimizations, if
   * possible.
   *
   * \param data A pointer to the ArrayObj containing input data.
   * Passed by value to allow for mutate-in-place optimizations.
   *
   * \param fmap The mapping function
   *
   * \tparam F The type of the mutation function.
   *
   * \tparam U The output type of the mutation function.  Inferred
   * from the callable type given.  Must inherit from ObjectRef.
   *
   * \return The mapped array.  Depending on whether mutate-in-place
   * or copy-on-write optimizations were applicable, may be the same
   * underlying array as the `data` parameter.
   */
  template <typename F, typename U = std::invoke_result_t<F, T>>
  static ObjectPtr<Object> MapHelper(ObjectPtr<Object> data, F fmap) {
    if (data == nullptr) {
      return nullptr;
    }

    TVM_FFI_ICHECK(data->IsInstance<ArrayObj>());

    constexpr bool is_same_output_type = std::is_same_v<T, U>;

    if constexpr (is_same_output_type) {
      if (data.unique()) {
        // Mutate-in-place path.  Only allowed if the output type U is
        // the same as type T, we have a mutable this*, and there are
        // no other shared copies of the array.
        auto arr = static_cast<ArrayObj*>(data.get());
        for (auto it = arr->MutableBegin(); it != arr->MutableEnd(); it++) {
          T value = details::AnyUnsafe::CopyFromAnyStorageAfterCheck<T>(*it);
          // reset the original value to nullptr, to ensure unique ownership
          it->reset();
          T mapped = fmap(std::move(value));
          *it = std::move(mapped);
        }
        return data;
      }
    }

    constexpr bool compatible_types = is_valid_iterator_v<T, U*> || is_valid_iterator_v<U, T*>;

    ObjectPtr<ArrayObj> output = nullptr;
    auto arr = static_cast<ArrayObj*>(data.get());

    auto it = arr->begin();
    if constexpr (compatible_types) {
      // Copy-on-write path, if the output Array<U> might be
      // represented by the same underlying array as the existing
      // Array<T>.  Typically, this is for functions that map `T` to
      // `T`, but can also apply to functions that map `T` to
      // `Optional<T>`, or that map `T` to a subclass or superclass of
      // `T`.
      bool all_identical = true;
      for (; it != arr->end(); it++) {
        U mapped = fmap(details::AnyUnsafe::CopyFromAnyStorageAfterCheck<T>(*it));
        if (!(*it).same_as(mapped)) {
          // At least one mapped element is different than the
          // original.  Therefore, prepare the output array,
          // consisting of any previous elements that had mapped to
          // themselves (if any), and the element that didn't map to
          // itself.
          //
          // We cannot use `U()` as the default object, as `U` may be
          // a non-nullable type.  Since the default `Any()`
          // will be overwritten before returning, all objects will be
          // of type `U` for the calling scope.
          all_identical = false;
          output = ArrayObj::CreateRepeated(arr->size(), Any());
          output->InitRange(0, arr->begin(), it);
          output->SetItem(it - arr->begin(), std::move(mapped));
          it++;
          break;
        }
      }
      if (all_identical) {
        return data;
      }
    } else {
      // Path for incompatible types.  The constexpr check for
      // compatible types isn't strictly necessary, as the first
      // (*it).same_as(mapped) would return false, but we might as well
      // avoid it altogether.
      //
      // We cannot use `U()` as the default object, as `U` may be a
      // non-nullable type.  Since the default `Any()` will be
      // overwritten before returning, all objects will be of type `U`
      // for the calling scope.
      output = ArrayObj::CreateRepeated(arr->size(), Any());
    }

    // Normal path for incompatible types, or post-copy path for
    // copy-on-write instances.
    //
    // If the types are incompatible, then at this point `output` is
    // empty, and `it` points to the first element of the input.
    //
    // If the types were compatible, then at this point `output`
    // contains zero or more elements that mapped to themselves
    // followed by the first element that does not map to itself, and
    // `it` points to the element just after the first element that
    // does not map to itself.  Because at least one element has been
    // changed, we no longer have the opportunity to avoid a copy, so
    // we don't need to check the result.
    //
    // In both cases, `it` points to the next element to be processed,
    // so we can either start or resume the iteration from that point,
    // with no further checks on the result.
    for (; it != arr->end(); it++) {
      U mapped = fmap(details::AnyUnsafe::CopyFromAnyStorageAfterCheck<T>(*it));
      output->SetItem(it - arr->begin(), std::move(mapped));
    }

    return output;
  }
  template <typename, typename>
  friend class Array;
};

/*!
 * \brief Concat two Arrays.
 * \param lhs first Array to be concatenated.
 * \param rhs second Array to be concatenated.
 * \return The concatenated Array. Original Arrays are kept unchanged.
 */
template <typename T, typename = typename std::enable_if_t<std::is_same_v<T, Any> ||
                                                           TypeTraits<T>::convert_enabled>>
inline Array<T> Concat(Array<T> lhs, const Array<T>& rhs) {
  for (const auto& x : rhs) {
    lhs.push_back(x);
  }
  return std::move(lhs);
}

// Specialize make_object<ArrayObj> to make sure it is correct.
template <>
inline ObjectPtr<ArrayObj> make_object() {
  return ArrayObj::Empty();
}

// Traits for Array
template <typename T>
inline constexpr bool use_default_type_traits_v<Array<T>> = false;

template <typename T>
struct TypeTraits<Array<T>> : public ObjectRefTypeTraitsBase<Array<T>> {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIArray;
  using ObjectRefTypeTraitsBase<Array<T>>::CopyFromAnyStorageAfterCheck;

  static TVM_FFI_INLINE std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) {
      return TypeTraitsBase::GetMismatchTypeInfo(src);
    }
    if constexpr (!std::is_same_v<T, Any>) {
      const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
      for (size_t i = 0; i < n->size(); i++) {
        const Any& any_v = (*n)[i];
        // CheckAnyStorage is cheaper than as<T>
        if (details::AnyUnsafe::CheckAnyStorage<T>(any_v)) continue;
        // try see if p is convertible to T
        if (any_v.as<T>()) continue;
        // now report the accurate mismatch information
        return "Array[index " + std::to_string(i) + ": " +
               details::AnyUnsafe::GetMismatchTypeInfo<T>(any_v) + "]";
      }
    }
    TVM_FFI_THROW(InternalError) << "Cannot reach here";
    TVM_FFI_UNREACHABLE();
  }

  static TVM_FFI_INLINE bool CheckAnyStorage(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIArray) return false;
    if constexpr (std::is_same_v<T, Any>) {
      return true;
    } else {
      const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
      for (size_t i = 0; i < n->size(); i++) {
        const Any& any_v = (*n)[i];
        if (!details::AnyUnsafe::CheckAnyStorage<T>(any_v)) return false;
      }
      return true;
    }
  }

  static TVM_FFI_INLINE std::optional<Array<T>> TryConvertFromAnyView(const TVMFFIAny* src) {
    // try to run conversion.
    if (src->type_index != TypeIndex::kTVMFFIArray) return std::nullopt;
    if constexpr (!std::is_same_v<T, Any>) {
      const ArrayObj* n = reinterpret_cast<const ArrayObj*>(src->v_obj);
      bool storage_check = [&]() {
        for (size_t i = 0; i < n->size(); i++) {
          const Any& any_v = (*n)[i];
          if (!details::AnyUnsafe::CheckAnyStorage<T>(any_v)) return false;
        }
        return true;
      }();
      // fast path, if storage check passes, we can return the array directly.
      if (storage_check) {
        return CopyFromAnyStorageAfterCheck(src);
      }
      // slow path, try to run a conversion to Array<T>
      Array<T> result;
      result.reserve(n->size());
      for (size_t i = 0; i < n->size(); i++) {
        const Any& any_v = (*n)[i];
        if (auto opt_v = any_v.as<T>()) {
          result.push_back(*std::move(opt_v));
        } else {
          return std::nullopt;
        }
      }
      return result;
    } else {
      return CopyFromAnyStorageAfterCheck(src);
    }
  }

  static TVM_FFI_INLINE std::string TypeStr() { return "Array<" + details::Type2Str<T>::v() + ">"; }
};

namespace details {
template <typename T, typename U>
inline constexpr bool type_contains_v<Array<T>, Array<U>> = type_contains_v<T, U>;
}  // namespace details

}  // namespace ffi

// Expose to the tvm namespace
// Rationale: convinience and no ambiguity
using ffi::Array;
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_ARRAY_H_
