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
 * \file tvm/runtime/container/array.h
 * \brief Runtime Array container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_ARRAY_H_
#define TVM_RUNTIME_CONTAINER_ARRAY_H_

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "./base.h"
#include "./optional.h"

namespace tvm {
namespace runtime {

/*! \brief array node content in array */
class ArrayNode : public Object, public InplaceArrayBase<ArrayNode, ObjectRef> {
 public:
  /*! \return The size of the array */
  size_t size() const { return this->size_; }

  /*!
   * \brief Read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  const ObjectRef at(int64_t i) const { return this->operator[](i); }

  /*! \return begin constant iterator */
  const ObjectRef* begin() const { return static_cast<ObjectRef*>(InplaceArrayBase::AddressOf(0)); }

  /*! \return end constant iterator */
  const ObjectRef* end() const { return begin() + size_; }

  /*! \brief Release reference to all the elements */
  void clear() { ShrinkBy(size_); }

  /*!
   * \brief Set i-th element of the array in-place
   * \param i The index
   * \param item The value to be set
   */
  void SetItem(int64_t i, ObjectRef item) { this->operator[](i) = std::move(item); }

  /*!
   * \brief Constructs a container and copy from another
   * \param cap The capacity of the container
   * \param from Source of the copy
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> CopyFrom(int64_t cap, ArrayNode* from) {
    int64_t size = from->size_;
    ICHECK_GE(cap, size) << "ValueError: not enough capacity";
    ObjectPtr<ArrayNode> p = ArrayNode::Empty(cap);
    ObjectRef* write = p->MutableBegin();
    ObjectRef* read = from->MutableBegin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (int64_t& i = p->size_ = 0; i < size; ++i) {
      new (write++) ObjectRef(*read++);
    }
    return p;
  }

  /*!
   * \brief Constructs a container and move from another
   * \param cap The capacity of the container
   * \param from Source of the move
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> MoveFrom(int64_t cap, ArrayNode* from) {
    int64_t size = from->size_;
    ICHECK_GE(cap, size) << "ValueError: not enough capacity";
    ObjectPtr<ArrayNode> p = ArrayNode::Empty(cap);
    ObjectRef* write = p->MutableBegin();
    ObjectRef* read = from->MutableBegin();
    // To ensure exception safety, size is only incremented after the initialization succeeds
    for (int64_t& i = p->size_ = 0; i < size; ++i) {
      new (write++) ObjectRef(std::move(*read++));
    }
    from->size_ = 0;
    return p;
  }

  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> CreateRepeated(int64_t n, const ObjectRef& val) {
    ObjectPtr<ArrayNode> p = ArrayNode::Empty(n);
    ObjectRef* itr = p->MutableBegin();
    for (int64_t& i = p->size_ = 0; i < n; ++i) {
      new (itr++) ObjectRef(val);
    }
    return p;
  }

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeArray;
  static constexpr const char* _type_key = "Array";
  TVM_DECLARE_FINAL_OBJECT_INFO(ArrayNode, Object);

 private:
  /*! \return Size of initialized memory, used by InplaceArrayBase. */
  size_t GetSize() const { return this->size_; }

  /*! \return begin mutable iterator */
  ObjectRef* MutableBegin() const {
    return static_cast<ObjectRef*>(InplaceArrayBase::AddressOf(0));
  }

  /*! \return end mutable iterator */
  ObjectRef* MutableEnd() const { return MutableBegin() + size_; }

  /*!
   * \brief Create an ArrayNode with the given capacity.
   * \param n Required capacity
   * \return Ref-counted ArrayNode requested
   */
  static ObjectPtr<ArrayNode> Empty(int64_t n = kInitSize) {
    ICHECK_GE(n, 0);
    ObjectPtr<ArrayNode> p = make_inplace_array_object<ArrayNode, ObjectRef>(n);
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
  ArrayNode* InitRange(int64_t idx, IterType first, IterType last) {
    ObjectRef* itr = MutableBegin() + idx;
    for (; first != last; ++first) {
      ObjectRef ref = *first;
      new (itr++) ObjectRef(std::move(ref));
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
  ArrayNode* MoveElementsLeft(int64_t dst, int64_t src_begin, int64_t src_end) {
    ObjectRef* from = MutableBegin() + src_begin;
    ObjectRef* to = MutableBegin() + dst;
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
  ArrayNode* MoveElementsRight(int64_t dst, int64_t src_begin, int64_t src_end) {
    ObjectRef* from = MutableBegin() + src_end;
    ObjectRef* to = MutableBegin() + (src_end - src_begin + dst);
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
  ArrayNode* EnlargeBy(int64_t delta, const ObjectRef& val = ObjectRef(nullptr)) {
    ObjectRef* itr = MutableEnd();
    while (delta-- > 0) {
      new (itr++) ObjectRef(val);
      ++size_;
    }
    return this;
  }

  /*!
   * \brief Shrinks the size of the array
   * \param delta Size shrinked, should be positive
   * \return Self
   */
  ArrayNode* ShrinkBy(int64_t delta) {
    ObjectRef* itr = MutableEnd();
    while (delta-- > 0) {
      (--itr)->ObjectRef::~ObjectRef();
      --size_;
    }
    return this;
  }

  /*! \brief Number of elements used */
  int64_t size_;

  /*! \brief Number of elements allocated */
  int64_t capacity_;

  /*! \brief Initial size of ArrayNode */
  static constexpr int64_t kInitSize = 4;

  /*! \brief Expansion factor of the Array */
  static constexpr int64_t kIncFactor = 2;

  // CRTP parent class
  friend InplaceArrayBase<ArrayNode, ObjectRef>;

  // Reference class
  template <typename, typename>
  friend class Array;

  // To specialize make_object<ArrayNode>
  friend ObjectPtr<ArrayNode> make_object<>();
};

/*! \brief Helper struct for type-checking
 *
 * is_valid_iterator<T,IterType>::value will be true if IterType can
 * be dereferenced into a type that can be stored in an Array<T>, and
 * false otherwise.
 */
template <typename T, typename IterType>
struct is_valid_iterator
    : std::bool_constant<std::is_base_of_v<
          T, std::remove_cv_t<std::remove_reference_t<decltype(*std::declval<IterType>())>>>> {};

template <typename T, typename IterType>
struct is_valid_iterator<Optional<T>, IterType> : is_valid_iterator<T, IterType> {};

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
 * \tparam T The content ObjectRef type.
 */
template <typename T,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
class Array : public ObjectRef {
 public:
  using value_type = T;
  // constructors
  /*!
   * \brief default constructor
   */
  Array() { data_ = ArrayNode::Empty(); }

  /*!
   * \brief move constructor
   * \param other source
   */
  Array(Array<T>&& other) : ObjectRef() {  // NOLINT(*)
    data_ = std::move(other.data_);
  }

  /*!
   * \brief copy constructor
   * \param other source
   */
  Array(const Array<T>& other) : ObjectRef() {  // NOLINT(*)
    data_ = other.data_;
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
  explicit Array(const size_t n, const T& val) { data_ = ArrayNode::CreateRepeated(n, val); }

  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(Array<T>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(const Array<T>& other) {
    data_ = other.data_;
    return *this;
  }

 public:
  // iterators
  struct ValueConverter {
    using ResultType = T;
    static T convert(const ObjectRef& n) { return DowncastNoCheck<T>(n); }
  };

  using iterator = IterAdapter<ValueConverter, const ObjectRef*>;
  using reverse_iterator = ReverseIterAdapter<ValueConverter, const ObjectRef*>;

  /*! \return begin iterator */
  iterator begin() const { return iterator(GetArrayNode()->begin()); }

  /*! \return end iterator */
  iterator end() const { return iterator(GetArrayNode()->end()); }

  /*! \return rbegin iterator */
  reverse_iterator rbegin() const {
    // ArrayNode::end() is never nullptr
    return reverse_iterator(GetArrayNode()->end() - 1);
  }

  /*! \return rend iterator */
  reverse_iterator rend() const {
    // ArrayNode::begin() is never nullptr
    return reverse_iterator(GetArrayNode()->begin() - 1);
  }

 public:
  // const methods in std::vector
  /*!
   * \brief Immutably read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  const T operator[](int64_t i) const {
    ArrayNode* p = GetArrayNode();
    ICHECK(p != nullptr) << "ValueError: cannot index a null array";
    ICHECK(0 <= i && i < p->size_)
        << "IndexError: indexing " << i << " on an array of size " << p->size_;
    return DowncastNoCheck<T>(*(p->begin() + i));
  }

  /*! \return The size of the array */
  size_t size() const {
    ArrayNode* p = GetArrayNode();
    return p == nullptr ? 0 : GetArrayNode()->size_;
  }

  /*! \return The capacity of the array */
  size_t capacity() const {
    ArrayNode* p = GetArrayNode();
    return p == nullptr ? 0 : GetArrayNode()->capacity_;
  }

  /*! \return Whether array is empty */
  bool empty() const { return size() == 0; }

  /*! \return The first element of the array */
  const T front() const {
    ArrayNode* p = GetArrayNode();
    ICHECK(p != nullptr) << "ValueError: cannot index a null array";
    ICHECK_GT(p->size_, 0) << "IndexError: cannot index an empty array";
    return DowncastNoCheck<T>(*(p->begin()));
  }

  /*! \return The last element of the array */
  const T back() const {
    ArrayNode* p = GetArrayNode();
    ICHECK(p != nullptr) << "ValueError: cannot index a null array";
    ICHECK_GT(p->size_, 0) << "IndexError: cannot index an empty array";
    return DowncastNoCheck<T>(*(p->end() - 1));
  }

 public:
  // mutation in std::vector, implements copy-on-write

  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  void push_back(const T& item) {
    ArrayNode* p = CopyOnWrite(1);
    p->EmplaceInit(p->size_++, item);
  }

  /*!
   * \brief Insert an element into the given position
   * \param position An iterator pointing to the insertion point
   * \param val The element to insert
   */
  void insert(iterator position, const T& val) {
    ICHECK(data_ != nullptr) << "ValueError: cannot insert a null array";
    int64_t idx = std::distance(begin(), position);
    int64_t size = GetArrayNode()->size_;
    auto addr = CopyOnWrite(1)                               //
                    ->EnlargeBy(1)                           //
                    ->MoveElementsRight(idx + 1, idx, size)  //
                    ->MutableBegin();
    new (addr + idx) ObjectRef(val);
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
    ICHECK(data_ != nullptr) << "ValueError: cannot insert a null array";
    int64_t idx = std::distance(begin(), position);
    int64_t size = GetArrayNode()->size_;
    int64_t numel = std::distance(first, last);
    CopyOnWrite(numel)
        ->EnlargeBy(numel)
        ->MoveElementsRight(idx + numel, idx, size)
        ->InitRange(idx, first, last);
  }

  /*! \brief Remove the last item of the list */
  void pop_back() {
    ICHECK(data_ != nullptr) << "ValueError: cannot pop_back because array is null";
    int64_t size = GetArrayNode()->size_;
    ICHECK_GT(size, 0) << "ValueError: cannot pop_back because array is empty";
    CopyOnWrite()->ShrinkBy(1);
  }

  /*!
   * \brief Erase an element on the given position
   * \param position An iterator pointing to the element to be erased
   */
  void erase(iterator position) {
    ICHECK(data_ != nullptr) << "ValueError: cannot erase a null array";
    int64_t st = std::distance(begin(), position);
    int64_t size = GetArrayNode()->size_;
    ICHECK(0 <= st && st < size) << "ValueError: cannot erase at index " << st
                                 << ", because Array size is " << size;
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
    ICHECK(data_ != nullptr) << "ValueError: cannot erase a null array";
    int64_t size = GetArrayNode()->size_;
    int64_t st = std::distance(begin(), first);
    int64_t ed = std::distance(begin(), last);
    ICHECK_LT(st, ed) << "ValueError: cannot erase array in range [" << st << ", " << ed << ")";
    ICHECK(0 <= st && st <= size && 0 <= ed && ed <= size)
        << "ValueError: cannot erase array in range [" << st << ", " << ed << ")"
        << ", because array size is " << size;
    CopyOnWrite()                         //
        ->MoveElementsLeft(st, ed, size)  //
        ->ShrinkBy(ed - st);
  }

  /*!
   * \brief Resize the array.
   * \param n The new size.
   */
  void resize(int64_t n) {
    ICHECK_GE(n, 0) << "ValueError: cannot resize an Array to negative size";
    if (data_ == nullptr) {
      SwitchContainer(n);
      return;
    }
    int64_t size = GetArrayNode()->size_;
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
    if (data_ == nullptr || n > GetArrayNode()->capacity_) {
      SwitchContainer(n);
    }
  }

  /*! \brief Release reference to all the elements */
  void clear() {
    if (data_ != nullptr) {
      ArrayNode* p = CopyOnWrite();
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
    ArrayNode* p = this->CopyOnWrite();
    ICHECK(0 <= i && i < p->size_)
        << "IndexError: indexing " << i << " on an array of size " << p->size_;
    *(p->MutableBegin() + i) = std::move(value);
  }

  /*! \return The underlying ArrayNode */
  ArrayNode* GetArrayNode() const { return static_cast<ArrayNode*>(data_.get()); }

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
    ICHECK_GE(cap, 0) << "ValueError: cannot construct an Array of negative size";
    ArrayNode* p = GetArrayNode();
    if (p != nullptr && data_.unique() && p->capacity_ >= cap) {
      // do not have to make new space
      p->clear();
    } else {
      // create new space
      data_ = ArrayNode::Empty(cap);
      p = GetArrayNode();
    }
    // To ensure exception safety, size is only incremented after the initialization succeeds
    ObjectRef* itr = p->MutableBegin();
    for (int64_t& i = p->size_ = 0; i < cap; ++i, ++first, ++itr) {
      new (itr) ObjectRef(*first);
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
  ArrayNode* CopyOnWrite() {
    if (data_ == nullptr) {
      return SwitchContainer(ArrayNode::kInitSize);
    }
    if (!data_.unique()) {
      return SwitchContainer(capacity());
    }
    return static_cast<ArrayNode*>(data_.get());
  }

  /*! \brief specify container node */
  using ContainerType = ArrayNode;

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
   * \return ArrayNode pointer to the unique copy
   */
  ArrayNode* CopyOnWrite(int64_t reserve_extra) {
    ArrayNode* p = GetArrayNode();
    if (p == nullptr) {
      // necessary to get around the constexpr address issue before c++17
      const int64_t kInitSize = ArrayNode::kInitSize;
      return SwitchContainer(std::max(kInitSize, reserve_extra));
    }
    if (p->capacity_ >= p->size_ + reserve_extra) {
      return CopyOnWrite();
    }
    int64_t cap = p->capacity_ * ArrayNode::kIncFactor;
    cap = std::max(cap, p->size_ + reserve_extra);
    return SwitchContainer(cap);
  }

  /*!
   * \brief Move or copy the ArrayNode to new address with the given capacity
   * \param capacity The capacity requirement of the new address
   */
  ArrayNode* SwitchContainer(int64_t capacity) {
    if (data_ == nullptr) {
      data_ = ArrayNode::Empty(capacity);
    } else if (data_.unique()) {
      data_ = ArrayNode::MoveFrom(capacity, GetArrayNode());
    } else {
      data_ = ArrayNode::CopyFrom(capacity, GetArrayNode());
    }
    return static_cast<ArrayNode*>(data_.get());
  }

  /*! \brief Helper method for mutate/map
   *
   * A helper function used internally by both `Array::Map` and
   * `Array::MutateInPlace`.  Given an array of data, apply the
   * mapping function to each element, returning the collected array.
   * Applies both mutate-in-place and copy-on-write optimizations, if
   * possible.
   *
   * \param data A pointer to the ArrayNode containing input data.
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

    ICHECK(data->IsInstance<ArrayNode>());

    constexpr bool is_same_output_type = std::is_same_v<T, U>;

    if constexpr (is_same_output_type) {
      if (data.unique()) {
        // Mutate-in-place path.  Only allowed if the output type U is
        // the same as type T, we have a mutable this*, and there are
        // no other shared copies of the array.
        auto arr = static_cast<ArrayNode*>(data.get());
        for (auto it = arr->MutableBegin(); it != arr->MutableEnd(); it++) {
          T mapped = fmap(DowncastNoCheck<T>(std::move(*it)));
          *it = std::move(mapped);
        }
        return data;
      }
    }

    constexpr bool compatible_types = is_valid_iterator_v<T, U*> || is_valid_iterator_v<U, T*>;

    ObjectPtr<ArrayNode> output = nullptr;
    auto arr = static_cast<ArrayNode*>(data.get());

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
        U mapped = fmap(DowncastNoCheck<T>(*it));
        if (!mapped.same_as(*it)) {
          // At least one mapped element is different than the
          // original.  Therefore, prepare the output array,
          // consisting of any previous elements that had mapped to
          // themselves (if any), and the element that didn't map to
          // itself.
          //
          // We cannot use `U()` as the default object, as `U` may be
          // a non-nullable type.  Since the default `ObjectRef()`
          // will be overwritten before returning, all objects will be
          // of type `U` for the calling scope.
          all_identical = false;
          output = ArrayNode::CreateRepeated(arr->size(), ObjectRef());
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
      // mapped.same_as(*it) would return false, but we might as well
      // avoid it altogether.
      //
      // We cannot use `U()` as the default object, as `U` may be a
      // non-nullable type.  Since the default `ObjectRef()` will be
      // overwritten before returning, all objects will be of type `U`
      // for the calling scope.
      output = ArrayNode::CreateRepeated(arr->size(), ObjectRef());
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
      U mapped = fmap(DowncastNoCheck<T>(*it));
      output->SetItem(it - arr->begin(), std::move(mapped));
    }

    return output;
  }
};

template <typename T>
inline constexpr bool is_tvm_array = false;

template <typename T>
inline constexpr bool is_tvm_array<Array<T>> = true;

/*!
 * \brief Concat two Arrays.
 * \param lhs first Array to be concatenated.
 * \param rhs second Array to be concatenated.
 * \return The concatenated Array. Original Arrays are kept unchanged.
 */
template <typename T,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type>
inline Array<T> Concat(Array<T> lhs, const Array<T>& rhs) {
  for (const auto& x : rhs) {
    lhs.push_back(x);
  }
  return std::move(lhs);
}

// Specialize make_object<ArrayNode> to make sure it is correct.
template <>
inline ObjectPtr<ArrayNode> make_object() {
  return ArrayNode::Empty();
}

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::Array;
using runtime::ArrayNode;
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_ARRAY_H_
