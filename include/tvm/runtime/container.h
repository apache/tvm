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
 * \file tvm/runtime/container.h
 * \brief Common POD(plain old data) container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_H_
#define TVM_RUNTIME_CONTAINER_H_

#include <dmlc/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
// We use c++14 std::experimental::string_view for optimizing hash computation
// only right now, its usage is limited in this file. Any broader usage of
// std::experiment in our core codebase is discouraged and needs community
// discussion for each use case. Reference for feature test macros of
// string_view:
// https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations
// https://en.cppreference.com/w/User:D41D8CD98F/feature_testing_macros
#if defined(__cpp_lib_experimental_string_view) && __cpp_lib_experimental_string_view >= 201411
#define TVM_USE_CXX14_STRING_VIEW_HASH 1
#else
#define TVM_USE_CXX14_STRING_VIEW_HASH 0
#endif

// Tested with clang version 9.0.1 and c++17. It will detect string_view support
// correctly.
#if defined(__cpp_lib_string_view) && __cpp_lib_string_view >= 201606
#define TVM_USE_CXX17_STRING_VIEW_HASH 1
#else
#define TVM_USE_CXX17_STRING_VIEW_HASH 0
#endif

#if TVM_USE_CXX17_STRING_VIEW_HASH
#include <string_view>
#elif TVM_USE_CXX14_STRING_VIEW_HASH
#include <experimental/string_view>
#endif

#include <type_traits>
#include <utility>
#include <vector>

namespace llvm {
// String to llvm object compatibility.
class StringRef;
}  // namespace llvm

namespace tvm {
namespace runtime {

/*! \brief String-aware ObjectRef equal functor */
struct ObjectHash {
  /*!
   * \brief Calculate the hash code of an ObjectRef
   * \param a The given ObjectRef
   * \return Hash code of a, string hash for strings and pointer address otherwise.
   */
  size_t operator()(const ObjectRef& a) const;
};

/*! \brief String-aware ObjectRef hash functor */
struct ObjectEqual {
  /*!
   * \brief Check if the two ObjectRef are equal
   * \param a One ObjectRef
   * \param b The other ObjectRef
   * \return String equality if both are strings, pointer address equality otherwise.
   */
  bool operator()(const ObjectRef& a, const ObjectRef& b) const;
};

/*!
 * \brief Base template for classes with array like memory layout.
 *
 *        It provides general methods to access the memory. The memory
 *        layout is ArrayType + [ElemType]. The alignment of ArrayType
 *        and ElemType is handled by the memory allocator.
 *
 * \tparam ArrayType The array header type, contains object specific metadata.
 * \tparam ElemType The type of objects stored in the array right after
 * ArrayType.
 *
 * \code
 * // Example usage of the template to define a simple array wrapper
 * class ArrayObj : public InplaceArrayBase<ArrayObj, Elem> {
 * public:
 *  // Wrap EmplaceInit to initialize the elements
 *  template <typename Iterator>
 *  void Init(Iterator begin, Iterator end) {
 *   size_t num_elems = std::distance(begin, end);
 *   auto it = begin;
 *   this->size = 0;
 *   for (size_t i = 0; i < num_elems; ++i) {
 *     InplaceArrayBase::EmplaceInit(i, *it++);
 *     this->size++;
 *   }
 *  }
 * }
 *
 * void test_function() {
 *   vector<Elem> fields;
 *   auto ptr = make_inplace_array_object<ArrayObj, Elem>(fields.size());
 *   ptr->Init(fields.begin(), fields.end());
 *
 *   // Access the 0th element in the array.
 *   assert(ptr->operator[](0) == fields[0]);
 * }
 *
 * \endcode
 */
template <typename ArrayType, typename ElemType>
class InplaceArrayBase {
 public:
  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Const reference to ElemType at the index.
   */
  const ElemType& operator[](size_t idx) const {
    size_t size = Self()->GetSize();
    CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Reference to ElemType at the index.
   */
  ElemType& operator[](size_t idx) {
    size_t size = Self()->GetSize();
    CHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Destroy the Inplace Array Base object
   */
  ~InplaceArrayBase() {
    if (!(std::is_standard_layout<ElemType>::value && std::is_trivial<ElemType>::value)) {
      size_t size = Self()->GetSize();
      for (size_t i = 0; i < size; ++i) {
        ElemType* fp = reinterpret_cast<ElemType*>(AddressOf(i));
        fp->ElemType::~ElemType();
      }
    }
  }

 protected:
  /*!
   * \brief Construct a value in place with the arguments.
   *
   * \tparam Args Type parameters of the arguments.
   * \param idx Index of the element.
   * \param args Arguments to construct the new value.
   *
   * \note Please make sure ArrayType::GetSize returns 0 before first call of
   * EmplaceInit, and increment GetSize by 1 each time EmplaceInit succeeds.
   */
  template <typename... Args>
  void EmplaceInit(size_t idx, Args&&... args) {
    void* field_ptr = AddressOf(idx);
    new (field_ptr) ElemType(std::forward<Args>(args)...);
  }

  /*!
   * \brief Return the self object for the array.
   *
   * \return Pointer to ArrayType.
   */
  inline ArrayType* Self() const {
    return static_cast<ArrayType*>(const_cast<InplaceArrayBase*>(this));
  }

  /*!
   * \brief Return the raw pointer to the element at idx.
   *
   * \param idx The index of the element.
   * \return Raw pointer to the element.
   */
  void* AddressOf(size_t idx) const {
    static_assert(
        alignof(ArrayType) % alignof(ElemType) == 0 && sizeof(ArrayType) % alignof(ElemType) == 0,
        "The size and alignment of ArrayType should respect "
        "ElemType's alignment.");

    size_t kDataStart = sizeof(ArrayType);
    ArrayType* self = Self();
    char* data_start = reinterpret_cast<char*>(self) + kDataStart;
    return data_start + idx * sizeof(ElemType);
  }
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template <typename Converter, typename TIter>
class IterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = typename Converter::ResultType*;
  using reference = typename Converter::ResultType&;
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit IterAdapter(TIter iter) : iter_(iter) {}
  IterAdapter& operator++() {
    ++iter_;
    return *this;
  }
  IterAdapter& operator--() {
    --iter_;
    return *this;
  }
  IterAdapter operator++(int) {
    IterAdapter copy = *this;
    ++iter_;
    return copy;
  }
  IterAdapter operator--(int) {
    IterAdapter copy = *this;
    --iter_;
    return copy;
  }

  IterAdapter operator+(difference_type offset) const { return IterAdapter(iter_ + offset); }

  template <typename T = IterAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type inline
  operator-(const IterAdapter& rhs) const {
    return iter_ - rhs.iter_;
  }

  bool operator==(IterAdapter other) const { return iter_ == other.iter_; }
  bool operator!=(IterAdapter other) const { return !(*this == other); }
  const value_type operator*() const { return Converter::convert(*iter_); }

 private:
  TIter iter_;
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template <typename Converter, typename TIter>
class ReverseIterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = typename Converter::ResultType*;
  using reference = typename Converter::ResultType&;  // NOLINT(*)
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit ReverseIterAdapter(TIter iter) : iter_(iter) {}
  ReverseIterAdapter& operator++() {
    --iter_;
    return *this;
  }
  ReverseIterAdapter& operator--() {
    ++iter_;
    return *this;
  }
  ReverseIterAdapter& operator++(int) {
    ReverseIterAdapter copy = *this;
    --iter_;
    return copy;
  }
  ReverseIterAdapter& operator--(int) {
    ReverseIterAdapter copy = *this;
    ++iter_;
    return copy;
  }
  ReverseIterAdapter operator+(difference_type offset) const {
    return ReverseIterAdapter(iter_ - offset);
  }

  template <typename T = ReverseIterAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type inline
  operator-(const ReverseIterAdapter& rhs) const {
    return rhs.iter_ - iter_;
  }

  bool operator==(ReverseIterAdapter other) const { return iter_ == other.iter_; }
  bool operator!=(ReverseIterAdapter other) const { return !(*this == other); }
  const value_type operator*() const { return Converter::convert(*iter_); }

 private:
  TIter iter_;
};

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
    CHECK_GE(cap, size) << "ValueError: not enough capacity";
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
    CHECK_GE(cap, size) << "ValueError: not enough capacity";
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
    CHECK_GE(n, 0);
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

/*!
 * \brief Array, container representing a contigious sequence of ObjectRefs.
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
    CHECK(p != nullptr) << "ValueError: cannot index a null array";
    CHECK(0 <= i && i < p->size_) << "IndexError: indexing " << i << " on an array of size "
                                  << p->size_;
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
    CHECK(p != nullptr) << "ValueError: cannot index a null array";
    CHECK_GT(p->size_, 0) << "IndexError: cannot index an empty array";
    return DowncastNoCheck<T>(*(p->begin()));
  }

  /*! \return The last element of the array */
  const T back() const {
    ArrayNode* p = GetArrayNode();
    CHECK(p != nullptr) << "ValueError: cannot index a null array";
    CHECK_GT(p->size_, 0) << "IndexError: cannot index an empty array";
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
    CHECK(data_ != nullptr) << "ValueError: cannot insert a null array";
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
    if (first == last) {
      return;
    }
    CHECK(data_ != nullptr) << "ValueError: cannot insert a null array";
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
    CHECK(data_ != nullptr) << "ValueError: cannot pop_back because array is null";
    int64_t size = GetArrayNode()->size_;
    CHECK_GT(size, 0) << "ValueError: cannot pop_back because array is empty";
    CopyOnWrite()->ShrinkBy(1);
  }

  /*!
   * \brief Erase an element on the given position
   * \param position An iterator pointing to the element to be erased
   */
  void erase(iterator position) {
    CHECK(data_ != nullptr) << "ValueError: cannot erase a null array";
    int64_t st = std::distance(begin(), position);
    int64_t size = GetArrayNode()->size_;
    CHECK(0 <= st && st < size) << "ValueError: cannot erase at index " << st
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
    CHECK(data_ != nullptr) << "ValueError: cannot erase a null array";
    int64_t size = GetArrayNode()->size_;
    int64_t st = std::distance(begin(), first);
    int64_t ed = std::distance(begin(), last);
    CHECK_LT(st, ed) << "ValueError: cannot erase array in range [" << st << ", " << ed << ")";
    CHECK(0 <= st && st <= size && 0 <= ed && ed <= size)
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
    CHECK_GE(n, 0) << "ValueError: cannot resize an Array to negative size";
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

 public:
  // Array's own methods

  /*!
   * \brief set i-th element of the array.
   * \param i The index
   * \param value The value to be setted.
   */
  void Set(int64_t i, T value) {
    ArrayNode* p = this->CopyOnWrite();
    CHECK(0 <= i && i < p->size_) << "IndexError: indexing " << i << " on an array of size "
                                  << p->size_;
    *(p->MutableBegin() + i) = std::move(value);
  }

  /*! \return The underlying ArrayNode */
  ArrayNode* GetArrayNode() const { return static_cast<ArrayNode*>(data_.get()); }

  /*!
   * \brief Helper function to apply fmutate to mutate an array.
   * \param fmutate The transformation function T -> T.
   * \tparam F the type of the mutation function.
   * \note This function performs copy on write optimization.
   */
  template <typename F>
  void MutateByApply(F fmutate) {
    if (data_ == nullptr) {
      return;
    }
    struct StackFrame {
      ArrayNode* p;
      ObjectRef* itr;
      int64_t i;
      int64_t size;
    };
    std::unique_ptr<StackFrame> s = std::make_unique<StackFrame>();
    s->p = GetArrayNode();
    s->itr = s->p->MutableBegin();
    s->i = 0;
    s->size = s->p->size_;
    if (!data_.unique()) {
      // Loop invariant: keeps iterating when
      // 1) data is not unique
      // 2) no elements are actually mutated yet
      for (; s->i < s->size; ++s->i, ++s->itr) {
        T new_elem = fmutate(DowncastNoCheck<T>(*s->itr));
        // do nothing when there is no mutation
        if (new_elem.same_as(*s->itr)) {
          continue;
        }
        // loop invariant breaks when the first real mutation happens
        // we copy the elements into a new unique array
        ObjectPtr<ArrayNode> copy = ArrayNode::CopyFrom(s->p->capacity_, s->p);
        s->itr = copy->MutableBegin() + (s->i++);
        *s->itr++ = std::move(new_elem);
        data_ = std::move(copy);
        // make sure `data_` is unique and break
        break;
      }
    }
    // when execution comes to this line, it is guaranteed that either
    //    1) i == size
    // or 2) data_.unique() is true
    for (; s->i < s->size; ++s->i, ++s->itr) {
      *s->itr = std::move(fmutate(std::move(DowncastNoCheck<T>(std::move(*s->itr)))));
    }
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
    CHECK_GE(cap, 0) << "ValueError: cannot construct an Array of negative size";
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
};

// Specialize make_object<ArrayNode> to make sure it is correct.
template <>
inline ObjectPtr<ArrayNode> make_object() {
  return ArrayNode::Empty();
}

/*! \brief An object representing a structure or enumeration. */
class ADTObj : public Object, public InplaceArrayBase<ADTObj, ObjectRef> {
 public:
  /*! \brief The tag representing the constructor used. */
  int32_t tag;
  /*! \brief Number of fields in the ADT object. */
  uint32_t size;
  // The fields of the structure follows directly in memory.

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeADT;
  static constexpr const char* _type_key = "runtime.ADT";
  TVM_DECLARE_FINAL_OBJECT_INFO(ADTObj, Object);

 private:
  /*!
   * \return The number of elements in the array.
   */
  size_t GetSize() const { return size; }

  /*!
   * \brief Initialize the elements in the array.
   *
   * \tparam Iterator Iterator type of the array.
   * \param begin The begin iterator.
   * \param end The end iterator.
   */
  template <typename Iterator>
  void Init(Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    this->size = 0;
    auto it = begin;
    for (size_t i = 0; i < num_elems; ++i) {
      InplaceArrayBase::EmplaceInit(i, *it++);
      // Only increment size after the initialization succeeds
      this->size++;
    }
  }

  friend class ADT;
  friend InplaceArrayBase<ADTObj, ObjectRef>;
};

/*! \brief reference to algebraic data type objects. */
class ADT : public ObjectRef {
 public:
  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param fields The fields of the ADT object.
   * \return The constructed ADT object reference.
   */
  ADT(int32_t tag, std::vector<ObjectRef> fields) : ADT(tag, fields.begin(), fields.end()){};

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param begin The begin iterator to the start of the fields array.
   * \param end The end iterator to the end of the fields array.
   * \return The constructed ADT object reference.
   */
  template <typename Iterator>
  ADT(int32_t tag, Iterator begin, Iterator end) {
    size_t num_elems = std::distance(begin, end);
    auto ptr = make_inplace_array_object<ADTObj, ObjectRef>(num_elems);
    ptr->tag = tag;
    ptr->Init(begin, end);
    data_ = std::move(ptr);
  }

  /*!
   * \brief construct an ADT object reference.
   * \param tag The tag of the ADT object.
   * \param init The initializer list of fields.
   * \return The constructed ADT object reference.
   */
  ADT(int32_t tag, std::initializer_list<ObjectRef> init) : ADT(tag, init.begin(), init.end()){};

  /*!
   * \brief Access element at index.
   *
   * \param idx The array index
   * \return const ObjectRef
   */
  const ObjectRef& operator[](size_t idx) const { return operator->()->operator[](idx); }

  /*!
   * \brief Return the ADT tag.
   */
  int32_t tag() const { return operator->()->tag; }

  /*!
   * \brief Return the number of fields.
   */
  size_t size() const { return operator->()->size; }

  /*!
   * \brief Construct a tuple object.
   *
   * \tparam Args Type params of tuple feilds.
   * \param args Tuple fields.
   * \return ADT The tuple object reference.
   */
  template <typename... Args>
  static ADT Tuple(Args&&... args) {
    return ADT(0, std::forward<Args>(args)...);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ADT, ObjectRef, ADTObj);
};

/*! \brief An object representing string. It's POD type. */
class StringObj : public Object {
 public:
  /*! \brief The pointer to string data. */
  const char* data;

  /*! \brief The length of the string object. */
  uint64_t size;

  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeString;
  static constexpr const char* _type_key = "runtime.String";
  TVM_DECLARE_FINAL_OBJECT_INFO(StringObj, Object);

 private:
  /*! \brief String object which is moved from std::string container. */
  class FromStd;

  friend class String;
};

/*!
 * \brief Reference to string objects.
 *
 * \code
 *
 * // Example to create runtime String reference object from std::string
 * std::string s = "hello world";
 *
 * // You can create the reference from existing std::string
 * String ref{std::move(s)};
 *
 * // You can rebind the reference to another string.
 * ref = std::string{"hello world2"};
 *
 * // You can use the reference as hash map key
 * std::unordered_map<String, int32_t> m;
 * m[ref] = 1;
 *
 * // You can compare the reference object with other string objects
 * assert(ref == "hello world", true);
 *
 * // You can convert the reference to std::string again
 * string s2 = (string)ref;
 *
 * \endcode
 */
class String : public ObjectRef {
 public:
  /*!
   * \brief Construct an empty string.
   */
  String() : String(std::string()) {}
  /*!
   * \brief Construct a new String object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  String(std::string other);  // NOLINT(*)

  /*!
   * \brief Construct a new String object
   *
   * \param other a char array.
   */
  String(const char* other)  // NOLINT(*)
      : String(std::string(other)) {}

  /*!
   * \brief Change the value the reference object points to.
   *
   * \param other The value for the new String
   *
   */
  inline String& operator=(std::string other);

  /*!
   * \brief Change the value the reference object points to.
   *
   * \param other The value for the new String
   */
  inline String& operator=(const char* other);

  /*!
   * \brief Compares this String object to other
   *
   * \param other The String to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const String& other) const {
    return memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this String object to other
   *
   * \param other The string to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const std::string& other) const {
    return memncmp(data(), other.data(), size(), other.size());
  }

  /*!
   * \brief Compares this to other
   *
   * \param other The character array to compare with.
   *
   * \return zero if both char sequences compare equal. negative if this appear
   * before other, positive otherwise.
   */
  int compare(const char* other) const {
    return memncmp(data(), other, size(), std::strlen(other));
  }

  /*!
   * \brief Returns a pointer to the char array in the string.
   *
   * \return const char*
   */
  const char* c_str() const { return get()->data; }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t size() const {
    const auto* ptr = get();
    return ptr->size;
  }

  /*!
   * \brief Return the length of the string
   *
   * \return size_t string length
   */
  size_t length() const { return size(); }

  /*!
   * \brief Retun if the string is empty
   *
   * \return true if empty, false otherwise.
   */
  bool empty() const { return size() == 0; }

  /*!
   * \brief Return the data pointer
   *
   * \return const char* data pointer
   */
  const char* data() const { return get()->data; }

  /*!
   * \brief Convert String to an std::string object
   *
   * \return std::string
   */
  operator std::string() const { return std::string{get()->data, size()}; }

  // LLVM compatibility function, implemented in src/target/llvm/llvm_common.h
  /*!
   * \brief Convert String to an llvm::StringRef object
   *
   * \return llvm::StringRef
   */
  inline operator llvm::StringRef() const;

  /*!
   * \brief Check if a TVMArgValue can be converted to String, i.e. it can be std::string or String
   * \param val The value to be checked
   * \return A boolean indicating if val can be converted to String
   */
  static bool CanConvertFrom(const TVMArgValue& val) {
    return val.type_code() == kTVMStr || val.IsObjectRef<tvm::runtime::String>();
  }

  /*!
   * \brief Hash the binary bytes
   * \param data The data pointer
   * \param size The size of the bytes.
   * \return the hash value.
   */
  static size_t HashBytes(const char* data, size_t size) {
    // This function falls back to string copy with c++11 compiler and is
    // recommended to be compiled with c++14
#if TVM_USE_CXX17_STRING_VIEW_HASH
    return std::hash<std::string_view>()(std::string_view(data, size));
#elif TVM_USE_CXX14_STRING_VIEW_HASH
    return std::hash<std::experimental::string_view>()(std::experimental::string_view(data, size));
#else
    return std::hash<std::string>()(std::string(data, size));
#endif
  }

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(String, ObjectRef, StringObj);

 private:
  /*!
   * \brief Compare two char sequence
   *
   * \param lhs Pointers to the char array to compare
   * \param rhs Pointers to the char array to compare
   * \param lhs_count Length of the char array to compare
   * \param rhs_count Length of the char array to compare
   * \return int zero if both char sequences compare equal. negative if this
   * appear before other, positive otherwise.
   */
  static int memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count);

  /*!
   * \brief Concatenate two char sequences
   *
   * \param lhs Pointers to the lhs char array
   * \param lhs_size The size of the lhs char array
   * \param rhs Pointers to the rhs char array
   * \param rhs_size The size of the rhs char array
   *
   * \return The concatenated char sequence
   */
  static String Concat(const char* lhs, size_t lhs_size, const char* rhs, size_t rhs_size) {
    std::string ret(lhs, lhs_size);
    ret.append(rhs, rhs_size);
    return String(ret);
  }

  // Overload + operator
  friend String operator+(const String& lhs, const String& rhs);
  friend String operator+(const String& lhs, const std::string& rhs);
  friend String operator+(const std::string& lhs, const String& rhs);
  friend String operator+(const String& lhs, const char* rhs);
  friend String operator+(const char* lhs, const String& rhs);

  friend struct tvm::runtime::ObjectEqual;
};

/*! \brief An object representing string moved from std::string. */
class StringObj::FromStd : public StringObj {
 public:
  /*!
   * \brief Construct a new FromStd object
   *
   * \param other The moved/copied std::string object
   *
   * \note If user passes const reference, it will trigger copy. If it's rvalue,
   * it will be moved into other.
   */
  explicit FromStd(std::string other) : data_container{other} {}

 private:
  /*! \brief Container that holds the memory. */
  std::string data_container;

  friend class String;
};

inline String::String(std::string other) {
  auto ptr = make_object<StringObj::FromStd>(std::move(other));
  ptr->size = ptr->data_container.size();
  ptr->data = ptr->data_container.data();
  data_ = std::move(ptr);
}

inline String& String::operator=(std::string other) {
  String replace{std::move(other)};
  data_.swap(replace.data_);
  return *this;
}

inline String& String::operator=(const char* other) { return operator=(std::string(other)); }

inline String operator+(const String& lhs, const String& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const String& lhs, const std::string& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const std::string& lhs, const String& rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = rhs.size();
  return String::Concat(lhs.data(), lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const char* lhs, const String& rhs) {
  size_t lhs_size = std::strlen(lhs);
  size_t rhs_size = rhs.size();
  return String::Concat(lhs, lhs_size, rhs.data(), rhs_size);
}

inline String operator+(const String& lhs, const char* rhs) {
  size_t lhs_size = lhs.size();
  size_t rhs_size = std::strlen(rhs);
  return String::Concat(lhs.data(), lhs_size, rhs, rhs_size);
}

// Overload < operator
inline bool operator<(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

inline bool operator<(const String& lhs, const String& rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const String& lhs, const char* rhs) { return lhs.compare(rhs) < 0; }

inline bool operator<(const char* lhs, const String& rhs) { return rhs.compare(lhs) > 0; }

// Overload > operator
inline bool operator>(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

inline bool operator>(const String& lhs, const String& rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const String& lhs, const char* rhs) { return lhs.compare(rhs) > 0; }

inline bool operator>(const char* lhs, const String& rhs) { return rhs.compare(lhs) < 0; }

// Overload <= operator
inline bool operator<=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

inline bool operator<=(const String& lhs, const String& rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const String& lhs, const char* rhs) { return lhs.compare(rhs) <= 0; }

inline bool operator<=(const char* lhs, const String& rhs) { return rhs.compare(lhs) >= 0; }

// Overload >= operator
inline bool operator>=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) <= 0; }

inline bool operator>=(const String& lhs, const String& rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const String& lhs, const char* rhs) { return lhs.compare(rhs) >= 0; }

inline bool operator>=(const char* lhs, const String& rhs) { return rhs.compare(rhs) <= 0; }

// Overload == operator
inline bool operator==(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

inline bool operator==(const String& lhs, const String& rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const String& lhs, const char* rhs) { return lhs.compare(rhs) == 0; }

inline bool operator==(const char* lhs, const String& rhs) { return rhs.compare(lhs) == 0; }

// Overload != operator
inline bool operator!=(const String& lhs, const std::string& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const std::string& lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline bool operator!=(const String& lhs, const String& rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const String& lhs, const char* rhs) { return lhs.compare(rhs) != 0; }

inline bool operator!=(const char* lhs, const String& rhs) { return rhs.compare(lhs) != 0; }

inline std::ostream& operator<<(std::ostream& out, const String& input) {
  out.write(input.data(), input.size());
  return out;
}

inline int String::memncmp(const char* lhs, const char* rhs, size_t lhs_count, size_t rhs_count) {
  if (lhs == rhs && lhs_count == rhs_count) return 0;

  for (size_t i = 0; i < lhs_count && i < rhs_count; ++i) {
    if (lhs[i] < rhs[i]) return -1;
    if (lhs[i] > rhs[i]) return 1;
  }
  if (lhs_count < rhs_count) {
    return -1;
  } else if (lhs_count > rhs_count) {
    return 1;
  } else {
    return 0;
  }
}

inline size_t ObjectHash::operator()(const ObjectRef& a) const {
  if (const auto* str = a.as<StringObj>()) {
    return String::HashBytes(str->data, str->size);
  }
  return ObjectPtrHash()(a);
}

inline bool ObjectEqual::operator()(const ObjectRef& a, const ObjectRef& b) const {
  if (a.same_as(b)) {
    return true;
  }
  if (const auto* str_a = a.as<StringObj>()) {
    if (const auto* str_b = b.as<StringObj>()) {
      return String::memncmp(str_a->data, str_b->data, str_a->size, str_b->size) == 0;
    }
  }
  return false;
}

template <>
struct PackedFuncValueConverter<::tvm::runtime::String> {
  static String From(const TVMArgValue& val) {
    if (val.IsObjectRef<tvm::runtime::String>()) {
      return val.AsObjectRef<tvm::runtime::String>();
    } else {
      return tvm::runtime::String(val.operator std::string());
    }
  }

  static String From(const TVMRetValue& val) {
    if (val.IsObjectRef<tvm::runtime::String>()) {
      return val.AsObjectRef<tvm::runtime::String>();
    } else {
      return tvm::runtime::String(val.operator std::string());
    }
  }
};

/*! \brief Helper to represent nullptr for optional. */
struct NullOptType {};

/*!
 * \brief Optional container that to represent to a Nullable variant of T.
 * \tparam T The original ObjectRef.
 *
 * \code
 *
 *  Optional<String> opt0 = nullptr;
 *  Optional<String> opt1 = String("xyz");
 *  CHECK(opt0 == nullptr);
 *  CHECK(opt1 == "xyz");
 *
 * \endcode
 */
template <typename T>
class Optional : public ObjectRef {
 public:
  using ContainerType = typename T::ContainerType;
  static_assert(std::is_base_of<ObjectRef, T>::value, "Optional is only defined for ObjectRef.");
  // default constructors.
  Optional() = default;
  Optional(const Optional<T>&) = default;
  Optional(Optional<T>&&) = default;
  Optional<T>& operator=(const Optional<T>&) = default;
  Optional<T>& operator=(Optional<T>&&) = default;
  /*!
   * \brief Construct from an ObjectPtr
   *        whose type already matches the ContainerType.
   * \param ptr
   */
  explicit Optional(ObjectPtr<Object> ptr) : ObjectRef(ptr) {}
  /*! \brief Nullopt handling */
  Optional(NullOptType) {}  // NOLINT(*)
  // nullptr handling.
  // disallow implicit conversion as 0 can be implicitly converted to nullptr_t
  explicit Optional(std::nullptr_t) {}
  Optional<T>& operator=(std::nullptr_t) {
    data_ = nullptr;
    return *this;
  }
  // normal value handling.
  Optional(T other)  // NOLINT(*)
      : ObjectRef(std::move(other)) {}
  Optional<T>& operator=(T other) {
    ObjectRef::operator=(std::move(other));
    return *this;
  }
  // delete the int constructor
  // since Optional<Integer>(0) is ambiguious
  // 0 can be implicitly casted to nullptr_t
  explicit Optional(int val) = delete;
  Optional<T>& operator=(int val) = delete;
  /*!
   * \return A not-null container value in the optional.
   * \note This function performs not-null checking.
   */
  T value() const {
    CHECK(data_ != nullptr);
    return T(data_);
  }
  /*!
   * \return The contained value if the Optional is not null
   *         otherwise return the default_value.
   */
  T value_or(T default_value) const { return data_ != nullptr ? T(data_) : default_value; }

  /*! \return Whether the container is not nullptr.*/
  explicit operator bool() const { return *this != nullptr; }
  // operator overloadings
  bool operator==(std::nullptr_t) const { return data_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return data_ != nullptr; }
  auto operator==(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() == other.value());
    if (same_as(other)) return RetType(true);
    if (*this != nullptr && other != nullptr) {
      return value() == other.value();
    } else {
      // one of them is nullptr.
      return RetType(false);
    }
  }
  auto operator!=(const Optional<T>& other) const {
    // support case where sub-class returns a symbolic ref type.
    using RetType = decltype(value() != other.value());
    if (same_as(other)) return RetType(false);
    if (*this != nullptr && other != nullptr) {
      return value() != other.value();
    } else {
      // one of them is nullptr.
      return RetType(true);
    }
  }
  auto operator==(const T& other) const {
    using RetType = decltype(value() == other);
    if (same_as(other)) return RetType(true);
    if (*this != nullptr) return value() == other;
    return RetType(false);
  }
  auto operator!=(const T& other) const { return !(*this == other); }
  template <typename U>
  auto operator==(const U& other) const {
    using RetType = decltype(value() == other);
    if (*this == nullptr) return RetType(false);
    return value() == other;
  }
  template <typename U>
  auto operator!=(const U& other) const {
    using RetType = decltype(value() != other);
    if (*this == nullptr) return RetType(true);
    return value() != other;
  }
  static constexpr bool _type_is_nullable = true;
};

template <typename T>
struct PackedFuncValueConverter<Optional<T>> {
  static Optional<T> From(const TVMArgValue& val) {
    if (val.type_code() == kTVMNullptr) return Optional<T>(nullptr);
    return PackedFuncValueConverter<T>::From(val);
  }
  static Optional<T> From(const TVMRetValue& val) {
    if (val.type_code() == kTVMNullptr) return Optional<T>(nullptr);
    return PackedFuncValueConverter<T>::From(val);
  }
};

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::Optional;
using runtime::String;
constexpr runtime::NullOptType NullOpt{};
}  // namespace tvm

namespace std {

template <>
struct hash<::tvm::runtime::String> {
  std::size_t operator()(const ::tvm::runtime::String& str) const {
    return ::tvm::runtime::String::HashBytes(str.data(), str.size());
  }
};
}  // namespace std

#endif  // TVM_RUNTIME_CONTAINER_H_
