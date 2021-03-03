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

#ifndef USE_FALLBACK_STL_MAP
#define USE_FALLBACK_STL_MAP 0
#endif

#include <dmlc/logging.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
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

// Forward declare TVMArgValue
class TVMArgValue;

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
    ICHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
    return *(reinterpret_cast<ElemType*>(AddressOf(idx)));
  }

  /*!
   * \brief Access element at index
   * \param idx The index of the element.
   * \return Reference to ElemType at the index.
   */
  ElemType& operator[](size_t idx) {
    size_t size = Self()->GetSize();
    ICHECK_LT(idx, size) << "Index " << idx << " out of bounds " << size << "\n";
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

  IterAdapter operator-(difference_type offset) const { return IterAdapter(iter_ - offset); }

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
   * \brief Read an element.
   * \param pos The position at which to read the character.
   *
   * \return The char at position
   */
  char at(size_t pos) const {
    if (pos < size()) {
      return data()[pos];
    } else {
      throw std::out_of_range("tvm::String index out of bounds");
    }
  }

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
  inline static bool CanConvertFrom(const TVMArgValue& val);

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
 *  ICHECK(opt0 == nullptr);
 *  ICHECK(opt1 == "xyz");
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
    ICHECK(data_ != nullptr);
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

/*!
 * \brief An object representing a closure. This object is used by both the
 * Relay VM and interpreter.
 */
class ClosureObj : public Object {
 public:
  static constexpr const uint32_t _type_index = TypeIndex::kRuntimeClosure;
  static constexpr const char* _type_key = "runtime.Closure";
  TVM_DECLARE_BASE_OBJECT_INFO(ClosureObj, Object);
};

/*! \brief reference to closure. */
class Closure : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(Closure, ObjectRef, ClosureObj);
};

#if (USE_FALLBACK_STL_MAP != 0)

/*! \brief Shared content of all specializations of hash map */
class MapNode : public Object {
 public:
  /*! \brief Type of the keys in the hash map */
  using key_type = ObjectRef;
  /*! \brief Type of the values in the hash map */
  using mapped_type = ObjectRef;
  /*! \brief Type of the actual underlying container */
  using ContainerType = std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual>;
  /*! \brief Iterator class */
  using iterator = ContainerType::iterator;
  /*! \brief Iterator class */
  using const_iterator = ContainerType::const_iterator;
  /*! \brief Type of value stored in the hash map */
  using KVType = ContainerType::value_type;

  static_assert(std::is_standard_layout<KVType>::value, "KVType is not standard layout");
  static_assert(sizeof(KVType) == 16 || sizeof(KVType) == 8, "sizeof(KVType) incorrect");

  static constexpr const uint32_t _type_index = runtime::TypeIndex::kRuntimeMap;
  static constexpr const char* _type_key = "Map";
  TVM_DECLARE_FINAL_OBJECT_INFO(MapNode, Object);

  /*!
   * \brief Number of elements in the SmallMapNode
   * \return The result
   */
  size_t size() const { return data_.size(); }
  /*!
   * \brief Count the number of times a key exists in the hash map
   * \param key The indexing key
   * \return The result, 0 or 1
   */
  size_t count(const key_type& key) const { return data_.count(key); }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
  const mapped_type& at(const key_type& key) const { return data_.at(key); }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key) { return data_.at(key); }
  /*! \return begin iterator */
  iterator begin() { return data_.begin(); }
  /*! \return const begin iterator */
  const_iterator begin() const { return data_.begin(); }
  /*! \return end iterator */
  iterator end() { return data_.end(); }
  /*! \return end iterator */
  const_iterator end() const { return data_.end(); }
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  const_iterator find(const key_type& key) const { return data_.find(key); }
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) { return data_.find(key); }
  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position) { data_.erase(position); }
  /*!
   * \brief Erase the entry associated with the key, do nothing if not exists
   * \param key The indexing key
   */
  void erase(const key_type& key) { data_.erase(key); }
  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static ObjectPtr<MapNode> Empty() { return make_object<MapNode>(); }

 protected:
  /*!
   * \brief Create the map using contents from the given iterators.
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return ObjectPtr to the map created
   */
  template <typename IterType>
  static ObjectPtr<Object> CreateFromRange(IterType first, IterType last) {
    ObjectPtr<MapNode> p = make_object<MapNode>();
    p->data_ = ContainerType(first, last);
    return p;
  }
  /*!
   * \brief InsertMaybeReHash an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static void InsertMaybeReHash(const KVType& kv, ObjectPtr<Object>* map) {
    MapNode* map_node = static_cast<MapNode*>(map->get());
    map_node->data_[kv.first] = kv.second;
  }
  /*!
   * \brief Create an empty container with elements copying from another MapNode
   * \param from The source container
   * \return The object created
   */
  static ObjectPtr<MapNode> CopyFrom(MapNode* from) {
    ObjectPtr<MapNode> p = make_object<MapNode>();
    p->data_ = ContainerType(from->data_.begin(), from->data_.end());
    return p;
  }
  /*! \brief The real container storing data */
  ContainerType data_;
  template <typename, typename, typename, typename>
  friend class Map;
};

#else

/*! \brief Shared content of all specializations of hash map */
class MapNode : public Object {
 public:
  /*! \brief Type of the keys in the hash map */
  using key_type = ObjectRef;
  /*! \brief Type of the values in the hash map */
  using mapped_type = ObjectRef;
  /*! \brief Type of value stored in the hash map */
  using KVType = std::pair<ObjectRef, ObjectRef>;
  /*! \brief Iterator class */
  class iterator;

  static_assert(std::is_standard_layout<KVType>::value, "KVType is not standard layout");
  static_assert(sizeof(KVType) == 16 || sizeof(KVType) == 8, "sizeof(KVType) incorrect");

  static constexpr const uint32_t _type_index = runtime::TypeIndex::kRuntimeMap;
  static constexpr const char* _type_key = "Map";
  TVM_DECLARE_FINAL_OBJECT_INFO(MapNode, Object);

  /*!
   * \brief Number of elements in the SmallMapNode
   * \return The result
   */
  size_t size() const { return size_; }
  /*!
   * \brief Count the number of times a key exists in the hash map
   * \param key The indexing key
   * \return The result, 0 or 1
   */
  size_t count(const key_type& key) const;
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
  const mapped_type& at(const key_type& key) const;
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key);
  /*! \return begin iterator */
  iterator begin() const;
  /*! \return end iterator */
  iterator end() const;
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) const;
  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position);
  /*!
   * \brief Erase the entry associated with the key, do nothing if not exists
   * \param key The indexing key
   */
  void erase(const key_type& key) { erase(find(key)); }

  class iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = int64_t;
    using value_type = KVType;
    using pointer = KVType*;
    using reference = KVType&;
    /*! \brief Default constructor */
    iterator() : index(0), self(nullptr) {}
    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const {
      return index == other.index && self == other.self;
    }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return !(*this == other); }
    /*! \brief De-reference iterators */
    pointer operator->() const;
    /*! \brief De-reference iterators */
    reference operator*() const { return *((*this).operator->()); }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++();
    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--();
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      iterator copy = *this;
      --(*this);
      return copy;
    }

   protected:
    /*! \brief Construct by value */
    iterator(uint64_t index, const MapNode* self) : index(index), self(self) {}
    /*! \brief The position on the array */
    uint64_t index;
    /*! \brief The container it points to */
    const MapNode* self;

    friend class DenseMapNode;
    friend class SmallMapNode;
  };
  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static inline ObjectPtr<MapNode> Empty();

 protected:
  /*!
   * \brief Create the map using contents from the given iterators.
   * \param first Begin of iterator
   * \param last End of iterator
   * \tparam IterType The type of iterator
   * \return ObjectPtr to the map created
   */
  template <typename IterType>
  static inline ObjectPtr<Object> CreateFromRange(IterType first, IterType last);
  /*!
   * \brief InsertMaybeReHash an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static inline void InsertMaybeReHash(const KVType& kv, ObjectPtr<Object>* map);
  /*!
   * \brief Create an empty container with elements copying from another SmallMapNode
   * \param from The source container
   * \return The object created
   */
  static inline ObjectPtr<MapNode> CopyFrom(MapNode* from);
  /*! \brief number of slots minus 1 */
  uint64_t slots_;
  /*! \brief number of entries in the container */
  uint64_t size_;
  // Reference class
  template <typename, typename, typename, typename>
  friend class Map;
};

/*! \brief A specialization of small-sized hash map */
class SmallMapNode : public MapNode,
                     public runtime::InplaceArrayBase<SmallMapNode, MapNode::KVType> {
 private:
  static constexpr uint64_t kInitSize = 2;
  static constexpr uint64_t kMaxSize = 4;

 public:
  using MapNode::iterator;
  using MapNode::KVType;

  /*! \brief Defaults to the destructor of InplaceArrayBase */
  ~SmallMapNode() = default;
  /*!
   * \brief Count the number of times a key exists in the SmallMapNode
   * \param key The indexing key
   * \return The result, 0 or 1
   */
  size_t count(const key_type& key) const { return find(key).index < size_; }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
  const mapped_type& at(const key_type& key) const {
    iterator itr = find(key);
    ICHECK(itr.index < size_) << "IndexError: key is not in Map";
    return itr->second;
  }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key) {
    iterator itr = find(key);
    ICHECK(itr.index < size_) << "IndexError: key is not in Map";
    return itr->second;
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(0, this); }
  /*! \return end iterator */
  iterator end() const { return iterator(size_, this); }
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) const {
    KVType* ptr = static_cast<KVType*>(AddressOf(0));
    for (uint64_t i = 0; i < size_; ++i, ++ptr) {
      if (ObjectEqual()(ptr->first, key)) {
        return iterator(i, this);
      }
    }
    return iterator(size_, this);
  }
  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position) { Erase(position.index); }

 private:
  /*!
   * \brief Remove a position in SmallMapNode
   * \param index The position to be removed
   */
  void Erase(const uint64_t index) {
    if (index >= size_) {
      return;
    }
    KVType* begin = static_cast<KVType*>(AddressOf(0));
    KVType* last = begin + (size_ - 1);
    if (index + 1 == size_) {
      last->first.ObjectRef::~ObjectRef();
      last->second.ObjectRef::~ObjectRef();
    } else {
      *(begin + index) = std::move(*last);
    }
    size_ -= 1;
  }
  /*!
   * \brief Create an empty container
   * \param n Number of empty slots
   * \return The object created
   */
  static ObjectPtr<SmallMapNode> Empty(uint64_t n = kInitSize) {
    using ::tvm::runtime::make_inplace_array_object;
    ObjectPtr<SmallMapNode> p = make_inplace_array_object<SmallMapNode, KVType>(n);
    p->size_ = 0;
    p->slots_ = n;
    return p;
  }
  /*!
   * \brief Create an empty container initialized with a given range
   * \param n Number of empty slots
   * \param first begin of iterator
   * \param last end of iterator
   * \tparam IterType The type of iterator
   * \return The object created
   */
  template <typename IterType>
  static ObjectPtr<SmallMapNode> CreateFromRange(uint64_t n, IterType first, IterType last) {
    ObjectPtr<SmallMapNode> p = Empty(n);
    KVType* ptr = static_cast<KVType*>(p->AddressOf(0));
    for (; first != last; ++first, ++p->size_) {
      new (ptr++) KVType(*first);
    }
    return p;
  }
  /*!
   * \brief Create an empty container with elements copying from another SmallMapNode
   * \param from The source container
   * \return The object created
   */
  static ObjectPtr<SmallMapNode> CopyFrom(SmallMapNode* from) {
    KVType* first = static_cast<KVType*>(from->AddressOf(0));
    KVType* last = first + from->size_;
    return CreateFromRange(from->size_, first, last);
  }
  /*!
   * \brief InsertMaybeReHash an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static void InsertMaybeReHash(const KVType& kv, ObjectPtr<Object>* map) {
    SmallMapNode* map_node = static_cast<SmallMapNode*>(map->get());
    iterator itr = map_node->find(kv.first);
    if (itr.index < map_node->size_) {
      itr->second = kv.second;
      return;
    }
    if (map_node->size_ < map_node->slots_) {
      KVType* ptr = static_cast<KVType*>(map_node->AddressOf(map_node->size_));
      new (ptr) KVType(kv);
      ++map_node->size_;
      return;
    }
    uint64_t next_size = std::max(map_node->slots_ * 2, uint64_t(kInitSize));
    next_size = std::min(next_size, uint64_t(kMaxSize));
    ICHECK_GT(next_size, map_node->slots_);
    ObjectPtr<Object> new_map = CreateFromRange(next_size, map_node->begin(), map_node->end());
    InsertMaybeReHash(kv, &new_map);
    *map = std::move(new_map);
  }
  /*!
   * \brief Increment the pointer
   * \param index The pointer to be incremented
   * \return The increased pointer
   */
  uint64_t IncItr(uint64_t index) const { return index + 1 < size_ ? index + 1 : size_; }
  /*!
   * \brief Decrement the pointer
   * \param index The pointer to be decremented
   * \return The decreased pointer
   */
  uint64_t DecItr(uint64_t index) const { return index > 0 ? index - 1 : size_; }
  /*!
   * \brief De-reference the pointer
   * \param index The pointer to be dereferenced
   * \return The result
   */
  KVType* DeRefItr(uint64_t index) const { return static_cast<KVType*>(AddressOf(index)); }
  /*! \brief A size function used by InplaceArrayBase */
  uint64_t GetSize() const { return size_; }

 protected:
  friend class MapNode;
  friend class DenseMapNode;
  friend class runtime::InplaceArrayBase<SmallMapNode, MapNode::KVType>;
};

/*! \brief A specialization of hash map that implements the idea of array-based hash map.
 * Another reference implementation can be found [1].
 *
 * A. Overview
 *
 * DenseMapNode did several improvements over traditional separate chaining hash,
 * in terms of cache locality, memory footprints and data organization.
 *
 * A1. Implicit linked list. For better cache locality, instead of using linked list
 * explicitly for each bucket, we store list data into a single array that spans contiguously
 * in memory, and then carefully design access patterns to make sure most of them fall into
 * a single cache line.
 *
 * A2. 1-byte metadata. There is only 1 byte overhead for each slot in the array to indexing and
 * traversal. This can be divided in 3 parts.
 * 1) Reserved code: (0b11111111)_2 indicates a slot is empty; (0b11111110)_2 indicates protected,
 * which means the slot is empty but not allowed to be written.
 * 2) If not empty or protected, the highest bit is used to indicate whether data in the slot is
 * head of a linked list.
 * 3) The rest 7 bits are used as the "next pointer" (i.e. pointer to the next element). On 64-bit
 * architecture, an ordinary pointer can take up to 8 bytes, which is not acceptable overhead when
 * dealing with 16-byte ObjectRef pairs. Based on a commonly noticed fact that the lists are
 * relatively short (length <= 3) in hash maps, we follow [1]'s idea that only allows the pointer to
 * be one of the 126 possible values, i.e. if the next element of i-th slot is (i + x)-th element,
 * then x must be one of the 126 pre-defined values.
 *
 * A3. Data blocking. We organize the array in the way that every 16 elements forms a data block.
 * The 16-byte metadata of those 16 elements are stored together, followed by the real data, i.e.
 * 16 key-value pairs.
 *
 * B. Implementation details
 *
 * B1. Power-of-2 table size and Fibonacci Hashing. We use power-of-two as table size to avoid
 * modulo for more efficient arithmetics. To make the hash-to-slot mapping distribute more evenly,
 * we use the Fibonacci Hashing [2] trick.
 *
 * B2. Traverse a linked list in the array.
 * 1) List head. Assume Fibonacci Hashing maps a given key to slot i, if metadata at slot i
 * indicates that it is list head, then we found the head; otherwise the list is empty. No probing
 * is done in this procedure. 2) Next element. To find the next element of a non-empty slot i, we
 * look at the last 7 bits of the metadata at slot i. If they are all zeros, then it is the end of
 * list; otherwise, we know that the next element is (i + candidates[the-last-7-bits]).
 *
 * B3. InsertMaybeReHash an element. Following B2, we first traverse the linked list to see if this
 * element is in the linked list, and if not, we put it at the end by probing the next empty
 * position in one of the 126 candidate positions. If the linked list does not even exist, but the
 * slot for list head has been occupied by another linked list, we should find this intruder another
 * place.
 *
 * B4. Quadratic probing with triangle numbers. In open address hashing, it is provable that probing
 * with triangle numbers can traverse power-of-2-sized table [3]. In our algorithm, we follow the
 * suggestion in [1] that also use triangle numbers for "next pointer" as well as sparing for list
 * head.
 *
 * [1] https://github.com/skarupke/flat_hash_map
 * [2] https://programmingpraxis.com/2018/06/19/fibonacci-hash/
 * [3] https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
 */
class DenseMapNode : public MapNode {
 private:
  /*! \brief The number of elements in a memory block */
  static constexpr int kBlockCap = 16;
  /*! \brief Maximum load factor of the hash map */
  static constexpr double kMaxLoadFactor = 0.99;
  /*! \brief Binary representation of the metadata of an empty slot */
  static constexpr uint8_t kEmptySlot = uint8_t(0b11111111);
  /*! \brief Binary representation of the metadata of a protected slot */
  static constexpr uint8_t kProtectedSlot = uint8_t(0b11111110);
  /*! \brief Number of probing choices available */
  static constexpr int kNumJumpDists = 126;
  /*! \brief Head of the implicit linked list */
  struct ListNode;
  /*! \brief POD type of a block of memory */
  struct Block {
    uint8_t bytes[kBlockCap + kBlockCap * sizeof(KVType)];
  };
  static_assert(sizeof(Block) == kBlockCap * (sizeof(KVType) + 1), "sizeof(Block) incorrect");
  static_assert(std::is_standard_layout<Block>::value, "Block is not standard layout");

 public:
  using MapNode::iterator;

  /*!
   * \brief Destroy the DenseMapNode
   */
  ~DenseMapNode() { this->Reset(); }
  /*! \return The number of elements of the key */
  size_t count(const key_type& key) const { return !Search(key).IsNone(); }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The const reference to the value
   */
  const mapped_type& at(const key_type& key) const { return At(key); }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key) { return At(key); }
  /*!
   * \brief Index value associated with a key
   * \param key The indexing key
   * \return The iterator of the entry associated with the key, end iterator if not exists
   */
  iterator find(const key_type& key) const {
    ListNode node = Search(key);
    return node.IsNone() ? end() : iterator(node.index, this);
  }
  /*!
   * \brief Erase the entry associated with the iterator
   * \param position The iterator
   */
  void erase(const iterator& position) {
    uint64_t index = position.index;
    if (position.self != nullptr && index <= this->slots_) {
      Erase(ListNode(index, this));
    }
  }
  /*! \return begin iterator */
  iterator begin() const {
    if (slots_ == 0) {
      return iterator(0, this);
    }
    for (uint64_t index = 0; index <= slots_; ++index) {
      if (!ListNode(index, this).IsEmpty()) {
        return iterator(index, this);
      }
    }
    return iterator(slots_ + 1, this);
  }
  /*! \return end iterator */
  iterator end() const { return slots_ == 0 ? iterator(0, this) : iterator(slots_ + 1, this); }

 private:
  /*!
   * \brief Search for the given key
   * \param key The key
   * \return ListNode that associated with the key
   */
  ListNode Search(const key_type& key) const {
    if (this->size_ == 0) {
      return ListNode();
    }
    for (ListNode iter = GetListHead(ObjectHash()(key)); !iter.IsNone(); iter.MoveToNext(this)) {
      if (ObjectEqual()(key, iter.Key())) {
        return iter;
      }
    }
    return ListNode();
  }
  /*!
   * \brief Search for the given key, throw exception if not exists
   * \param key The key
   * \return ListNode that associated with the key
   */
  mapped_type& At(const key_type& key) const {
    ListNode iter = Search(key);
    ICHECK(!iter.IsNone()) << "IndexError: key is not in Map";
    return iter.Val();
  }
  /*!
   * \brief Try to insert a key, or do nothing if already exists
   * \param key The indexing key
   * \param result The linked-list entry found or just constructed
   * \return A boolean, indicating if actual insertion happens
   */
  bool TryInsert(const key_type& key, ListNode* result) {
    if (slots_ == 0) {
      return false;
    }
    // required that `iter` to be the head of a linked list through which we can iterator
    ListNode iter = IndexFromHash(ObjectHash()(key));
    // `iter` can be: 1) empty; 2) body of an irrelevant list; 3) head of the relevant list
    // Case 1: empty
    if (iter.IsEmpty()) {
      iter.NewHead(KVType(key, ObjectRef(nullptr)));
      this->size_ += 1;
      *result = iter;
      return true;
    }
    // Case 2: body of an irrelevant list
    if (!iter.IsHead()) {
      // we move the elements around and construct the single-element linked list
      return IsFull() ? false : TrySpareListHead(iter, key, result);
    }
    // Case 3: head of the relevant list
    // we iterate through the linked list until the end
    // make sure `iter` is the previous element of `next`
    ListNode next = iter;
    do {
      // find equal item, do not insert
      if (ObjectEqual()(key, next.Key())) {
        *result = next;
        return true;
      }
      // make sure `iter` is the previous element of `next`
      iter = next;
    } while (next.MoveToNext(this));
    // `iter` is the tail of the linked list
    // always check capacity before insertion
    if (IsFull()) {
      return false;
    }
    // find the next empty slot
    uint8_t jump;
    if (!iter.GetNextEmpty(this, &jump, result)) {
      return false;
    }
    result->NewTail(KVType(key, ObjectRef(nullptr)));
    // link `iter` to `empty`, and move forward
    iter.SetJump(jump);
    this->size_ += 1;
    return true;
  }
  /*!
   * \brief Spare an entry to be the head of a linked list.
   * As described in B3, during insertion, it is possible that the entire linked list does not
   * exist, but the slot of its head has been occupied by other linked lists. In this case, we need
   * to spare the slot by moving away the elements to another valid empty one to make insertion
   * possible.
   * \param target The given entry to be spared
   * \param key The indexing key
   * \param result The linked-list entry constructed as the head
   * \return A boolean, if actual insertion happens
   */
  bool TrySpareListHead(ListNode target, const key_type& key, ListNode* result) {
    // `target` is not the head of the linked list
    // move the original item of `target` (if any)
    // and construct new item on the position `target`
    // To make `target` empty, we
    // 1) find `w` the previous element of `target` in the linked list
    // 2) copy the linked list starting from `r = target`
    // 3) paste them after `w`
    // read from the linked list after `r`
    ListNode r = target;
    // write to the tail of `w`
    ListNode w = target.FindPrev(this);
    // after `target` is moved, we disallow writing to the slot
    bool is_first = true;
    uint8_t r_meta, jump;
    ListNode empty;
    do {
      // `jump` describes how `w` is jumped to `empty`
      // rehash if there is no empty space after `w`
      if (!w.GetNextEmpty(this, &jump, &empty)) {
        return false;
      }
      // move `r` to `empty`
      empty.NewTail(std::move(r.Data()));
      // clear the metadata of `r`
      r_meta = r.Meta();
      if (is_first) {
        is_first = false;
        r.SetProtected();
      } else {
        r.SetEmpty();
      }
      // link `w` to `empty`, and move forward
      w.SetJump(jump);
      w = empty;
      // move `r` forward as well
    } while (r.MoveToNext(this, r_meta));
    // finally we have done moving the linked list
    // fill data_ into `target`
    target.NewHead(KVType(key, ObjectRef(nullptr)));
    this->size_ += 1;
    *result = target;
    return true;
  }
  /*!
   * \brief Remove a ListNode
   * \param iter The node to be removed
   */
  void Erase(const ListNode& iter) {
    this->size_ -= 1;
    if (!iter.HasNext()) {
      // `iter` is the last
      if (!iter.IsHead()) {
        // cut the link if there is any
        iter.FindPrev(this).SetJump(0);
      }
      iter.Data().KVType::~KVType();
      iter.SetEmpty();
    } else {
      ListNode last = iter, prev = iter;
      for (last.MoveToNext(this); last.HasNext(); prev = last, last.MoveToNext(this)) {
      }
      iter.Data() = std::move(last.Data());
      last.SetEmpty();
      prev.SetJump(0);
    }
  }
  /*! \brief Clear the container to empty, release all entries and memory acquired */
  void Reset() {
    uint64_t n_blocks = CalcNumBlocks(this->slots_);
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* meta_ptr = data_[bi].bytes;
      KVType* data_ptr = reinterpret_cast<KVType*>(data_[bi].bytes + kBlockCap);
      for (int j = 0; j < kBlockCap; ++j, ++meta_ptr, ++data_ptr) {
        uint8_t& meta = *meta_ptr;
        if (meta != uint8_t(kProtectedSlot) && meta != uint8_t(kEmptySlot)) {
          meta = uint8_t(kEmptySlot);
          data_ptr->KVType::~KVType();
        }
      }
    }
    ReleaseMemory();
  }
  /*! \brief Release the memory acquired by the container without deleting its entries stored inside
   */
  void ReleaseMemory() {
    delete[] data_;
    data_ = nullptr;
    slots_ = 0;
    size_ = 0;
    fib_shift_ = 63;
  }
  /*!
   * \brief Create an empty container
   * \param fib_shift The fib shift provided
   * \param n_slots Number of slots required, should be power-of-two
   * \return The object created
   */
  static ObjectPtr<DenseMapNode> Empty(uint32_t fib_shift, uint64_t n_slots) {
    ICHECK_GT(n_slots, uint64_t(SmallMapNode::kMaxSize));
    ObjectPtr<DenseMapNode> p = make_object<DenseMapNode>();
    uint64_t n_blocks = CalcNumBlocks(n_slots - 1);
    Block* block = p->data_ = new Block[n_blocks];
    p->slots_ = n_slots - 1;
    p->size_ = 0;
    p->fib_shift_ = fib_shift;
    for (uint64_t i = 0; i < n_blocks; ++i, ++block) {
      std::fill(block->bytes, block->bytes + kBlockCap, uint8_t(kEmptySlot));
    }
    return p;
  }
  /*!
   * \brief Create an empty container with elements copying from another DenseMapNode
   * \param from The source container
   * \return The object created
   */
  static ObjectPtr<DenseMapNode> CopyFrom(DenseMapNode* from) {
    ObjectPtr<DenseMapNode> p = make_object<DenseMapNode>();
    uint64_t n_blocks = CalcNumBlocks(from->slots_);
    p->data_ = new Block[n_blocks];
    p->slots_ = from->slots_;
    p->size_ = from->size_;
    p->fib_shift_ = from->fib_shift_;
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* meta_ptr_from = from->data_[bi].bytes;
      KVType* data_ptr_from = reinterpret_cast<KVType*>(from->data_[bi].bytes + kBlockCap);
      uint8_t* meta_ptr_to = p->data_[bi].bytes;
      KVType* data_ptr_to = reinterpret_cast<KVType*>(p->data_[bi].bytes + kBlockCap);
      for (int j = 0; j < kBlockCap;
           ++j, ++meta_ptr_from, ++data_ptr_from, ++meta_ptr_to, ++data_ptr_to) {
        uint8_t& meta = *meta_ptr_to = *meta_ptr_from;
        ICHECK(meta != kProtectedSlot);
        if (meta != uint8_t(kEmptySlot)) {
          new (data_ptr_to) KVType(*data_ptr_from);
        }
      }
    }
    return p;
  }
  /*!
   * \brief InsertMaybeReHash an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static void InsertMaybeReHash(const KVType& kv, ObjectPtr<Object>* map) {
    DenseMapNode* map_node = static_cast<DenseMapNode*>(map->get());
    ListNode iter;
    // Try to insert. If succeed, we simply return
    if (map_node->TryInsert(kv.first, &iter)) {
      iter.Val() = kv.second;
      return;
    }
    ICHECK_GT(map_node->slots_, uint64_t(SmallMapNode::kMaxSize));
    // Otherwise, start rehash
    ObjectPtr<Object> p = Empty(map_node->fib_shift_ - 1, map_node->slots_ * 2 + 2);
    // Insert the given `kv` into the new hash map
    InsertMaybeReHash(kv, &p);
    uint64_t n_blocks = CalcNumBlocks(map_node->slots_);
    // Then Insert data from the original block.
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* meta_ptr = map_node->data_[bi].bytes;
      KVType* data_ptr = reinterpret_cast<KVType*>(map_node->data_[bi].bytes + kBlockCap);
      for (int j = 0; j < kBlockCap; ++j, ++meta_ptr, ++data_ptr) {
        uint8_t& meta = *meta_ptr;
        if (meta != uint8_t(kProtectedSlot) && meta != uint8_t(kEmptySlot)) {
          meta = uint8_t(kEmptySlot);
          KVType kv = std::move(*data_ptr);
          InsertMaybeReHash(kv, &p);
        }
      }
    }
    map_node->ReleaseMemory();
    *map = p;
  }
  /*!
   * \brief Check whether the hash table is full
   * \return A boolean indicating whether hash table is full
   */
  bool IsFull() const { return size_ + 1 > (slots_ + 1) * kMaxLoadFactor; }
  /*!
   * \brief Increment the pointer
   * \param index The pointer to be incremented
   * \return The increased pointer
   */
  uint64_t IncItr(uint64_t index) const {
    for (++index; index <= slots_; ++index) {
      if (!ListNode(index, this).IsEmpty()) {
        return index;
      }
    }
    return slots_ + 1;
  }
  /*!
   * \brief Decrement the pointer
   * \param index The pointer to be decremented
   * \return The decreased pointer
   */
  uint64_t DecItr(uint64_t index) const {
    while (index != 0) {
      index -= 1;
      if (!ListNode(index, this).IsEmpty()) {
        return index;
      }
    }
    return slots_ + 1;
  }
  /*!
   * \brief De-reference the pointer
   * \param index The pointer to be dereferenced
   * \return The result
   */
  KVType* DeRefItr(uint64_t index) const { return &ListNode(index, this).Data(); }
  /*! \brief Construct from hash code */
  ListNode IndexFromHash(uint64_t hash_value) const {
    return ListNode(FibHash(hash_value, fib_shift_), this);
  }
  /*! \brief Construct from hash code if the position is head of list */
  ListNode GetListHead(uint64_t hash_value) const {
    ListNode node = IndexFromHash(hash_value);
    return node.IsHead() ? node : ListNode();
  }
  /*! \brief Construct the number of blocks in the hash table */
  static uint64_t CalcNumBlocks(uint64_t n_slots_m1) {
    uint64_t n_slots = n_slots_m1 > 0 ? n_slots_m1 + 1 : 0;
    return (n_slots + kBlockCap - 1) / kBlockCap;
  }
  /*!
   * \brief Calculate the power-of-2 table size given the lower-bound of required capacity.
   * \param cap The lower-bound of the required capacity
   * \param fib_shift The result shift for Fibonacci Hashing
   * \param n_slots The result number of slots
   */
  static void CalcTableSize(uint64_t cap, uint32_t* fib_shift, uint64_t* n_slots) {
    uint32_t shift = 64;
    uint64_t slots = 1;
    for (uint64_t c = cap; c; c >>= 1) {
      shift -= 1;
      slots <<= 1;
    }
    ICHECK_GT(slots, cap);
    if (slots < cap * 2) {
      *fib_shift = shift - 1;
      *n_slots = slots << 1;
    } else {
      *fib_shift = shift;
      *n_slots = slots;
    }
  }
  /*!
   * \brief Fibonacci Hashing, maps a hash code to an index in a power-of-2-sized table.
   * See also: https://programmingpraxis.com/2018/06/19/fibonacci-hash/.
   * \param hash_value The raw hash value
   * \param fib_shift The shift in Fibonacci Hashing
   * \return An index calculated using Fibonacci Hashing
   */
  static uint64_t FibHash(uint64_t hash_value, uint32_t fib_shift) {
    constexpr uint64_t coeff = 11400714819323198485ull;
    return (coeff * hash_value) >> fib_shift;
  }
  /*! \brief The implicit in-place linked list used to index a chain */
  struct ListNode {
    /*! \brief Construct None */
    ListNode() : index(0), block(nullptr) {}
    /*! \brief Construct from position */
    ListNode(uint64_t index, const DenseMapNode* self)
        : index(index), block(self->data_ + (index / kBlockCap)) {}
    /*! \brief Metadata on the entry */
    uint8_t& Meta() const { return *(block->bytes + index % kBlockCap); }
    /*! \brief Data on the entry */
    KVType& Data() const {
      return *(reinterpret_cast<KVType*>(block->bytes + kBlockCap +
                                         (index % kBlockCap) * sizeof(KVType)));
    }
    /*! \brief Key on the entry */
    key_type& Key() const { return Data().first; }
    /*! \brief Value on the entry */
    mapped_type& Val() const { return Data().second; }
    /*! \brief If the entry is head of linked list */
    bool IsHead() const { return (Meta() & 0b10000000) == 0b00000000; }
    /*! \brief If the entry is none */
    bool IsNone() const { return block == nullptr; }
    /*! \brief If the entry is empty slot */
    bool IsEmpty() const { return Meta() == uint8_t(kEmptySlot); }
    /*! \brief If the entry is protected slot */
    bool IsProtected() const { return Meta() == uint8_t(kProtectedSlot); }
    /*! \brief Set the entry to be empty */
    void SetEmpty() const { Meta() = uint8_t(kEmptySlot); }
    /*! \brief Set the entry to be protected */
    void SetProtected() const { Meta() = uint8_t(kProtectedSlot); }
    /*! \brief Set the entry's jump to its next entry */
    void SetJump(uint8_t jump) const { (Meta() &= 0b10000000) |= jump; }
    /*! \brief Construct a head of linked list in-place */
    void NewHead(KVType v) const {
      Meta() = 0b00000000;
      new (&Data()) KVType(std::move(v));
    }
    /*! \brief Construct a tail of linked list in-place */
    void NewTail(KVType v) const {
      Meta() = 0b10000000;
      new (&Data()) KVType(std::move(v));
    }
    /*! \brief If the entry has next entry on the linked list */
    bool HasNext() const { return kNextProbeLocation[Meta() & 0b01111111] != 0; }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const DenseMapNode* self, uint8_t meta) {
      uint64_t offset = kNextProbeLocation[meta & 0b01111111];
      if (offset == 0) {
        index = 0;
        block = nullptr;
        return false;
      }
      index = (index + offset) & (self->slots_);
      block = self->data_ + (index / kBlockCap);
      return true;
    }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const DenseMapNode* self) { return MoveToNext(self, Meta()); }
    /*! \brief Get the previous entry on the linked list */
    ListNode FindPrev(const DenseMapNode* self) const {
      // start from the head of the linked list, which must exist
      ListNode next = self->IndexFromHash(ObjectHash()(Key()));
      // `prev` is always the previous item of `next`
      ListNode prev = next;
      for (next.MoveToNext(self); index != next.index; prev = next, next.MoveToNext(self)) {
      }
      return prev;
    }
    /*! \brief Get the next empty jump */
    bool GetNextEmpty(const DenseMapNode* self, uint8_t* jump, ListNode* result) const {
      for (uint8_t idx = 1; idx < kNumJumpDists; ++idx) {
        ListNode candidate((index + kNextProbeLocation[idx]) & (self->slots_), self);
        if (candidate.IsEmpty()) {
          *jump = idx;
          *result = candidate;
          return true;
        }
      }
      return false;
    }
    /*! \brief Index on the real array */
    uint64_t index;
    /*! \brief Pointer to the actual block */
    Block* block;
  };

 protected:
  /*! \brief fib shift in Fibonacci Hashing */
  uint32_t fib_shift_;
  /*! \brief array of data blocks */
  Block* data_;
  /* clang-format off */
  /*! \brief Candidates of probing distance */
  TVM_DLL static constexpr uint64_t kNextProbeLocation[kNumJumpDists] {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    // Quadratic probing with triangle numbers. See also:
    // 1) https://en.wikipedia.org/wiki/Quadratic_probing
    // 2) https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
    // 3) https://github.com/skarupke/flat_hash_map
    21, 28, 36, 45, 55, 66, 78, 91, 105, 120,
    136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
    351, 378, 406, 435, 465, 496, 528, 561, 595, 630,
    666, 703, 741, 780, 820, 861, 903, 946, 990, 1035,
    1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540,
    1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
    2211, 2278, 2346, 2415, 2485, 2556, 2628,
    // larger triangle numbers
    8515, 19110, 42778, 96141, 216153,
    486591, 1092981, 2458653, 5532801, 12442566,
    27993903, 62983476, 141717030, 318844378, 717352503,
    1614057336, 3631522476, 8170957530, 18384510628, 41364789378,
    93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695,
    5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000,
    309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701,
    17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251, 457381325854679626,
    1029107982097042876, 2315492959180353330, 5209859154120846435,
  };
  /* clang-format on */
  friend class MapNode;
};

#define TVM_DISPATCH_MAP(base, var, body)     \
  {                                           \
    using TSmall = SmallMapNode*;             \
    using TDense = DenseMapNode*;             \
    uint64_t slots = base->slots_;            \
    if (slots <= SmallMapNode::kMaxSize) {    \
      TSmall var = static_cast<TSmall>(base); \
      body;                                   \
    } else {                                  \
      TDense var = static_cast<TDense>(base); \
      body;                                   \
    }                                         \
  }

#define TVM_DISPATCH_MAP_CONST(base, var, body) \
  {                                             \
    using TSmall = const SmallMapNode*;         \
    using TDense = const DenseMapNode*;         \
    uint64_t slots = base->slots_;              \
    if (slots <= SmallMapNode::kMaxSize) {      \
      TSmall var = static_cast<TSmall>(base);   \
      body;                                     \
    } else {                                    \
      TDense var = static_cast<TDense>(base);   \
      body;                                     \
    }                                           \
  }

inline MapNode::iterator::pointer MapNode::iterator::operator->() const {
  TVM_DISPATCH_MAP_CONST(self, p, { return p->DeRefItr(index); });
}

inline MapNode::iterator& MapNode::iterator::operator++() {
  TVM_DISPATCH_MAP_CONST(self, p, {
    index = p->IncItr(index);
    return *this;
  });
}

inline MapNode::iterator& MapNode::iterator::operator--() {
  TVM_DISPATCH_MAP_CONST(self, p, {
    index = p->DecItr(index);
    return *this;
  });
}

inline size_t MapNode::count(const key_type& key) const {
  TVM_DISPATCH_MAP_CONST(this, p, { return p->count(key); });
}

inline const MapNode::mapped_type& MapNode::at(const MapNode::key_type& key) const {
  TVM_DISPATCH_MAP_CONST(this, p, { return p->at(key); });
}

inline MapNode::mapped_type& MapNode::at(const MapNode::key_type& key) {
  TVM_DISPATCH_MAP(this, p, { return p->at(key); });
}

inline MapNode::iterator MapNode::begin() const {
  TVM_DISPATCH_MAP_CONST(this, p, { return p->begin(); });
}

inline MapNode::iterator MapNode::end() const {
  TVM_DISPATCH_MAP_CONST(this, p, { return p->end(); });
}

inline MapNode::iterator MapNode::find(const MapNode::key_type& key) const {
  TVM_DISPATCH_MAP_CONST(this, p, { return p->find(key); });
}

inline void MapNode::erase(const MapNode::iterator& position) {
  TVM_DISPATCH_MAP(this, p, { return p->erase(position); });
}

#undef TVM_DISPATCH_MAP
#undef TVM_DISPATCH_MAP_CONST

inline ObjectPtr<MapNode> MapNode::Empty() { return SmallMapNode::Empty(); }

inline ObjectPtr<MapNode> MapNode::CopyFrom(MapNode* from) {
  if (from->slots_ <= SmallMapNode::kMaxSize) {
    return SmallMapNode::CopyFrom(static_cast<SmallMapNode*>(from));
  } else {
    return DenseMapNode::CopyFrom(static_cast<DenseMapNode*>(from));
  }
}

template <typename IterType>
inline ObjectPtr<Object> MapNode::CreateFromRange(IterType first, IterType last) {
  int64_t _cap = std::distance(first, last);
  if (_cap < 0) {
    return SmallMapNode::Empty();
  }
  uint64_t cap = static_cast<uint64_t>(_cap);
  if (cap < SmallMapNode::kMaxSize) {
    return SmallMapNode::CreateFromRange(cap, first, last);
  }
  uint32_t fib_shift;
  uint64_t n_slots;
  DenseMapNode::CalcTableSize(cap, &fib_shift, &n_slots);
  ObjectPtr<Object> obj = DenseMapNode::Empty(fib_shift, n_slots);
  for (; first != last; ++first) {
    KVType kv(*first);
    DenseMapNode::InsertMaybeReHash(kv, &obj);
  }
  return obj;
}

inline void MapNode::InsertMaybeReHash(const KVType& kv, ObjectPtr<Object>* map) {
  constexpr uint64_t kSmallMapMaxSize = SmallMapNode::kMaxSize;
  MapNode* base = static_cast<MapNode*>(map->get());
  if (base->slots_ < kSmallMapMaxSize) {
    SmallMapNode::InsertMaybeReHash(kv, map);
  } else if (base->slots_ == kSmallMapMaxSize) {
    if (base->size_ < base->slots_) {
      SmallMapNode::InsertMaybeReHash(kv, map);
    } else {
      ObjectPtr<Object> new_map = MapNode::CreateFromRange(base->begin(), base->end());
      DenseMapNode::InsertMaybeReHash(kv, &new_map);
      *map = std::move(new_map);
    }
  } else {
    DenseMapNode::InsertMaybeReHash(kv, map);
  }
}

template <>
inline ObjectPtr<MapNode> make_object<>() = delete;

#endif

/*!
 * \brief Map container of NodeRef->NodeRef in DSL graph.
 *  Map implements copy on write semantics, which means map is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam K The key NodeRef type.
 * \tparam V The value NodeRef type.
 */
template <typename K, typename V,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, K>::value>::type,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
class Map : public ObjectRef {
 public:
  using key_type = K;
  using mapped_type = V;
  class iterator;
  /*!
   * \brief default constructor
   */
  Map() { data_ = MapNode::Empty(); }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V>&& other) { data_ = std::move(other.data_); }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V>& other) : ObjectRef(other.data_) {}
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(Map<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V>& other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Map(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  Map(IterType begin, IterType end) {
    data_ = MapNode::CreateFromRange(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V>> init) {
    data_ = MapNode::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    data_ = MapNode::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V at(const K& key) const { return DowncastNoCheck<V>(GetMapNode()->at(key)); }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    MapNode* n = GetMapNode();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapNode* n = GetMapNode();
    return n == nullptr ? 0 : GetMapNode()->count(key);
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  void Set(const K& key, const V& value) {
    CopyOnWrite();
    MapNode::InsertMaybeReHash(MapNode::KVType(key, value), &data_);
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetMapNode()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetMapNode()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetMapNode()->find(key)); }

  void erase(const K& key) { CopyOnWrite()->erase(key); }

  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  MapNode* CopyOnWrite() {
    if (data_.get() == nullptr) {
      data_ = MapNode::Empty();
    } else if (!data_.unique()) {
      data_ = MapNode::CopyFrom(GetMapNode());
    }
    return GetMapNode();
  }
  /*! \brief specify container node */
  using ContainerType = MapNode;

  /*! \brief Iterator of the hash map */
  class iterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = int64_t;
    using value_type = const std::pair<K, V>;
    using pointer = value_type*;
    using reference = value_type;

    iterator() : itr() {}

    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const { return itr == other.itr; }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return itr != other.itr; }
    /*! \brief De-reference iterators is not allowed */
    pointer operator->() const = delete;
    /*! \brief De-reference iterators */
    reference operator*() const {
      auto& kv = *itr;
      return std::make_pair(DowncastNoCheck<K>(kv.first), DowncastNoCheck<V>(kv.second));
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++() {
      ++itr;
      return *this;
    }
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      iterator copy = *this;
      ++(*this);
      return copy;
    }

   private:
    iterator(const MapNode::iterator& itr)  // NOLINT(*)
        : itr(itr) {}

    template <typename, typename, typename, typename>
    friend class Map;

    MapNode::iterator itr;
  };

 private:
  /*! \brief Return data_ as type of pointer of MapNode */
  MapNode* GetMapNode() const { return static_cast<MapNode*>(data_.get()); }
};

/*!
 * \brief Merge two Maps.
 * \param lhs the first Map to merge.
 * \param rhs the second Map to merge.
 * @return The merged Array. Original Maps are kept unchanged.
 */
template <typename K, typename V,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, K>::value>::type,
          typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
inline Map<K, V> Merge(Map<K, V> lhs, const Map<K, V>& rhs) {
  for (const auto& p : rhs) {
    lhs.Set(p.first, p.second);
  }
  return std::move(lhs);
}

}  // namespace runtime

// expose the functions to the root namespace.
using runtime::Array;
using runtime::ArrayNode;
using runtime::Downcast;
using runtime::IterAdapter;
using runtime::make_object;
using runtime::Map;
using runtime::MapNode;
using runtime::Object;
using runtime::ObjectEqual;
using runtime::ObjectHash;
using runtime::ObjectPtr;
using runtime::ObjectPtrEqual;
using runtime::ObjectPtrHash;
using runtime::ObjectRef;
using runtime::Optional;
using runtime::String;
using runtime::StringObj;
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
