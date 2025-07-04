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
 * \file tvm/ffi/container/map.h
 * \brief Runtime Map container types.
 */
#ifndef TVM_FFI_CONTAINER_MAP_H_
#define TVM_FFI_CONTAINER_MAP_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/container_details.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ffi {

#if TVM_FFI_DEBUG_WITH_ABI_CHANGE
#define TVM_FFI_MAP_FAIL_IF_CHANGED() \
  TVM_FFI_ICHECK(state_marker == self->state_marker) << "Concurrent modification of the Map";
#else
#define TVM_FFI_MAP_FAIL_IF_CHANGED()
#endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE

/*! \brief Shared content of all specializations of hash map */
class MapObj : public Object {
 public:
  /*! \brief Type of the keys in the hash map */
  using key_type = Any;
  /*! \brief Type of the values in the hash map */
  using mapped_type = Any;
  /*! \brief Type of value stored in the hash map */
  using KVType = std::pair<Any, Any>;
  /*! \brief Type of raw storage of the key-value pair in the hash map */
  struct KVRawStorageType {
    TVMFFIAny first;
    TVMFFIAny second;
  };
  /*! \brief Iterator class */
  class iterator;

  static_assert(std::is_standard_layout<KVType>::value, "KVType is not standard layout");
  static_assert(sizeof(KVType) == 32, "sizeof(KVType) incorrect");

  static constexpr const int32_t _type_index = TypeIndex::kTVMFFIMap;
  static constexpr const char* _type_key = StaticTypeKey::kTVMFFIMap;
  static const constexpr bool _type_final = true;
  TVM_FFI_DECLARE_STATIC_OBJECT_INFO(MapObj, Object);

  /*!
   * \brief Number of elements in the SmallMapObj
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
#if TVM_FFI_DEBUG_WITH_ABI_CHANGE
    iterator() : state_marker(0), index(0), self(nullptr) {}
#else
    iterator() : index(0), self(nullptr) {}
#endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const {
      TVM_FFI_MAP_FAIL_IF_CHANGED()
      return index == other.index && self == other.self;
    }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return !(*this == other); }
    /*! \brief De-reference iterators */
    pointer operator->() const;
    /*! \brief De-reference iterators */
    reference operator*() const {
      TVM_FFI_MAP_FAIL_IF_CHANGED()
      return *((*this).operator->());
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++();
    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--();
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      TVM_FFI_MAP_FAIL_IF_CHANGED()
      iterator copy = *this;
      ++(*this);
      return copy;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      TVM_FFI_MAP_FAIL_IF_CHANGED()
      iterator copy = *this;
      --(*this);
      return copy;
    }

   protected:
#if TVM_FFI_DEBUG_WITH_ABI_CHANGE
    uint64_t state_marker;
    /*! \brief Construct by value */
    iterator(uint64_t index, const MapObj* self)
        : state_marker(self->state_marker), index(index), self(self) {}

#else
    iterator(uint64_t index, const MapObj* self) : index(index), self(self) {}
#endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
    /*! \brief The position on the array */
    uint64_t index;
    /*! \brief The container it points to */
    const MapObj* self;

    friend class DenseMapObj;
    friend class SmallMapObj;
  };
  /*!
   * \brief Create an empty container
   * \return The object created
   */
  static inline ObjectPtr<MapObj> Empty();

 protected:
#if TVM_FFI_DEBUG_WITH_ABI_CHANGE
  uint64_t state_marker;
#endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
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
  static inline void InsertMaybeReHash(KVType&& kv, ObjectPtr<Object>* map);
  /*!
   * \brief Create an empty container with elements copying from another SmallMapObj
   * \param from The source container
   * \return The object created
   */
  static inline ObjectPtr<MapObj> CopyFrom(MapObj* from);
  /*!
   * \brief data pointer to the data region of the map.
   * \note For immutable inplace small map we do not need data_,
   *       but we keep it here for future compact with mutable container.
   */
  void* data_;
  /*! \brief number of entries in the container */
  uint64_t size_;
  /*! \brief number of slots  */
  uint64_t slots_;
  /*!
   * \brief Optional data deleter when data is allocated separately
   *        and its deletion is not managed by MapObj::deleter_.
   */
  void (*data_deleter_)(void*) = nullptr;
  // Reference class
  template <typename, typename, typename>
  friend class Map;
};

/*! \brief A specialization of small-sized hash map */
class SmallMapObj : public MapObj,
                    public details::InplaceArrayBase<SmallMapObj, MapObj::KVRawStorageType> {
 private:
  static constexpr uint64_t kInitSize = 2;
  static constexpr uint64_t kMaxSize = 4;

 public:
  using MapObj::iterator;
  using MapObj::KVType;

  ~SmallMapObj() {
    KVType* begin = static_cast<KVType*>(data_);
    for (uint64_t index = 0; index < size_; ++index) {
      // call destructor to destroy the item in `begin + index`
      // Explicit call Any::~Any() to destroy the Any object
      // Favor this over ~KVType as MSVC may not support ~KVType (need the original name)
      (begin + index)->first.Any::~Any();
      (begin + index)->second.Any::~Any();
    }
    if (data_deleter_ != nullptr) {
      data_deleter_(data_);
    }
  }
  /*!
   * \brief Count the number of times a key exists in the SmallMapObj
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
    if (itr.index >= size_) {
      TVM_FFI_THROW(KeyError) << "key is not in Map";
    }
    return itr->second;
  }
  /*!
   * \brief Index value associated with a key, throw exception if the key does not exist
   * \param key The indexing key
   * \return The mutable reference to the value
   */
  mapped_type& at(const key_type& key) {
    iterator itr = find(key);
    if (itr.index >= size_) {
      TVM_FFI_THROW(KeyError) << "key is not in Map";
    }
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
    KVType* ptr = static_cast<KVType*>(data_);
    for (uint64_t i = 0; i < size_; ++i, ++ptr) {
      if (AnyEqual()(ptr->first, key)) {
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
   * \brief Remove a position in SmallMapObj
   * \param index The position to be removed
   */
  void Erase(const uint64_t index) {
    if (index >= size_) {
      return;
    }
    KVType* begin = static_cast<KVType*>(data_);
    // call destructor to destroy the item in `begin + index`
    // Explicit call Any::~Any() to destroy the Any object
    // Favor this over ~KVType as MSVC may not support ~KVType (need the original name)
    (begin + index)->first.Any::~Any();
    (begin + index)->second.Any::~Any();
    // IMPORTANT: We do direct raw memmove to bring later items to the current position
    // to preserve the order of insertion.
    // This works because direct memory copy preserves the Any's move semantics.
    if (index + 1 < size_) {
      std::memmove(reinterpret_cast<char*>(begin + index),
                   reinterpret_cast<char*>(begin + index + 1),
                   (size_ - index - 1) * sizeof(KVType));
    }
    size_ -= 1;
  }
  /*!
   * \brief Create an empty container
   * \param n Number of empty slots
   * \return The object created
   */
  static ObjectPtr<SmallMapObj> Empty(uint64_t n = kInitSize) {
    using ::tvm::ffi::make_inplace_array_object;
    ObjectPtr<SmallMapObj> p = make_inplace_array_object<SmallMapObj, KVType>(n);
    p->data_ = p->AddressOf(0);
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
  static ObjectPtr<SmallMapObj> CreateFromRange(uint64_t n, IterType first, IterType last) {
    ObjectPtr<SmallMapObj> p = Empty(n);
    KVType* ptr = static_cast<KVType*>(p->data_);
    for (; first != last; ++first, ++p->size_) {
      new (ptr++) KVType(*first);
    }
    return p;
  }
  /*!
   * \brief Create an empty container with elements copying from another SmallMapObj
   * \param from The source container
   * \return The object created
   */
  static ObjectPtr<SmallMapObj> CopyFrom(SmallMapObj* from) {
    KVType* first = static_cast<KVType*>(from->data_);
    KVType* last = first + from->size_;
    return CreateFromRange(from->size_, first, last);
  }
  /*!
   * \brief InsertMaybeReHash an entry into the given hash map
   * \param kv The entry to be inserted
   * \param map The pointer to the map, can be changed if re-hashing happens
   */
  static void InsertMaybeReHash(KVType&& kv, ObjectPtr<Object>* map) {
    SmallMapObj* map_node = static_cast<SmallMapObj*>(map->get());
    iterator itr = map_node->find(kv.first);
    if (itr.index < map_node->size_) {
      itr->second = kv.second;
      return;
    }
    if (map_node->size_ < map_node->slots_) {
      KVType* ptr = static_cast<KVType*>(map_node->data_) + map_node->size_;
      new (ptr) KVType(std::move(kv));
      ++map_node->size_;
      return;
    }
    uint64_t next_size = std::max(map_node->slots_ * 2, uint64_t(kInitSize));
    next_size = std::min(next_size, uint64_t(kMaxSize));
    TVM_FFI_ICHECK_GT(next_size, map_node->slots_);
    ObjectPtr<Object> new_map = CreateFromRange(next_size, map_node->begin(), map_node->end());
    InsertMaybeReHash(std::move(kv), &new_map);
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
  KVType* DeRefItr(uint64_t index) const { return static_cast<KVType*>(data_) + index; }
  /*! \brief A size function used by InplaceArrayBase */
  uint64_t GetSize() const { return size_; }

 protected:
  friend class MapObj;
  friend class DenseMapObj;
  friend class details::InplaceArrayBase<SmallMapObj, MapObj::KVType>;
};

/*! \brief A specialization of hash map that implements the idea of array-based hash map.
 * Another reference implementation can be found [1].
 *
 * A. Overview
 *
 * DenseMapObj did several improvements over traditional separate chaining hash,
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
class DenseMapObj : public MapObj {
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
  /*! \brief Index indicator to indicate an invalid index */
  static constexpr uint64_t kInvalidIndex = std::numeric_limits<uint64_t>::max();
  /*! \brief Head of the implicit linked list */
  struct ListNode;
  /*! \brief item type of the dense map, including a kv data and prev/next pointer */
  struct ItemType {
    KVType data;
    uint64_t prev = kInvalidIndex;
    uint64_t next = kInvalidIndex;

    explicit ItemType(KVType&& data) : data(std::move(data)) {}
    explicit ItemType(key_type key, mapped_type value) : data(key, value) {}
  };
  /*! \brief POD type of a block of memory */
  struct Block {
    uint8_t bytes[kBlockCap + kBlockCap * sizeof(ItemType)];
  };
  static_assert(sizeof(Block) == kBlockCap * (sizeof(ItemType) + 1), "sizeof(Block) incorrect");
  static_assert(std::is_standard_layout<Block>::value, "Block is not standard layout");

  /*!
   * \brief Deleter for the Block
   * \param data The pointer to the Block
   */
  static void BlockDeleter(void* data) { delete[] static_cast<Block*>(data); }

 public:
  using MapObj::iterator;

  /*!
   * \brief Destroy the DenseMapObj
   */
  ~DenseMapObj() { this->Reset(); }
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
  iterator begin() const { return iterator(iter_list_head_, this); }
  /*! \return end iterator */
  iterator end() const { return iterator(kInvalidIndex, this); }

 private:
  Block* GetBlock(size_t index) const { return static_cast<Block*>(data_) + index; }
  /*!
   * \brief Unlink the entry from iterator list
   * \param node The node to be unlinked
   * \note This function is usually used before deletion,
   *       and it does not change data content of the node.
   */
  void IterListUnlink(ListNode node) {
    // update head and tail of iterator list if needed
    if (node.Item().prev == kInvalidIndex) {
      iter_list_head_ = node.Item().next;
    } else {
      ListNode prev_node(node.Item().prev, this);
      prev_node.Item().next = node.Item().next;
    }
    if (node.Item().next == kInvalidIndex) {
      iter_list_tail_ = node.Item().prev;
    } else {
      ListNode next_node(node.Item().next, this);
      next_node.Item().prev = node.Item().prev;
    }
  }
  /*!
   * \brief Insert the entry into tail of iterator list
   * \param node The node to be inserted
   * \note this function does not change data content of the node.
   */
  void IterListPushBack(ListNode node) {
    node.Item().prev = iter_list_tail_;
    node.Item().next = kInvalidIndex;
    if (iter_list_tail_ != kInvalidIndex) {
      ListNode prev_node(iter_list_tail_, this);
      prev_node.Item().next = node.index;
    }
    if (iter_list_head_ == kInvalidIndex) {
      iter_list_head_ = node.index;
    }
    iter_list_tail_ = node.index;
  }
  /*!
   * \brief Replace node src by dst in the iter list
   * \param src The source node
   * \param dst The destination node, must be empty
   * \note This function does not change data content of the nodes,
   *       which needs to be updated by the caller.
   */
  void IterListReplaceNodeBy(ListNode src, ListNode dst) {
    // set link correctly on the dst
    dst.Item().prev = src.Item().prev;
    dst.Item().next = src.Item().next;
    // update prev and next of dst
    if (dst.Item().prev == kInvalidIndex) {
      iter_list_head_ = dst.index;
    } else {
      ListNode prev_node(dst.Item().prev, this);
      prev_node.Item().next = dst.index;
    }
    if (dst.Item().next == kInvalidIndex) {
      iter_list_tail_ = dst.index;
    } else {
      ListNode next_node(dst.Item().next, this);
      next_node.Item().prev = dst.index;
    }
  }
  /*!
   * \brief Search for the given key
   * \param key The key
   * \return ListNode that associated with the key
   */
  ListNode Search(const key_type& key) const {
    if (this->size_ == 0) {
      return ListNode();
    }
    for (ListNode iter = GetListHead(AnyHash()(key)); !iter.IsNone(); iter.MoveToNext(this)) {
      if (AnyEqual()(key, iter.Key())) {
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
    if (iter.IsNone()) {
      TVM_FFI_THROW(IndexError) << "key is not in Map";
    }
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
    ListNode iter = IndexFromHash(AnyHash()(key));
    // `iter` can be: 1) empty; 2) body of an irrelevant list; 3) head of the relevant list
    // Case 1: empty
    if (iter.IsEmpty()) {
      iter.NewHead(ItemType(key, Any(nullptr)));
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
      if (AnyEqual()(key, next.Key())) {
        // we plan to take next, so we need to unlink it from iterator list
        IterListUnlink(next);
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
    result->NewTail(ItemType(key, Any(nullptr)));
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
      // first move the data over
      empty.NewTail(ItemType(std::move(r.Data())));
      // then move link list chain of r to empty
      // this needs to happen after NewTail so empty's prev/next get updated
      IterListReplaceNodeBy(r, empty);
      // explicit call destructor to destroy the item in `r`
      r.DestructData();
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
    target.NewHead(ItemType(key, Any(nullptr)));
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
      // unlink the node from iterator list
      IterListUnlink(iter);
      // IMPORTANT: must explicit call destructor `iter` to avoid memory leak
      // This is because we need to recycle iter's data
      iter.DestructData();
      // set the meta data to be empty
      iter.SetEmpty();
    } else {
      ListNode last = iter, prev = iter;
      for (last.MoveToNext(this); last.HasNext(); prev = last, last.MoveToNext(this)) {
      }
      // needs to first unlink iter from the list
      IterListUnlink(iter);
      // move data from last to iter
      iter.Data() = std::move(last.Data());
      // Move link chain of iter to last as we stores last node to the new iter loc.
      IterListReplaceNodeBy(last, iter);
      // IMPORTANT: must explicit call destructor `last` to avoid memory leak
      // likely we don't need this in this particular case because Any move behavior
      // keep it here to be safe so code do not depend on specific move behavior of KVType
      last.DestructData();
      // set the meta data to be empty
      last.SetEmpty();
      prev.SetJump(0);
    }
  }
  /*! \brief Clear the container to empty, release all entries and memory acquired */
  void Reset() {
    uint64_t n_blocks = CalcNumBlocks(this->slots_);
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* meta_ptr = GetBlock(bi)->bytes;
      ItemType* data_ptr = reinterpret_cast<ItemType*>(GetBlock(bi)->bytes + kBlockCap);
      for (int j = 0; j < kBlockCap; ++j, ++meta_ptr, ++data_ptr) {
        uint8_t& meta = *meta_ptr;
        if (meta != uint8_t(kProtectedSlot) && meta != uint8_t(kEmptySlot)) {
          meta = uint8_t(kEmptySlot);
          data_ptr->ItemType::~ItemType();
        }
      }
    }
    ReleaseMemory();
  }
  /*! \brief Release the memory acquired by the container without deleting its entries stored inside
   */
  void ReleaseMemory() {
    if (data_ != nullptr) {
      TVM_FFI_ICHECK(data_deleter_ != nullptr);
      data_deleter_(data_);
    }
    data_ = nullptr;
    data_deleter_ = nullptr;
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
  static ObjectPtr<DenseMapObj> Empty(uint32_t fib_shift, uint64_t n_slots) {
    TVM_FFI_ICHECK_GT(n_slots, uint64_t(SmallMapObj::kMaxSize));
    ObjectPtr<DenseMapObj> p = make_object<DenseMapObj>();
    uint64_t n_blocks = CalcNumBlocks(n_slots);
    Block* block = new Block[n_blocks];
    p->data_ = block;
    // assign block deleter so even if we take re-alloc data
    // in another shared-lib that may have different malloc/free behavior
    // it will still be safe.
    p->data_deleter_ = BlockDeleter;
    p->slots_ = n_slots;
    p->size_ = 0;
    p->fib_shift_ = fib_shift;
    p->iter_list_head_ = kInvalidIndex;
    p->iter_list_tail_ = kInvalidIndex;
    for (uint64_t i = 0; i < n_blocks; ++i, ++block) {
      std::fill(block->bytes, block->bytes + kBlockCap, uint8_t(kEmptySlot));
    }
    return p;
  }
  /*!
   * \brief Create an empty container with elements copying from another DenseMapObj
   * \param from The source container
   * \return The object created
   */
  static ObjectPtr<DenseMapObj> CopyFrom(DenseMapObj* from) {
    ObjectPtr<DenseMapObj> p = make_object<DenseMapObj>();
    uint64_t n_blocks = CalcNumBlocks(from->slots_);
    p->data_ = new Block[n_blocks];
    // assign block deleter so even if we take re-alloc data
    // in another shared-lib that may have different malloc/free behavior
    // it will still be safe.
    p->data_deleter_ = BlockDeleter;
    p->slots_ = from->slots_;
    p->size_ = from->size_;
    p->fib_shift_ = from->fib_shift_;
    p->iter_list_head_ = from->iter_list_head_;
    p->iter_list_tail_ = from->iter_list_tail_;
    for (uint64_t bi = 0; bi < n_blocks; ++bi) {
      uint8_t* meta_ptr_from = from->GetBlock(bi)->bytes;
      ItemType* data_ptr_from = reinterpret_cast<ItemType*>(from->GetBlock(bi)->bytes + kBlockCap);
      uint8_t* meta_ptr_to = p->GetBlock(bi)->bytes;
      ItemType* data_ptr_to = reinterpret_cast<ItemType*>(p->GetBlock(bi)->bytes + kBlockCap);
      for (int j = 0; j < kBlockCap;
           ++j, ++meta_ptr_from, ++data_ptr_from, ++meta_ptr_to, ++data_ptr_to) {
        uint8_t& meta = *meta_ptr_to = *meta_ptr_from;
        TVM_FFI_ICHECK(meta != kProtectedSlot);
        if (meta != uint8_t(kEmptySlot)) {
          new (data_ptr_to) ItemType(*data_ptr_from);
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
  static void InsertMaybeReHash(KVType&& kv, ObjectPtr<Object>* map) {
    DenseMapObj* map_node = static_cast<DenseMapObj*>(map->get());
    ListNode iter;
    // Try to insert. If succeed, we simply return
    if (map_node->TryInsert(kv.first, &iter)) {
      iter.Val() = std::move(kv.second);
      // update the iter list relation
      map_node->IterListPushBack(iter);
      return;
    }
    TVM_FFI_ICHECK_GT(map_node->slots_, uint64_t(SmallMapObj::kMaxSize));
    // Otherwise, start rehash
    ObjectPtr<Object> p = Empty(map_node->fib_shift_ - 1, map_node->slots_ * 2);

    // need to insert in the same order as the original map
    for (uint64_t index = map_node->iter_list_head_; index != kInvalidIndex;) {
      ListNode node(index, map_node);
      // now try move src_data into the new map, note that src may still not
      // be fully consumed into the call, but destructor will be called.
      InsertMaybeReHash(std::move(node.Data()), &p);
      // Important, needs to explicit call destructor in case move did remove
      // node's internal item
      index = node.Item().next;
      // IMPORTANT: must explicit call destructor `node` to avoid memory leak
      // We must call node.DestructData() here.
      // This is because std::move() arguments in IterMaybeReHash may or may not
      // explicitly move out the node.Data()
      // Remove this call will cause memory leak very likely.
      node.DestructData();
    }
    InsertMaybeReHash(std::move(kv), &p);
    map_node->ReleaseMemory();
    *map = p;
  }
  /*!
   * \brief Check whether the hash table is full
   * \return A boolean indicating whether hash table is full
   */
  bool IsFull() const { return size_ + 1 > slots_ * kMaxLoadFactor; }
  /*!
   * \brief Increment the pointer
   * \param index The pointer to be incremented
   * \return The increased pointer
   */
  uint64_t IncItr(uint64_t index) const {
    // keep at the end of iterator
    if (index == kInvalidIndex) {
      return index;
    }
    ListNode node(index, this);
    return node.Item().next;
  }
  /*!
   * \brief Decrement the pointer
   * \param index The pointer to be decremented
   * \return The decreased pointer
   */
  uint64_t DecItr(uint64_t index) const {
    // this is the end iterator, we need to return tail.
    if (index == kInvalidIndex) {
      return iter_list_tail_;
    }
    // circle around the iterator list, which is OK
    ListNode node(index, this);
    return node.Item().prev;
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
  static uint64_t CalcNumBlocks(uint64_t n_slots) { return (n_slots + kBlockCap - 1) / kBlockCap; }
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
    TVM_FFI_ICHECK_GT(slots, cap);
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
    ListNode(uint64_t index, const DenseMapObj* self)
        : index(index), block(self->GetBlock(index / kBlockCap)) {}
    /*! \brief Metadata on the entry */
    uint8_t& Meta() const { return *(block->bytes + index % kBlockCap); }
    /*! \brief Data on the entry */
    ItemType& Item() const {
      return *(reinterpret_cast<ItemType*>(block->bytes + kBlockCap +
                                           (index % kBlockCap) * sizeof(ItemType)));
    }
    /*! \brief Data on the entry */
    KVType& Data() const { return Item().data; }
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
    /*! \brief Destruct the item in the entry */
    void DestructData() const {
      // explicit call destructor to destroy the item
      // Favor this over ~KVType as MSVC may not support ~KVType (need the original name)
      (&Data())->first.Any::~Any();
      (&Data())->second.Any::~Any();
    }
    /*! \brief Set the entry to be protected */
    void SetProtected() const { Meta() = uint8_t(kProtectedSlot); }
    /*! \brief Set the entry's jump to its next entry */
    void SetJump(uint8_t jump) const { (Meta() &= 0b10000000) |= jump; }
    /*! \brief Construct a head of linked list in-place */
    void NewHead(ItemType v) const {
      Meta() = 0b00000000;
      new (&Item()) ItemType(std::move(v));
    }
    /*! \brief Construct a tail of linked list in-place */
    void NewTail(ItemType v) const {
      Meta() = 0b10000000;
      new (&Item()) ItemType(std::move(v));
    }

    /*! \brief If the entry has next entry on the linked list */
    bool HasNext() const { return NextProbeLocation(Meta() & 0b01111111) != 0; }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const DenseMapObj* self, uint8_t meta) {
      uint64_t offset = NextProbeLocation(meta & 0b01111111);
      if (offset == 0) {
        index = 0;
        block = nullptr;
        return false;
      }
      // the probing will go to next position and round back to stay within the
      // correct range of the slots
      index = (index + offset) % self->slots_;
      block = self->GetBlock(index / kBlockCap);
      return true;
    }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const DenseMapObj* self) { return MoveToNext(self, Meta()); }
    /*! \brief Get the previous entry on the linked list */
    ListNode FindPrev(const DenseMapObj* self) const {
      // start from the head of the linked list, which must exist
      ListNode next = self->IndexFromHash(AnyHash()(Key()));
      // `prev` is always the previous item of `next`
      ListNode prev = next;
      for (next.MoveToNext(self); index != next.index; prev = next, next.MoveToNext(self)) {
      }
      return prev;
    }
    /*! \brief Get the next empty jump */
    bool GetNextEmpty(const DenseMapObj* self, uint8_t* jump, ListNode* result) const {
      for (uint8_t idx = 1; idx < kNumJumpDists; ++idx) {
        // the probing will go to next position and round back to stay within the
        // correct range of the slots
        ListNode candidate((index + NextProbeLocation(idx)) % self->slots_, self);
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
  /*! \brief the head of iterator list */
  uint64_t iter_list_head_ = kInvalidIndex;
  /*! \brief the tail of iterator list */
  uint64_t iter_list_tail_ = kInvalidIndex;

  static uint64_t NextProbeLocation(size_t index) {
    /* clang-format off */
    /*! \brief Candidates of probing distance */
    static const uint64_t kNextProbeLocation[kNumJumpDists] {
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
      17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251,
      457381325854679626, 1029107982097042876, 2315492959180353330, 5209859154120846435,
    };
    /* clang-format on */
    return kNextProbeLocation[index];
  }
  friend class MapObj;
};

#define TVM_FFI_DISPATCH_MAP(base, var, body) \
  {                                           \
    using TSmall = SmallMapObj*;              \
    using TDense = DenseMapObj*;              \
    uint64_t slots = base->slots_;            \
    if (slots <= SmallMapObj::kMaxSize) {     \
      TSmall var = static_cast<TSmall>(base); \
      body;                                   \
    } else {                                  \
      TDense var = static_cast<TDense>(base); \
      body;                                   \
    }                                         \
  }

#define TVM_FFI_DISPATCH_MAP_CONST(base, var, body) \
  {                                                 \
    using TSmall = const SmallMapObj*;              \
    using TDense = const DenseMapObj*;              \
    uint64_t slots = base->slots_;                  \
    if (slots <= SmallMapObj::kMaxSize) {           \
      TSmall var = static_cast<TSmall>(base);       \
      body;                                         \
    } else {                                        \
      TDense var = static_cast<TDense>(base);       \
      body;                                         \
    }                                               \
  }

inline MapObj::iterator::pointer MapObj::iterator::operator->() const {
  TVM_FFI_MAP_FAIL_IF_CHANGED()
  TVM_FFI_DISPATCH_MAP_CONST(self, p, { return p->DeRefItr(index); });
}

inline MapObj::iterator& MapObj::iterator::operator++() {
  TVM_FFI_MAP_FAIL_IF_CHANGED()
  TVM_FFI_DISPATCH_MAP_CONST(self, p, {
    index = p->IncItr(index);
    return *this;
  });
}

inline MapObj::iterator& MapObj::iterator::operator--() {
  TVM_FFI_MAP_FAIL_IF_CHANGED()
  TVM_FFI_DISPATCH_MAP_CONST(self, p, {
    index = p->DecItr(index);
    return *this;
  });
}

inline size_t MapObj::count(const key_type& key) const {
  TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->count(key); });
}

inline const MapObj::mapped_type& MapObj::at(const MapObj::key_type& key) const {
  TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->at(key); });
}

inline MapObj::mapped_type& MapObj::at(const MapObj::key_type& key) {
  TVM_FFI_DISPATCH_MAP(this, p, { return p->at(key); });
}

inline MapObj::iterator MapObj::begin() const {
  TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->begin(); });
}

inline MapObj::iterator MapObj::end() const {
  TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->end(); });
}

inline MapObj::iterator MapObj::find(const MapObj::key_type& key) const {
  TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->find(key); });
}

inline void MapObj::erase(const MapObj::iterator& position) {
  TVM_FFI_DISPATCH_MAP(this, p, { return p->erase(position); });
}

#undef TVM_FFI_DISPATCH_MAP
#undef TVM_FFI_DISPATCH_MAP_CONST

inline ObjectPtr<MapObj> MapObj::Empty() { return SmallMapObj::Empty(); }

inline ObjectPtr<MapObj> MapObj::CopyFrom(MapObj* from) {
  if (from->slots_ <= SmallMapObj::kMaxSize) {
    return SmallMapObj::CopyFrom(static_cast<SmallMapObj*>(from));
  } else {
    return DenseMapObj::CopyFrom(static_cast<DenseMapObj*>(from));
  }
}

template <typename IterType>
inline ObjectPtr<Object> MapObj::CreateFromRange(IterType first, IterType last) {
  int64_t _cap = std::distance(first, last);
  if (_cap < 0) {
    return SmallMapObj::Empty();
  }
  uint64_t cap = static_cast<uint64_t>(_cap);
  if (cap < SmallMapObj::kMaxSize) {
    return SmallMapObj::CreateFromRange(cap, first, last);
  }
  uint32_t fib_shift;
  uint64_t n_slots;
  DenseMapObj::CalcTableSize(cap, &fib_shift, &n_slots);
  ObjectPtr<Object> obj = DenseMapObj::Empty(fib_shift, n_slots);
  for (; first != last; ++first) {
    KVType kv(*first);
    DenseMapObj::InsertMaybeReHash(std::move(kv), &obj);
  }
  return obj;
}

inline void MapObj::InsertMaybeReHash(KVType&& kv, ObjectPtr<Object>* map) {
  constexpr uint64_t kSmallMapMaxSize = SmallMapObj::kMaxSize;
  MapObj* base = static_cast<MapObj*>(map->get());
#if TVM_FFI_DEBUG_WITH_ABI_CHANGE
  base->state_marker++;
#endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
  if (base->slots_ < kSmallMapMaxSize) {
    SmallMapObj::InsertMaybeReHash(std::move(kv), map);
  } else if (base->slots_ == kSmallMapMaxSize) {
    if (base->size_ < base->slots_) {
      SmallMapObj::InsertMaybeReHash(std::move(kv), map);
    } else {
      ObjectPtr<Object> new_map = MapObj::CreateFromRange(base->begin(), base->end());
      DenseMapObj::InsertMaybeReHash(std::move(kv), &new_map);
      *map = std::move(new_map);
    }
  } else {
    DenseMapObj::InsertMaybeReHash(std::move(kv), map);
  }
}

template <>
inline ObjectPtr<MapObj> make_object<>() = delete;

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
          typename = typename std::enable_if_t<details::storage_enabled_v<K> &&
                                               details::storage_enabled_v<V>>>
class Map : public ObjectRef {
 public:
  using key_type = K;
  using mapped_type = V;
  class iterator;
  /*!
   * \brief default constructor
   */
  Map() { data_ = MapObj::Empty(); }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V>&& other) : ObjectRef(std::move(other.data_)) {}
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V>& other) : ObjectRef(other.data_) {}

  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map(Map<KU, VU>&& other) : ObjectRef(std::move(other.data_)) {}

  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map(const Map<KU, VU>& other) : ObjectRef(other.data_) {}
  Map<K, V>& operator=(Map<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  Map<K, V>& operator=(const Map<K, V>& other) {
    data_ = other.data_;
    return *this;
  }

  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map<K, V>& operator=(Map<KU, VU>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }

  template <typename KU, typename VU,
            typename = std::enable_if_t<details::type_contains_v<K, KU> &&
                                        details::type_contains_v<V, VU>>>
  Map<K, V>& operator=(const Map<KU, VU>& other) {
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
    data_ = MapObj::CreateFromRange(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V>> init) {
    data_ = MapObj::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    data_ = MapObj::CreateFromRange(init.begin(), init.end());
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V at(const K& key) const {
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(GetMapObj()->at(key));
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  const V operator[](const K& key) const { return this->at(key); }
  /*! \return The size of the array */
  size_t size() const {
    MapObj* n = GetMapObj();
    return n == nullptr ? 0 : n->size();
  }
  /*! \return The number of elements of the key */
  size_t count(const K& key) const {
    MapObj* n = GetMapObj();
    return n == nullptr ? 0 : GetMapObj()->count(key);
  }
  /*! \return whether array is empty */
  bool empty() const { return size() == 0; }
  /*! \brief Release reference to all the elements */
  void clear() {
    MapObj* n = GetMapObj();
    if (n != nullptr) {
      data_ = MapObj::Empty();
    }
  }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  void Set(const K& key, const V& value) {
    CopyOnWrite();
    MapObj::InsertMaybeReHash(MapObj::KVType(key, value), &data_);
  }
  /*! \return begin iterator */
  iterator begin() const { return iterator(GetMapObj()->begin()); }
  /*! \return end iterator */
  iterator end() const { return iterator(GetMapObj()->end()); }
  /*! \return find the key and returns the associated iterator */
  iterator find(const K& key) const { return iterator(GetMapObj()->find(key)); }
  /*! \return The value associated with the key, std::nullopt if not found */
  std::optional<V> Get(const K& key) const {
    MapObj::iterator iter = GetMapObj()->find(key);
    if (iter == GetMapObj()->end()) {
      return std::nullopt;
    }
    return details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(iter->second);
  }
  void erase(const K& key) { CopyOnWrite()->erase(key); }

  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which guarantees to be unique)
   */
  MapObj* CopyOnWrite() {
    if (data_.get() == nullptr) {
      data_ = MapObj::Empty();
    } else if (!data_.unique()) {
      data_ = MapObj::CopyFrom(GetMapObj());
    }
    return GetMapObj();
  }
  /*! \brief specify container node */
  using ContainerType = MapObj;

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
      return std::make_pair(details::AnyUnsafe::CopyFromAnyViewAfterCheck<K>(kv.first),
                            details::AnyUnsafe::CopyFromAnyViewAfterCheck<V>(kv.second));
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

    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--() {
      --itr;
      return *this;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      iterator copy = *this;
      --(*this);
      return copy;
    }

   private:
    iterator(const MapObj::iterator& itr)  // NOLINT(*)
        : itr(itr) {}

    template <typename, typename, typename>
    friend class Map;

    MapObj::iterator itr;
  };

 private:
  /*! \brief Return data_ as type of pointer of MapObj */
  MapObj* GetMapObj() const { return static_cast<MapObj*>(data_.get()); }

  template <typename, typename, typename>
  friend class Map;
};

/*!
 * \brief Merge two Maps.
 * \param lhs the first Map to merge.
 * \param rhs the second Map to merge.
 * @return The merged Array. Original Maps are kept unchanged.
 */
template <typename K, typename V,
          typename = typename std::enable_if_t<details::storage_enabled_v<K> &&
                                               details::storage_enabled_v<V>>>
inline Map<K, V> Merge(Map<K, V> lhs, const Map<K, V>& rhs) {
  for (const auto& p : rhs) {
    lhs.Set(p.first, p.second);
  }
  return std::move(lhs);
}

// Traits for Map
template <typename K, typename V>
inline constexpr bool use_default_type_traits_v<Map<K, V>> = false;

template <typename K, typename V>
struct TypeTraits<Map<K, V>> : public ObjectRefTypeTraitsBase<Map<K, V>> {
  static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIMap;
  using ObjectRefTypeTraitsBase<Map<K, V>>::CopyFromAnyViewAfterCheck;

  TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIMap) {
      return TypeTraitsBase::GetMismatchTypeInfo(src);
    }
    if constexpr (!std::is_same_v<K, Any> || !std::is_same_v<V, Any>) {
      const MapObj* n = reinterpret_cast<const MapObj*>(src->v_obj);
      for (const auto& kv : *n) {
        if constexpr (!std::is_same_v<K, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first) &&
              !kv.first.try_cast<K>().has_value()) {
            return "Map[some key is " + details::AnyUnsafe::GetMismatchTypeInfo<K>(kv.first) +
                   ", V]";
          }
        }
        if constexpr (!std::is_same_v<V, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second) &&
              !kv.second.try_cast<V>().has_value()) {
            return "Map[K, some value is " + details::AnyUnsafe::GetMismatchTypeInfo<V>(kv.second) +
                   "]";
          }
        }
      }
    }
    TVM_FFI_THROW(InternalError) << "Cannot reach here";
    TVM_FFI_UNREACHABLE();
  }

  TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIMap) return false;
    if constexpr (std::is_same_v<K, Any> && std::is_same_v<V, Any>) {
      return true;
    } else {
      const MapObj* n = reinterpret_cast<const MapObj*>(src->v_obj);
      for (const auto& kv : *n) {
        if constexpr (!std::is_same_v<K, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first)) return false;
        }
        if constexpr (!std::is_same_v<V, Any>) {
          if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second)) return false;
        }
      }
      return true;
    }
  }

  TVM_FFI_INLINE static std::optional<Map<K, V>> TryCastFromAnyView(const TVMFFIAny* src) {
    if (src->type_index != TypeIndex::kTVMFFIMap) return std::nullopt;
    if constexpr (!std::is_same_v<K, Any> || !std::is_same_v<V, Any>) {
      const MapObj* n = reinterpret_cast<const MapObj*>(src->v_obj);
      bool storage_check = [&]() {
        for (const auto& kv : *n) {
          if constexpr (!std::is_same_v<K, Any>) {
            if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first)) return false;
          }
          if constexpr (!std::is_same_v<V, Any>) {
            if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second)) return false;
          }
        }
        return true;
      }();
      // fast path, if storage check passes, we can return the array directly.
      if (storage_check) return CopyFromAnyViewAfterCheck(src);
      // slow path, we need to create a new map and convert to the target type.
      Map<K, V> ret;
      for (const auto& kv : *n) {
        auto k = kv.first.try_cast<K>();
        auto v = kv.second.try_cast<V>();
        if (!k.has_value() || !v.has_value()) return std::nullopt;
        ret.Set(*std::move(k), *std::move(v));
      }
      return ret;
    } else {
      return CopyFromAnyViewAfterCheck(src);
    }
  }

  TVM_FFI_INLINE static std::string TypeStr() {
    return "Map<" + details::Type2Str<K>::v() + ", " + details::Type2Str<V>::v() + ">";
  }
};

namespace details {
template <typename K, typename V, typename KU, typename VU>
inline constexpr bool type_contains_v<Map<K, V>, Map<KU, VU>> =
    type_contains_v<K, KU> && type_contains_v<V, VU>;
}  // namespace details

}  // namespace ffi

// Expose to the tvm namespace
// Rationale: convinience and no ambiguity
using ffi::Map;
}  // namespace tvm
#endif  // TVM_FFI_CONTAINER_MAP_H_
