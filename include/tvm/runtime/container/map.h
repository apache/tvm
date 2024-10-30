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
 * \file tvm/runtime/container/map.h
 * \brief Runtime Map container types.
 */
#ifndef TVM_RUNTIME_CONTAINER_MAP_H_
#define TVM_RUNTIME_CONTAINER_MAP_H_

#ifndef USE_FALLBACK_STL_MAP
#define USE_FALLBACK_STL_MAP 0
#endif

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "./base.h"
#include "./optional.h"

namespace tvm {
namespace runtime {

#if TVM_DEBUG_WITH_ABI_CHANGE
#define TVM_MAP_FAIL_IF_CHANGED() \
  ICHECK(state_marker == self->state_marker) << "Concurrent modification of the Map";
#else
#define TVM_MAP_FAIL_IF_CHANGED()
#endif  // TVM_DEBUG_WITH_ABI_CHANGE

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
#if TVM_DEBUG_WITH_ABI_CHANGE
    iterator() : state_marker(0), index(0), self(nullptr) {}
#else
    iterator() : index(0), self(nullptr) {}
#endif  // TVM_DEBUG_WITH_ABI_CHANGE
    /*! \brief Compare iterators */
    bool operator==(const iterator& other) const {
      TVM_MAP_FAIL_IF_CHANGED()
      return index == other.index && self == other.self;
    }
    /*! \brief Compare iterators */
    bool operator!=(const iterator& other) const { return !(*this == other); }
    /*! \brief De-reference iterators */
    pointer operator->() const;
    /*! \brief De-reference iterators */
    reference operator*() const {
      TVM_MAP_FAIL_IF_CHANGED()
      return *((*this).operator->());
    }
    /*! \brief Prefix self increment, e.g. ++iter */
    iterator& operator++();
    /*! \brief Prefix self decrement, e.g. --iter */
    iterator& operator--();
    /*! \brief Suffix self increment */
    iterator operator++(int) {
      TVM_MAP_FAIL_IF_CHANGED()
      iterator copy = *this;
      ++(*this);
      return copy;
    }
    /*! \brief Suffix self decrement */
    iterator operator--(int) {
      TVM_MAP_FAIL_IF_CHANGED()
      iterator copy = *this;
      --(*this);
      return copy;
    }

   protected:
#if TVM_DEBUG_WITH_ABI_CHANGE
    uint64_t state_marker;
    /*! \brief Construct by value */
    iterator(uint64_t index, const MapNode* self)
        : state_marker(self->state_marker), index(index), self(self) {}

#else
    iterator(uint64_t index, const MapNode* self) : index(index), self(self) {}
#endif  // TVM_DEBUG_WITH_ABI_CHANGE
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
#if TVM_DEBUG_WITH_ABI_CHANGE
  uint64_t state_marker;
#endif  // TVM_DEBUG_WITH_ABI_CHANGE
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
    bool HasNext() const { return NextProbeLocation(Meta() & 0b01111111) != 0; }
    /*! \brief Move the entry to the next entry on the linked list */
    bool MoveToNext(const DenseMapNode* self, uint8_t meta) {
      uint64_t offset = NextProbeLocation(meta & 0b01111111);
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
        ListNode candidate((index + NextProbeLocation(idx)) & (self->slots_), self);
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
  TVM_MAP_FAIL_IF_CHANGED()
  TVM_DISPATCH_MAP_CONST(self, p, { return p->DeRefItr(index); });
}

inline MapNode::iterator& MapNode::iterator::operator++() {
  TVM_MAP_FAIL_IF_CHANGED()
  TVM_DISPATCH_MAP_CONST(self, p, {
    index = p->IncItr(index);
    return *this;
  });
}

inline MapNode::iterator& MapNode::iterator::operator--() {
  TVM_MAP_FAIL_IF_CHANGED()
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
#if TVM_DEBUG_WITH_ABI_CHANGE
  base->state_marker++;
#endif  // TVM_DEBUG_WITH_ABI_CHANGE
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
  /*! \brief Release reference to all the elements */
  void clear() {
    MapNode* n = GetMapNode();
    if (n != nullptr) {
      data_ = MapNode::Empty();
    }
  }
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
  /*! \return The value associated with the key, NullOpt if not found */
  Optional<V> Get(const K& key) const {
    MapNode::iterator iter = GetMapNode()->find(key);
    if (iter == GetMapNode()->end()) {
      return NullOptType{};
    }
    return DowncastNoCheck<V>(iter->second);
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
using runtime::Map;
using runtime::MapNode;
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTAINER_MAP_H_
