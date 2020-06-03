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
 * \file tvm/node/container.h
 * \brief Array/Map container in the DSL graph.
 */
#ifndef TVM_NODE_CONTAINER_H_
#define TVM_NODE_CONTAINER_H_

#include <tvm/runtime/container.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>

#include <initializer_list>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {

using runtime::Array;
using runtime::ArrayNode;
using runtime::Downcast;
using runtime::IterAdapter;
using runtime::make_object;
using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectPtrEqual;
using runtime::ObjectPtrHash;
using runtime::ObjectRef;
using runtime::String;
using runtime::StringObj;

/*! \brief String-aware ObjectRef hash functor */
struct ObjectHash {
  size_t operator()(const ObjectRef& a) const {
    if (const auto* str = a.as<StringObj>()) {
      return String::HashBytes(str->data, str->size);
    }
    return ObjectPtrHash()(a);
  }
};

/*! \brief String-aware ObjectRef equal functor */
struct ObjectEqual {
  bool operator()(const ObjectRef& a, const ObjectRef& b) const {
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
};

/*! \brief map node content */
class MapNode : public Object {
 public:
  /*! \brief The corresponding conatiner type */
  using ContainerType = std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual>;

  /*! \brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "Map";
  TVM_DECLARE_FINAL_OBJECT_INFO(MapNode, Object);
};

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
  /*!
   * \brief default constructor
   */
  Map() { data_ = make_object<MapNode>(); }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V>&& other) {  // NOLINT(*)
    data_ = std::move(other.data_);
  }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V>& other) : ObjectRef(other.data_) {  // NOLINT(*)
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
    assign(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V> > init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief constructor from unordered_map
   * \param init The unordered_map
   */
  template <typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) {  // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(Map<K, V>&& other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V>& other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief reset the array to content from iterator.
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template <typename IterType>
  void assign(IterType begin, IterType end) {
    ObjectPtr<MapNode> n = make_object<MapNode>();
    for (IterType i = begin; i != end; ++i) {
      n->data.emplace(std::make_pair(i->first, i->second));
    }
    data_ = std::move(n);
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  inline const V operator[](const K& key) const {
    return DowncastNoCheck<V>(static_cast<const MapNode*>(data_.get())->data.at(key));
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  inline const V at(const K& key) const {
    return DowncastNoCheck<V>(static_cast<const MapNode*>(data_.get())->data.at(key));
  }
  /*! \return The size of the array */
  inline size_t size() const {
    if (data_.get() == nullptr) return 0;
    return static_cast<const MapNode*>(data_.get())->data.size();
  }
  /*! \return The number of elements of the key */
  inline size_t count(const K& key) const {
    if (data_.get() == nullptr) return 0;
    return static_cast<const MapNode*>(data_.get())->data.count(key);
  }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  inline MapNode* CopyOnWrite() {
    if (data_.get() == nullptr || !data_.unique()) {
      ObjectPtr<MapNode> n = make_object<MapNode>();
      n->data = static_cast<const MapNode*>(data_.get())->data;
      ObjectPtr<Object>(std::move(n)).swap(data_);
    }
    return static_cast<MapNode*>(data_.get());
  }
  /*!
   * \brief set the Map.
   * \param key The index key.
   * \param value The value to be setted.
   */
  inline void Set(const K& key, const V& value) {
    MapNode* n = this->CopyOnWrite();
    n->data[key] = value;
  }

  /*! \return whether array is empty */
  inline bool empty() const { return size() == 0; }
  /*! \brief specify container node */
  using ContainerType = MapNode;

  struct ValueConverter {
    using ResultType = std::pair<K, V>;
    static inline ResultType convert(const std::pair<ObjectRef, ObjectRef>& n) {
      return std::make_pair(DowncastNoCheck<K>(n.first), DowncastNoCheck<V>(n.second));
    }
  };

  using iterator = IterAdapter<ValueConverter, MapNode::ContainerType::const_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const MapNode*>(data_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const MapNode*>(data_.get())->data.end());
  }
  /*! \return begin iterator */
  inline iterator find(const K& key) const {
    return iterator(static_cast<const MapNode*>(data_.get())->data.find(key));
  }
};

}  // namespace tvm

namespace tvm {
namespace runtime {
// Additional overloads for PackedFunc checking.
template <typename T>
struct ObjectTypeChecker<Array<T> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<ArrayNode>()) return false;
    const ArrayNode* n = static_cast<const ArrayNode*>(ptr);
    for (const ObjectRef& p : *n) {
      if (!ObjectTypeChecker<T>::Check(p.get())) {
        return false;
      }
    }
    return true;
  }
  static std::string TypeName() { return "List[" + ObjectTypeChecker<T>::TypeName() + "]"; }
};

template <typename K, typename V>
struct ObjectTypeChecker<Map<K, V> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<MapNode>()) return false;
    const MapNode* n = static_cast<const MapNode*>(ptr);
    for (const auto& kv : n->data) {
      if (!ObjectTypeChecker<K>::Check(kv.first.get())) return false;
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static std::string TypeName() {
    return "Map[" + ObjectTypeChecker<K>::TypeName() + ", " + ObjectTypeChecker<V>::TypeName() +
           ']';
  }
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_NODE_CONTAINER_H_
