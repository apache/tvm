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

#include <tvm/runtime/object.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/container.h>

#include <type_traits>
#include <vector>
#include <initializer_list>
#include <unordered_map>
#include <utility>
#include <string>

namespace tvm {

using runtime::String;
using runtime::StringObj;
using runtime::Object;
using runtime::ObjectPtr;
using runtime::ObjectRef;
using runtime::make_object;
using runtime::ObjectHash;
using runtime::ObjectEqual;

/*! \brief array node content in array */
class ArrayNode : public Object {
 public:
  /*! \brief the data content */
  std::vector<ObjectRef> data;

  static constexpr const char* _type_key = "Array";
  TVM_DECLARE_FINAL_OBJECT_INFO(ArrayNode, Object);
};

/*! \brief map node content */
class MapNode : public Object {
 public:
  /*! \brief The corresponding conatiner type */
  using ContainerType = std::unordered_map<
    ObjectRef,
    ObjectRef,
    ObjectHash, ObjectEqual>;

  /*! \brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "Map";
  TVM_DECLARE_FINAL_OBJECT_INFO(MapNode, Object);
};


/*! \brief specialized map node with string as key */
class StrMapNode : public Object {
 public:
  /*! \brief The corresponding conatiner type */
  using ContainerType = std::unordered_map<std::string, ObjectRef>;

  /*! \brief the data content */
  ContainerType data;

  static constexpr const char* _type_key = "StrMap";
  TVM_DECLARE_FINAL_OBJECT_INFO(StrMapNode, Object);
};

/*!
 * \brief iterator adapter that adapts TIter to return another type.
 * \tparam Converter a struct that contains converting function
 * \tparam TIter the content iterator type.
 */
template<typename Converter,
         typename TIter>
class IterAdapter {
 public:
  using difference_type = typename std::iterator_traits<TIter>::difference_type;
  using value_type = typename Converter::ResultType;
  using pointer = typename Converter::ResultType*;
  using reference = typename Converter::ResultType&;   // NOLINT(*)
  using iterator_category = typename std::iterator_traits<TIter>::iterator_category;

  explicit IterAdapter(TIter iter) : iter_(iter) {}
  inline IterAdapter& operator++() {
    ++iter_;
    return *this;
  }
  inline IterAdapter operator+(difference_type offset) const {
    return IterAdapter(iter_ + offset);
  }

  template<typename T = IterAdapter>
  typename std::enable_if<std::is_same<iterator_category, std::random_access_iterator_tag>::value,
                          typename T::difference_type>::type
  inline operator-(const IterAdapter& rhs) const {
    return iter_ - rhs.iter_;
  }

  inline bool operator==(IterAdapter other) const {
    return iter_ == other.iter_;
  }
  inline bool operator!=(IterAdapter other) const {
    return !(*this == other);
  }
  inline const value_type operator*() const {
    return Converter::convert(*iter_);
  }

 private:
  TIter iter_;
};

/*!
 * \brief Array container of NodeRef in DSL graph.
 *  Array implements copy on write semantics, which means array is mutable
 *  but copy will happen when array is referenced in more than two places.
 *
 * operator[] only provide const acces, use Set to mutate the content.
 * \tparam T The content NodeRef type.
 */
template<typename T,
         typename = typename std::enable_if<std::is_base_of<ObjectRef, T>::value>::type >
class Array : public ObjectRef {
 public:
  /*!
   * \brief default constructor
   */
  Array() {
    data_ = make_object<ArrayNode>();
  }
  /*!
   * \brief move constructor
   * \param other source
   */
  Array(Array<T> && other) : ObjectRef() {  // NOLINT(*)
    data_ = std::move(other.data_);
  }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Array(const Array<T> &other) : ObjectRef() { // NOLINT(*)
    data_ = std::move(other.data_);
  }
  /*!
   * \brief constructor from pointer
   * \param n the container pointer
   */
  explicit Array(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*!
   * \brief constructor from iterator
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  Array(IterType begin, IterType end) {
    assign(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Array(std::initializer_list<T> init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  Array(const std::vector<T>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief Constructs a container with n elements. Each element is a copy of val
   * \param n The size of the container
   * \param val The init value
   */
  explicit Array(size_t n, const T& val) {
    auto tmp_node = make_object<ArrayNode>();
    for (size_t i = 0; i < n; ++i) {
      tmp_node->data.push_back(val);
    }
    data_ = std::move(tmp_node);
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(Array<T> && other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Array<T>& operator=(const Array<T> & other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief reset the array to content from iterator.
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = make_object<ArrayNode>();
    for (IterType it = begin; it != end; ++it) {
      n->data.push_back(T(*it));
    }
    data_ = std::move(n);
  }
  /*!
   * \brief Read i-th element from array.
   * \param i The index
   * \return the i-th element.
   */
  inline const T operator[](size_t i) const {
    return DowncastNoCheck<T>(
        static_cast<const ArrayNode*>(data_.get())->data[i]);
  }
  /*! \return The size of the array */
  inline size_t size() const {
    if (data_.get() == nullptr) return 0;
    return static_cast<const ArrayNode*>(data_.get())->data.size();
  }
  /*!
   * \brief copy on write semantics
   *  Do nothing if current handle is the unique copy of the array.
   *  Otherwise make a new copy of the array to ensure the current handle
   *  hold a unique copy.
   *
   * \return Handle to the internal node container(which ganrantees to be unique)
   */
  inline ArrayNode* CopyOnWrite() {
    if (data_.get() == nullptr || !data_.unique())  {
      ObjectPtr<ArrayNode> n = make_object<ArrayNode>();
      n->data = static_cast<ArrayNode*>(data_.get())->data;
      ObjectPtr<Object>(std::move(n)).swap(data_);
    }
    return static_cast<ArrayNode*>(data_.get());
  }
  /*!
   * \brief push a new item to the back of the list
   * \param item The item to be pushed.
   */
  inline void push_back(const T& item) {
    ArrayNode* n = this->CopyOnWrite();
    n->data.push_back(item);
  }
  /*!
   * \brief Resize the array.
   * \param size The new size.
   */
  inline void resize(size_t size) {
    ArrayNode* n = this->CopyOnWrite();
    n->data.resize(size);
  }
  /*!
   * \brief set i-th element of the array.
   * \param i The index
   * \param value The value to be setted.
   */
  inline void Set(size_t i, const T& value) {
    ArrayNode* n = this->CopyOnWrite();
    n->data[i] = value;
  }
  /*! \return whether array is empty */
  inline bool empty() const {
    return size() == 0;
  }
  /*!
   * \brief Helper function to apply fmutate to mutate an array.
   * \param fmutate The transformation function T -> T.
   * \tparam F the type of the mutation function.
   * \note This function performs copy on write optimization.
   */
  template<typename F>
  inline void MutateByApply(F fmutate) {
    ArrayNode* ptr = static_cast<ArrayNode*>(data_.get());
    if (ptr == nullptr) return;
    if (data_.unique()) {
      // Copy on write optimization.
      // Perform inplace update because this is an unique copy.
      for (size_t i = 0; i < ptr->data.size(); ++i) {
        // It is important to use move here
        // to make prevent the element's ref count from increasing
        // so fmutate itself can perform copy-on-write optimization
        T old_elem = DowncastNoCheck<T>(std::move(ptr->data[i]));
        T new_elem = fmutate(std::move(old_elem));
        ptr->data[i] = std::move(new_elem);
      }
    } else {
      // lazily trigger copy if there is element change.
      ObjectPtr<ArrayNode> copy;
      for (size_t i = 0; i < ptr->data.size(); ++i) {
        T old_elem = DowncastNoCheck<T>(ptr->data[i]);
        T new_elem = fmutate(old_elem);
        if (!new_elem.same_as(ptr->data[i])) {
          // copy the old array
          if (copy == nullptr) {
            copy = runtime::make_object<ArrayNode>(*ptr);
          }
          copy->data[i] = std::move(new_elem);
        }
      }
      // replace the data with the new copy.
      if (copy != nullptr) {
        data_ = std::move(copy);
      }
    }
  }

  /*! \brief specify container node */
  using ContainerType = ArrayNode;

  struct ValueConverter {
    using ResultType = T;
    static inline T convert(const ObjectRef& n) {
      return DowncastNoCheck<T>(n);
    }
  };
  using iterator = IterAdapter<ValueConverter,
                               std::vector<ObjectRef>::const_iterator>;

  using reverse_iterator = IterAdapter<
    ValueConverter,
    std::vector<ObjectRef>::const_reverse_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const ArrayNode*>(data_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const ArrayNode*>(data_.get())->data.end());
  }
  /*! \return rbegin iterator */
  inline reverse_iterator rbegin() const {
    return reverse_iterator(static_cast<const ArrayNode*>(data_.get())->data.rbegin());
  }
  /*! \return rend iterator */
  inline reverse_iterator rend() const {
    return reverse_iterator(static_cast<const ArrayNode*>(data_.get())->data.rend());
  }
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
template<typename K,
         typename V,
         typename = typename std::enable_if<
           std::is_base_of<ObjectRef, K>::value ||
           std::is_base_of<std::string, K>::value >::type,
         typename = typename std::enable_if<std::is_base_of<ObjectRef, V>::value>::type>
class Map : public ObjectRef {
 public:
  /*!
   * \brief default constructor
   */
  Map() {
    data_ = make_object<MapNode>();
  }
  /*!
   * \brief move constructor
   * \param other source
   */
  Map(Map<K, V> && other) {  // NOLINT(*)
    data_ = std::move(other.data_);
  }
  /*!
   * \brief copy constructor
   * \param other source
   */
  Map(const Map<K, V> &other) : ObjectRef(other.data_) { // NOLINT(*)
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
  template<typename IterType>
  Map(IterType begin, IterType end) {
    assign(begin, end);
  }
  /*!
   * \brief constructor from initializer list
   * \param init The initalizer list
   */
  Map(std::initializer_list<std::pair<K, V> > init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief constructor from vector
   * \param init The vector
   */
  template<typename Hash, typename Equal>
  Map(const std::unordered_map<K, V, Hash, Equal>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  /*!
   * \brief move assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(Map<K, V> && other) {
    data_ = std::move(other.data_);
    return *this;
  }
  /*!
   * \brief copy assign operator
   * \param other The source of assignment
   * \return reference to self.
   */
  Map<K, V>& operator=(const Map<K, V> & other) {
    data_ = other.data_;
    return *this;
  }
  /*!
   * \brief reset the array to content from iterator.
   * \param begin begin of iterator
   * \param end end of iterator
   * \tparam IterType The type of iterator
   */
  template<typename IterType>
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
    return DowncastNoCheck<V>(
        static_cast<const MapNode*>(data_.get())->data.at(key));
  }
  /*!
   * \brief Read element from map.
   * \param key The key
   * \return the corresonding element.
   */
  inline const V at(const K& key) const {
    return DowncastNoCheck<V>(
        static_cast<const MapNode*>(data_.get())->data.at(key));
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
    if (data_.get() == nullptr || !data_.unique())  {
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
  inline bool empty() const {
    return size() == 0;
  }
  /*! \brief specify container node */
  using ContainerType = MapNode;

  struct ValueConverter {
    using ResultType = std::pair<K, V>;
    static inline ResultType convert(const std::pair<
                                     ObjectRef,
                                     ObjectRef>& n) {
      return std::make_pair(DowncastNoCheck<K>(n.first),
                            DowncastNoCheck<V>(n.second));
    }
  };

  using iterator = IterAdapter<
    ValueConverter, MapNode::ContainerType::const_iterator>;

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
    return iterator(
        static_cast<const MapNode*>(data_.get())->data.find(key));
  }
};

// specialize of string map
template<typename V, typename T1, typename T2>
class Map<std::string, V, T1, T2> : public ObjectRef {
 public:
  // for code reuse
  Map() {
    data_ = make_object<StrMapNode>();
  }
  Map(Map<std::string, V> && other) {  // NOLINT(*)
    data_ = std::move(other.data_);
  }
  Map(const Map<std::string, V> &other) : ObjectRef(other.data_) { // NOLINT(*)
  }
  explicit Map(ObjectPtr<Object> n) : ObjectRef(n) {}
  template<typename IterType>
  Map(IterType begin, IterType end) {
    assign(begin, end);
  }
  Map(std::initializer_list<std::pair<std::string, V> > init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }

  template<typename Hash, typename Equal>
  Map(const std::unordered_map<std::string, V, Hash, Equal>& init) { // NOLINT(*)
    assign(init.begin(), init.end());
  }
  Map<std::string, V>& operator=(Map<std::string, V> && other) {
    data_ = std::move(other.data_);
    return *this;
  }
  Map<std::string, V>& operator=(const Map<std::string, V> & other) {
    data_ = other.data_;
    return *this;
  }
  template<typename IterType>
  void assign(IterType begin, IterType end) {
    auto n = make_object<StrMapNode>();
    for (IterType i = begin; i != end; ++i) {
      n->data.emplace(std::make_pair(i->first, i->second));
    }
    data_ = std::move(n);
  }
  inline const V operator[](const std::string& key) const {
    return DowncastNoCheck<V>(
        static_cast<const StrMapNode*>(data_.get())->data.at(key));
  }
  inline const V at(const std::string& key) const {
    return DowncastNoCheck<V>(
        static_cast<const StrMapNode*>(data_.get())->data.at(key));
  }
  inline size_t size() const {
    if (data_.get() == nullptr) return 0;
    return static_cast<const StrMapNode*>(data_.get())->data.size();
  }
  inline size_t count(const std::string& key) const {
    if (data_.get() == nullptr) return 0;
    return static_cast<const StrMapNode*>(data_.get())->data.count(key);
  }
  inline StrMapNode* CopyOnWrite() {
    if (data_.get() == nullptr || !data_.unique())  {
      ObjectPtr<StrMapNode> n = make_object<StrMapNode>();
      n->data = static_cast<const StrMapNode*>(data_.get())->data;
      ObjectPtr<Object>(std::move(n)).swap(data_);
    }
    return static_cast<StrMapNode*>(data_.get());
  }
  inline void Set(const std::string& key, const V& value) {
    StrMapNode* n = this->CopyOnWrite();
    n->data[key] = value;
  }
  inline bool empty() const {
    return size() == 0;
  }
  using ContainerType = StrMapNode;

  struct ValueConverter {
    using ResultType = std::pair<std::string, V>;
    static inline ResultType convert(const std::pair<
                                     std::string,
                                     ObjectRef>& n) {
      return std::make_pair(n.first, DowncastNoCheck<V>(n.second));
    }
  };

  using iterator = IterAdapter<
    ValueConverter, StrMapNode::ContainerType::const_iterator>;

  /*! \return begin iterator */
  inline iterator begin() const {
    return iterator(static_cast<const StrMapNode*>(data_.get())->data.begin());
  }
  /*! \return end iterator */
  inline iterator end() const {
    return iterator(static_cast<const StrMapNode*>(data_.get())->data.end());
  }
  /*! \return begin iterator */
  inline iterator find(const std::string& key) const {
    return iterator(static_cast<const StrMapNode*>(data_.get())->data.find(key));
  }
};
}  // namespace tvm

namespace tvm {
namespace runtime {
// Additional overloads for PackedFunc checking.
template<typename T>
struct ObjectTypeChecker<Array<T> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<ArrayNode>()) return false;
    const ArrayNode* n = static_cast<const ArrayNode*>(ptr);
    for (const auto& p : n->data) {
      if (!ObjectTypeChecker<T>::Check(p.get())) {
        return false;
      }
    }
    return true;
  }
  static std::string TypeName() {
    return "List[" + ObjectTypeChecker<T>::TypeName() + "]";
  }
};

template<typename V>
struct ObjectTypeChecker<Map<std::string, V> > {
  static bool Check(const Object* ptr) {
    if (ptr == nullptr) return true;
    if (!ptr->IsInstance<StrMapNode>()) return false;
    const StrMapNode* n = static_cast<const StrMapNode*>(ptr);
    for (const auto& kv : n->data) {
      if (!ObjectTypeChecker<V>::Check(kv.second.get())) return false;
    }
    return true;
  }
  static std::string TypeName() {
    return "Map[str, " +
        ObjectTypeChecker<V>::TypeName()+ ']';
  }
};

template<typename K, typename V>
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
    return "Map[" +
        ObjectTypeChecker<K>::TypeName() +
        ", " +
        ObjectTypeChecker<V>::TypeName()+ ']';
  }
};
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_NODE_CONTAINER_H_
