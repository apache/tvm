/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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
 * \file support/ordered_map.h
 * \brief An STL-like map that preserves insertion order.
 */
#ifndef TVM_SUPPORT_ORDERED_MAP_H_
#define TVM_SUPPORT_ORDERED_MAP_H_

#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace support {

/**
 * \brief An STL-like map that preserves insertion order.
 *
 * \tparam K The key type.
 * \tparam V The value type.
 * \tparam Hash The hash function.
 * \tparam KeyEqual The key equality function.
 * \note we don't support erase since it is less needed and vector backing is more efficient.
 */
template <typename K, typename V, typename Hash = std::hash<K>,
          typename KeyEqual = std::equal_to<K>>
class OrderedMap {
 public:
  OrderedMap() = default;

  /* \brief Explicit copy constructor
   *
   * The default copy constructor would copy both `elements_` and
   * `elem_to_iter_`.  While this is the correct behavior for
   * `elements_`, the copy of `elem_to_iter_` would contain references
   * to the original's `element_`, rather than to its own
   */
  OrderedMap(const OrderedMap<K, V, Hash, KeyEqual>& other) : elements_(other.elements_) {
    InitElementToIter();
  }

  /* \brief Explicit copy assignment
   *
   * Implemented in terms of the copy constructor, and the default
   * move assignment.
   */
  OrderedMap& operator=(const OrderedMap<K, V, Hash, KeyEqual>& other) {
    return *this = OrderedMap(other);
  }

  OrderedMap(OrderedMap<K, V, Hash, KeyEqual>&&) = default;
  OrderedMap& operator=(OrderedMap<K, V, Hash, KeyEqual>&&) = default;

  template <typename Iter>
  OrderedMap(Iter begin, Iter end) : elements_(begin, end) {
    InitElementToIter();
  }

  auto find(const K& k) {
    auto it = elem_to_index_.find(k);
    if (it != elem_to_index_.end()) {
      return elements_.begin() + it->second;
    }
    return elements_.end();
  }

  auto find(const K& k) const {
    auto it = elem_to_index_.find(k);
    if (it != elem_to_index_.end()) {
      return elements_.begin() + it->second;
    }
    return elements_.end();
  }

  V& operator[](const K& k) {
    auto it = elem_to_index_.find(k);
    if (it != elem_to_index_.end()) {
      return elements_[it->second].second;
    }
    elements_.emplace_back(k, V());
    elem_to_index_[k] = elements_.size() - 1;
    return elements_.back().second;
  }

  void insert(const K& k, V v) {
    auto it = elem_to_index_.find(k);
    if (it != elem_to_index_.end()) {
      elements_[it->second].second = std::move(v);
    } else {
      elements_.emplace_back(k, v);
      elem_to_index_[k] = elements_.size() - 1;
    }
  }

  void clear() {
    elements_.clear();
    elem_to_index_.clear();
  }

  size_t count(const K& k) const { return elem_to_index_.count(k); }

  auto begin() const { return elements_.begin(); }
  auto end() const { return elements_.end(); }
  auto begin() { return elements_.begin(); }
  auto end() { return elements_.end(); }

  size_t size() const { return elements_.size(); }
  bool empty() const { return elements_.empty(); }

  void reserve(size_t n) { elem_to_index_.reserve(n); }

 private:
  void InitElementToIter() {
    for (size_t i = 0; i < elements_.size(); i++) {
      elem_to_index_[elements_[i].first] = i;
    }
  }

  std::vector<std::pair<K, V>> elements_;
  std::unordered_map<K, size_t, Hash, KeyEqual> elem_to_index_;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_ORDERED_MAP_H_
