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
 * \file support/ordered_set.h
 * \brief An STL-like ordered set implementation.
 */
#ifndef TVM_SUPPORT_ORDERED_SET_H_
#define TVM_SUPPORT_ORDERED_SET_H_

#include <tvm/runtime/object.h>

#include <functional>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace support {

template <typename T, typename Hash = std::hash<T>, typename KeyEqual = std::equal_to<T>>
class OrderedSet {
 public:
  OrderedSet() = default;

  /* \brief Explicit copy constructor
   *
   * The default copy constructor would copy both `elements_` and
   * `elem_to_iter_`.  While this is the correct behavior for
   * `elements_`, the copy of `elem_to_iter_` would contain references
   * to the original's `element_`, rather than to its own
   */
  OrderedSet(const OrderedSet<T, Hash, KeyEqual>& other) : elements_(other.elements_) {
    InitElementToIter();
  }

  /* \brief Explicit copy assignment
   *
   * Implemented in terms of the copy constructor, and the default
   * move assignment.
   */
  OrderedSet& operator=(const OrderedSet<T, Hash, KeyEqual>& other) {
    return *this = OrderedSet(other);
  }

  OrderedSet(OrderedSet<T, Hash, KeyEqual>&&) = default;
  OrderedSet& operator=(OrderedSet<T, Hash, KeyEqual>&&) = default;

  template <typename Iter>
  OrderedSet(Iter begin, Iter end) : elements_(begin, end) {
    InitElementToIter();
  }

  void push_back(const T& t) {
    if (!elem_to_index_.count(t)) {
      elements_.push_back(t);
      elem_to_index_[t] = elements_.size() - 1;
    }
  }

  void insert(const T& t) { push_back(t); }

  void clear() {
    elements_.clear();
    elem_to_index_.clear();
  }

  size_t count(const T& t) const { return elem_to_index_.count(t); }

  auto begin() const { return elements_.begin(); }
  auto end() const { return elements_.end(); }
  auto size() const { return elements_.size(); }
  auto empty() const { return elements_.empty(); }

 private:
  void InitElementToIter() {
    for (size_t i = 0; i < elements_.size(); ++i) {
      elem_to_index_[elements_[i]] = i;
    }
  }

  std::vector<T> elements_;
  std::unordered_map<T, size_t, Hash, KeyEqual> elem_to_index_;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_ORDERED_SET_H_
