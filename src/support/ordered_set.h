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

#include <list>
#include <unordered_map>

namespace tvm {
namespace support {

namespace detail {
/* \brief Utility to allow use for standard and ObjectRef types
 *
 * \tparam T The type held by the OrderedSet
 */
template <typename T, typename = void>
struct OrderedSetLookupType {
  using MapType = std::unordered_map<T, typename std::list<T>::iterator>;
};

template <typename T>
struct OrderedSetLookupType<T, std::enable_if_t<std::is_base_of_v<runtime::ObjectRef, T>>> {
  using MapType = std::unordered_map<T, typename std::list<T>::iterator, runtime::ObjectPtrHash,
                                     runtime::ObjectPtrEqual>;
};
}  // namespace detail

template <typename T>
class OrderedSet {
 public:
  OrderedSet() = default;

  template <typename Iter>
  OrderedSet(Iter begin, Iter end) {
    for (auto it = begin; it != end; it++) {
      push_back(*it);
    }
  }

  void push_back(const T& t) {
    if (!elem_to_iter_.count(t)) {
      elements_.push_back(t);
      elem_to_iter_[t] = std::prev(elements_.end());
    }
  }

  void insert(const T& t) { push_back(t); }

  void erase(const T& t) {
    if (auto it = elem_to_iter_.find(t); it != elem_to_iter_.end()) {
      elements_.erase(it->second);
      elem_to_iter_.erase(it);
    }
  }

  void clear() {
    elements_.clear();
    elem_to_iter_.clear();
  }

  size_t count(const T& t) const { return elem_to_iter_.count(t); }

  auto begin() const { return elements_.begin(); }
  auto end() const { return elements_.end(); }
  auto size() const { return elements_.size(); }
  auto empty() const { return elements_.empty(); }

 private:
  std::list<T> elements_;
  typename detail::OrderedSetLookupType<T>::MapType elem_to_iter_;
};

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_ORDERED_SET_H_
