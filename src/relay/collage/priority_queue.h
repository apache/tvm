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
 * \file src/relay/collage/priority_queue.h
 * \brief An updatable priority queue.
 */

#ifndef TVM_RELAY_COLLAGE_PRIORITY_QUEUE_H_
#define TVM_RELAY_COLLAGE_PRIORITY_QUEUE_H_

#include <set>

namespace tvm {
namespace relay {
namespace collage {

/*! \brief Priority queue of search states, ordered by increasing cost. */
template <typename T, typename CmpTPtr, typename EqTPtr>
class PriorityQueue {
 public:
  PriorityQueue() = default;

  /*! \brief Pushes \p item onto the queue. */
  void Push(T* item) { set_.emplace(item); }

  /*! \brief Pops the item with the least cost off the queue. */
  T* Pop() {
    ICHECK(!set_.empty());
    T* item = *set_.begin();
    set_.erase(set_.begin());
    return item;
  }

  /*! \brief Updates the queue to account for \p item's best cost being lowered. */
  void Update(T* item) {
    auto itr = std::find_if(set_.begin(), set_.end(),
                            [item](const T* that) { return EqTPtr()(that, item); });
    ICHECK(itr != set_.end());
    set_.erase(itr);
    set_.emplace(item);
  }

  bool empty() const { return set_.empty(); }
  size_t size() const { return set_.size(); }

 private:
  // TODO(mbs): Actually use a pri-queue datastructure!
  std::set<T*, CmpTPtr> set_;
};

}  // namespace collage
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_COLLAGE_PRIORITY_QUEUE_H_
