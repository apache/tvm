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

#ifndef TVM_RUNTIME_HEXAGON_RING_BUFFER_H_
#define TVM_RUNTIME_HEXAGON_RING_BUFFER_H_

#include <functional>
#include <queue>
#include <unordered_map>
#include <vector>

#include "hexagon_common.h"

namespace tvm {
namespace runtime {
namespace hexagon {

template <class T>
class RingBuffer {
 public:
  //! \brief Returns the number of Ts in flight
  uint32_t InFlight() {
    while (id_oldest_ < id_next_ && !in_flight_(GetAddr(id_oldest_))) {
      id_oldest_++;
    }
    return id_next_ - id_oldest_;
  }

  //! \brief Returns pointer to next T; null if ring buffer is full
  T* Next() {
    if (InFlight() == ring_buff_size_) {
      return nullptr;
    }
    T* next = GetAddr(id_next_);
    id_next_++;
    return next;
  }

  /*! \brief Creates a ring buffer for storage items of type T
   *  \param ring_buff_size Size of the ring buffer in number of Ts
   *  \param in_flight Function that determines whether a T is in flight
   */
  RingBuffer(uint32_t ring_buff_size, std::function<bool(T*)> in_flight)
      : ring_buff_size_(ring_buff_size), in_flight_(in_flight) {
    CHECK_NE(ring_buff_size, 0);
    int ret = posix_memalign(reinterpret_cast<void**>(&ring_buff_ptr_), sizeof(T),
                             sizeof(T) * ring_buff_size_);
    CHECK_EQ(ret, 0);
    CHECK_NE(ring_buff_ptr_, nullptr);
  }

  ~RingBuffer() { free(ring_buff_ptr_); }

 private:
  //! \brief Returns the address of a T given its index
  T* GetAddr(uint32_t id) const {
    uint32_t ring_buff_index = id % ring_buff_size_;
    return ring_buff_ptr_ + ring_buff_index;
  }

  //! \brief Pointer to the ring buffer
  T* ring_buff_ptr_ = nullptr;

  //! \brief Size of the ring buffer in number of Ts
  const uint32_t ring_buff_size_ = 0;

  //! \brief Function that determines whether a T is in flight
  const std::function<bool(T*)> in_flight_;

  //! \brief Tracks the ID of the next T to be added to the ring buffer
  uint32_t id_next_ = 0;

  //! \brief Tracks the ID of the oldest T in flight
  uint32_t id_oldest_ = 0;
};

//! \brief Separates a single RingBuffer into multiple virtual queues with each queue having a
//! unique integer ID; queues allow for indepent users of the same RingBuffer while mainting overall
//! FIFO ordering among all queues
template <class T>
class QueuedRingBuffer : RingBuffer<T> {
 public:
  QueuedRingBuffer(uint32_t max_queues, uint32_t ring_buff_size, std::function<bool(T*)> in_flight)
      : RingBuffer<T>(ring_buff_size, in_flight), max_queues_(max_queues) {
    queue_descriptors_.resize(max_queues_);
  }

  //! \brief Returns pointer to next T; add the queue ID for tracking
  T* Next(uint32_t queue_id) {
    CHECK_LT(queue_id, max_queues_);
    queue_ids_.push_back(queue_id);
    queue_descriptor* d = &queue_descriptors_[queue_id];
    if (d->group_started) {
      // if we have a group started just update then pending count
      d->pending_in_group++;
    } else {
      // else create group with size one
      d->groups.push(1);
      d->pending_total++;
    }
    return RingBuffer<T>::Next();
  }

  //! \brief Returns the number of groups of Ts in flight for a given queue ID
  uint32_t InFlight(uint32_t queue_id) {
    CHECK_LT(queue_id, max_queues_);
    queue_descriptor* d = &queue_descriptors_[queue_id];
    CHECK(!d->group_started);

    uint32_t in_flight = 0;
    // look at the queue IDs for the RingBuffer entries in flight
    for (size_t i = queue_ids_.size() - RingBuffer<T>::InFlight(); i < queue_ids_.size(); ++i) {
      // increment return value if in flight queue ID matches
      if (queue_ids_[i] == queue_id) {
        in_flight++;
      }
    }

    // calculate number of groups in flight
    while (!d->groups.empty() && d->pending_total - d->groups.front() >= in_flight) {
      d->pending_total -= d->groups.front();
      d->groups.pop();
    }

    // return the number of groups in flight
    return d->groups.size();
  }

  //! \brief Start a group of Ts, if not called the deafault group size is one
  void StartGroup(uint32_t queue_id) {
    CHECK_LT(queue_id, max_queues_);
    queue_descriptor* d = &queue_descriptors_[queue_id];
    CHECK(!d->group_started);

    // start group
    d->group_started = true;
    d->pending_in_group = 0;
  }

  //! \brief End a group of Ts
  void EndGroup(uint32_t queue_id) {
    CHECK_LT(queue_id, max_queues_);
    queue_descriptor* d = &queue_descriptors_[queue_id];
    CHECK(d->group_started);
    CHECK(d->pending_in_group);

    // create group
    if (d->pending_in_group) {
      d->groups.emplace(d->pending_in_group);
    }
    d->pending_total += d->pending_in_group;

    // end group
    d->group_started = false;
    d->pending_in_group = 0;
  }

 private:
  struct queue_descriptor {
    uint32_t pending_total = 0;
    uint32_t pending_in_group = 0;
    bool group_started = false;
    std::queue<int> groups;
  };

  const int max_queues_;
  std::vector<int> queue_ids_;
  std::vector<queue_descriptor> queue_descriptors_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_RING_BUFFER_H_
