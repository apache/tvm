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
  T* ring_buff_ptr_{nullptr};

  //! \brief Size of the ring buffer in number of Ts
  const uint32_t ring_buff_size_;

  //! \brief Function that determines whether a T is in flight
  const std::function<bool(T*)> in_flight_;

  //! \brief Tracks the ID of the next T to be added to the ring buffer
  uint32_t id_next_{0};

  //! \brief Tracks the ID of the oldest T in flight
  uint32_t id_oldest_{0};
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_RING_BUFFER_H_
