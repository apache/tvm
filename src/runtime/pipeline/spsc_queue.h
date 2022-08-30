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
#ifndef TVM_RUNTIME_PIPELINE_SPSC_QUEUE_H_
#define TVM_RUNTIME_PIPELINE_SPSC_QUEUE_H_
#include <cstddef>
#include <thread>
/*!\brief A single producer and single consumer lock free queue.
 */
template <typename SlotType, typename IDType = int, int QueueLength = 1024>
class SPSCLockFreeQueue {
 public:
  explicit SPSCLockFreeQueue(IDType id) : id_(id) {}
  /*A read barrier enforcing the CPU to performe the reads before this barrier.*/
  inline void read_barrier() { std::atomic_thread_fence(std::memory_order_acquire); }
  /*A write barrier enforcing the CPU to performe the writes before this barrier.*/
  inline void write_barrier() { std::atomic_thread_fence(std::memory_order_release); }
  /*!\brief Checking whether the queue is full.*/
  bool Full() {
    read_barrier();
    return ((tail_ + 1) % len_) == head_;
  }
  /*!brief Checking whether the queue is empty.*/
  bool Empty() {
    read_barrier();
    return head_ == tail_;
  }
  /*!
   * \brief Pushing the data into the queue. Only a single producer will call this function.
   * \param data The data which is pushed into the queue.
   * \return Return false when the queue is full. Otherwise, return true.
   */
  template <typename data_type>
  bool Push(const data_type& data) {
    if (Full()) return false;
    queue_[tail_] = data;
    write_barrier();
    tail_ = (tail_ + 1) % len_;
    return true;
  }
  /*!
   * \brief Poll the data from the front of the queue. Only the single consumer will call this
   *  function.
   * \param data A pointer to the structure which stores the polled data..
   * \return Returning false when the queue is empty. Otherwise, return true.
   */
  template <typename data_type>
  bool Poll(data_type* data) {
    if (Empty()) return false;
    *data = queue_[head_];
    write_barrier();
    head_ = (head_ + 1) % len_;
    return true;
  }

 private:
  /*!\brief The pointer points to the first slot with valid data in the queue.*/
  size_t head_ = 0;
  /*!\brief The end of the queue at which elements are added.*/
  size_t tail_ = 0;
  /*!\brief The length of the queue.*/
  size_t len_ = QueueLength;
  /*!\brief The queue used to store the data.*/
  SlotType queue_[QueueLength];
  /*!\brief The ID of the queue.*/
  IDType id_;
};
#endif  // TVM_RUNTIME_PIPELINE_SPSC_QUEUE_H_
