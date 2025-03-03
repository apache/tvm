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
 * \file disco_worker_thread.h
 * \brief This file defines a worker in Disco. A worker can be launched in a separate thread or
 * process as long as the channel supports bi-directional communication in-between the worker and
 * the controler.
 */
#ifndef TVM_RUNTIME_DISCO_DISCO_WORKER_THREAD_H_
#define TVM_RUNTIME_DISCO_DISCO_WORKER_THREAD_H_

#include <tvm/runtime/disco/disco_worker.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <thread>
#include <utility>

namespace tvm {
namespace runtime {

/*!
 * \brief A worker thread in Disco, which upon creation, launches a new thread to run the
 * DiscoWorker.
 * \sa DiscoWorker
 */
class DiscoWorkerThread {
 public:
  /*!
   * \brief Construct a worker thread.
   * \param worker_id The id of the worker.
   * \param num_workers The total number of workers.
   * \param num_groups The total number of worker groups.
   * \param worker_zero_data_ The data shared between worker-0 and the controler. It's a nullptr if
   * the worker is not worker-0.
   * \note This method is implemented in threaded worker, because it depends on creation of a
   * sub-class of DiscoChannel, DiscoThreadChannel, which is hidden from the public interface.
   */
  explicit DiscoWorkerThread(int worker_id, int num_workers, int num_groups,
                             WorkerZeroData* worker_zero_data_);

  /*! \brief Move constructor. */
  explicit DiscoWorkerThread(DiscoWorkerThread&& other)
      : channel(std::move(other.channel)),
        worker(std::move(other.worker)),
        thread(std::move(other.thread)) {}

  /*! \brief Copy constructor is disabled */
  DiscoWorkerThread(const DiscoWorkerThread& other) = delete;

  /*! \brief Destructor that joins the thread before destruction */
  ~DiscoWorkerThread() {
    if (this->thread != nullptr) {
      this->thread->join();
    }
  }

  /*! \brief The communication channel between the controler and the worker */
  std::unique_ptr<DiscoChannel> channel;
  /*! \brief The worker whose internal state is visible to the controler */
  std::unique_ptr<DiscoWorker> worker;
  /*! \brief The thread that runs the worker's main loop. */
  std::unique_ptr<std::thread> thread;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_DISCO_WORKER_THREAD_H_
