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
 * \file worker.h
 * \brief This file defines a worker in Disco. A worker can be launched in a separate thread or
 * process as long as the channel supports bi-directional communication in-between the worker and
 * the controler.
 */
#ifndef TVM_RUNTIME_DISCO_WORKER_H_
#define TVM_RUNTIME_DISCO_WORKER_H_

#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/packed_func.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief A special communication channel between controler and worker-0,
 * assuming they are always collocated in the same process.
 */
class WorkerZeroData {
 public:
  /*!
   * \brief The host-side arrays to passed to worker-0 for special uses, for example,
   * copy-to-worker0 and copy-from-worker0
   */
  std::queue<NDArray> host_arrays;
  /*! \brief The mutex that guards `host_arrays` */
  std::mutex queue_mutex_;
};

/*!
 * \brief A worker in Disco. It takes a channel to communication with the controler.
 * The worker can be run in a separate thread or process as long as the channel supports
 * bi-directional communication in-between.
 */
class DiscoWorker {
 public:
  /*!
   * \brief Construct a worker.
   * \param worker_id The id of the worker.
   * \param worker_zero_data The data shared between worker-0 and the controler. It's a nullptr if
   * the worker is not worker-0.
   * \param channel The communication channel between the worker and the controler.
   */
  explicit DiscoWorker(int worker_id, int num_workers, WorkerZeroData* worker_zero_data,
                       DiscoChannel* channel)
      : worker_id(worker_id),
        num_workers(num_workers),
        default_device(Device{DLDeviceType::kDLCPU, 0}),
        worker_zero_data(worker_zero_data),
        channel(channel),
        register_file{} {}

  /*! \brief Main loop of the worker */
  void MainLoop();
  /*! \brief Get the worker instance on the current thread */
  static DiscoWorker* ThreadLocal();
  /*! \brief Set the specific register to a specific value */
  void SetRegister(int reg_id, TVMArgValue value);

  /*! \brief The id of the worker.*/
  int worker_id;
  /*! \brief Total number of workers */
  int num_workers;
  /*! \brief The default device to allocate data if not specified */
  Device default_device;
  /*! \brief The name of the underlying collective communication library. */
  String ccl;
  /*!
   * \brief The data shared between worker-0 and the controler. It's a nullptr if
   * the worker is not worker-0.
   * \note This data structure is owned by the controler.
   */
  WorkerZeroData* worker_zero_data;
  /*!
   * \brief The communication channel between the worker and the controler.
   * \note This data structure is owned by the controler.
   */
  DiscoChannel* channel;
  /*! \brief The registers in the worker */
  std::vector<TVMRetValue> register_file;

  struct Impl;
  friend struct DiscoWorker::Impl;
};

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
   * \param worker_zero_data_ The data shared between worker-0 and the controler. It's a nullptr if
   * the worker is not worker-0.
   */
  explicit DiscoWorkerThread(int worker_id, int num_workers, WorkerZeroData* worker_zero_data_);

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
#endif  // TVM_RUNTIME_DISCO_WORKER_H_
