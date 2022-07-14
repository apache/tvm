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
 * \file tvm/runtime/threading_backend.h
 * \brief Utilities for manipulating thread pool threads.
 */
#ifndef TVM_RUNTIME_THREADING_BACKEND_H_
#define TVM_RUNTIME_THREADING_BACKEND_H_

#include <functional>
#include <memory>
#include <vector>

#if defined(__linux__) || defined(__ANDROID__)
#if defined(__ANDROID__)
#ifndef CPU_SET
#define CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(uint64_t))
typedef struct {
  uint64_t __bits[CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
  ((cpusetp)->__bits[(cpu) / __NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))
#define CPU_ZERO(cpusetp) memset((cpusetp), 0, sizeof(cpu_set_t))
#define CPU_ISSET(cpu, cpusetp)    \
  (1UL << ((cpu) % __NCPUBITS)) == \
      ((cpusetp)->__bits[(cpu) / __NCPUBITS] & (1UL << ((cpu) % __NCPUBITS)))
#define CPU_EQUAL(left, right) (memcmp(&left, &right, sizeof(cpu_set_t)) == 0)

#endif
#endif
#endif

namespace tvm {
namespace runtime {
namespace threading {

/*!
 * \brief A platform-agnostic abstraction for managing a collection of
 *        thread pool threads.
 */
class ThreadGroup {
 public:
  class Impl;

  /*!
   * \brief Creates a collection of threads which run a provided function.
   *
   * \param num_workers The total number of worker threads in this group.
            Includes main thread if `exclude_worker0 = true`
   * \param worker_callback A callback which is run in its own thread.
            Receives the worker_id as an argument.
   * \param exclude_worker0 Whether to use the main thread as a worker.
   *        If  `true`, worker0 will not be launched in a new thread and
   *        `worker_callback` will only be called for values >= 1. This
   *        allows use of the main thread as a worker.
   */
  ThreadGroup(int num_workers, std::function<void(int)> worker_callback,
              bool exclude_worker0 = false);
  ~ThreadGroup();

  /*!
   * \brief Blocks until all non-main threads in the pool finish.
   */
  void Join();

  enum AffinityMode : int {
    kBig = 1,
    kLittle = -1,
    /*Different threads will get different affinities.*/
    kSpecifyOneCorePerThread = -2,
    /*All threads will get the same core group affinity.*/
    kSpecifyThreadShareAllCore = -3,
  };
  /*!
   * \brief configure the CPU id affinity
   *
   * \param mode The preferred CPU type (1 = big, -1 = little ...).
   * \param nthreads The number of threads to use (0 = use all).
   * \param exclude_worker0 Whether to use the main thread as a worker.
   *        If  `true`, worker0 will not be launched in a new thread and
   *        `worker_callback` will only be called for values >= 1. This
   *        allows use of the main thread as a worker.
   * \param cpus A list of CPU used to set 'cpu affinity'.
   *
   * \return The number of workers to use.
   */
  int Configure(AffinityMode mode, int nthreads, bool exclude_worker0,
                std::vector<unsigned int> cpus = {});

 private:
  Impl* impl_;
};

/*!
 * \brief Platform-agnostic no-op.
 */
void Yield();
/*!
 * \return the maximum number of effective workers for this system.
 */
int MaxConcurrency();
/*!
 * \brief Setting the maximum number of available cores.
 */
void SetMaxConcurrency(int value);
/*!
 * \brief Reset the threads in the pool. All current threads are destroyed and
 * new ones are created.
 *
 * Note that this does nothing when openmp is used.
 */
void ResetThreadPool();

/*!
 * \brief Configuring the CPU affinity mode for the working threads.
 * \param mode The preferred CPU type (1 = big, -1 = little, -2 = kSpecifyOneCorePerThread,
 *  -3 = kSpecifyThreadShareAllCore).
 * \param nthreads The number of threads to use (0 = use all).
 * \param cpus A list of CPUs is used to set the 'cpu affinity' for the worker threads.
 */
TVM_DLL void Configure(tvm::runtime::threading::ThreadGroup::AffinityMode mode, int nthreads,
                       std::vector<unsigned int> cpus);

/*!
 * \brief Get the number of threads being used by the TVM runtime
 * \returns The number of threads used.
 */
int32_t NumThreads();

}  // namespace threading
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREADING_BACKEND_H_
