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

#include <tvm/runtime/c_backend_api.h>

#include <algorithm>
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

/*!
 * \brief Execute the given lambda function in parallel with
 * threading backend in TVM.
 * \tparam T The type of the lambda: "void (int i)".
 * \param flambda The lambda to be executed in parallel.
 * It should have the signature "void (int i)".
 * \param begin The start index of this parallel loop (inclusive).
 * \param end The end index of this parallel loop (exclusive).
 * \example
 *
 * The for loop
 *   for (int i = 0; i < 10; i++) {
 *     a[i] = i;
 *   }
 * should work the same as:
 *   parallel_for_with_threading_backend([&a](int i) {
 *     a[i] = i;
 *   }, 0, 10);
 */
template <typename T>
inline void parallel_for_with_threading_backend(T flambda, int64_t begin, int64_t end);

namespace detail {

// The detailed implementation of `parallel_for_with_threading_backend`.
// To avoid template expansion, the implementation cannot be placed
// in .cc files.

template <typename T>
struct ParallelForWithThreadingBackendLambdaInvoker {
  static int TVMParallelLambdaInvoke(int task_id, TVMParallelGroupEnv* penv, void* cdata) {
    int num_task = penv->num_task;
    // Convert void* back to lambda type.
    T* lambda_ptr = static_cast<T*>(cdata);
    // Invoke the lambda with the task id (thread id).
    (*lambda_ptr)(task_id, num_task);
    return 0;
  }
};

template <typename T>
inline void parallel_launch_with_threading_backend(T flambda) {
  // Launch the lambda by passing its address.
  void* cdata = &flambda;
  TVMBackendParallelLaunch(ParallelForWithThreadingBackendLambdaInvoker<T>::TVMParallelLambdaInvoke,
                           cdata, /*num_task=*/0);
}

}  // namespace detail

template <typename T>
inline void parallel_for_with_threading_backend(T flambda, int64_t begin, int64_t end) {
  if (end - begin == 1) {
    flambda(begin);
    return;
  }

  auto flaunch = [begin, end, flambda](int task_id, int num_task) {
    // For each thread, do static division and call into flambda.
    int64_t total_len = end - begin;
    int64_t step = (total_len + num_task - 1) / num_task;
    int64_t local_begin = std::min(begin + step * task_id, end);
    int64_t local_end = std::min(local_begin + step, end);
    for (int64_t i = local_begin; i < local_end; ++i) {
      flambda(i);
    }
  };
  // Launch with all threads.
  detail::parallel_launch_with_threading_backend(flaunch);
}

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREADING_BACKEND_H_
