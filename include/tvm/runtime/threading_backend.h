/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/runtime/threading_backend.h
 * \brief Utilities for manipulating thread pool threads.
 */
#ifndef TVM_RUNTIME_THREADING_BACKEND_H_
#define TVM_RUNTIME_THREADING_BACKEND_H_

#include <functional>
#include <memory>
#include <vector>

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
  ThreadGroup(int num_workers,
              std::function<void(int)> worker_callback,
              bool exclude_worker0 = false);
  ~ThreadGroup();

   /*!
    * \brief Blocks until all non-main threads in the pool finish.
    */
  void Join();

   /*!
    * \brief Set CPU affinity of workers
    *
    * \param exclude_worker0 Whether to use the main thread as a worker. (same
    * as constructor)
    * \param it Optional pointer to an affinity order of CPU ids to bind threads
    * to.
    */
  void SetAffinity(bool exclude_worker0, const std::vector<unsigned int> *order = NULL, bool reverse = false);

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
 * \brief configure the CPU id affinity
 *
 * \param mode the preferred CPU type (0 = default, 1 = big, -1 = little)
 * \param nthreads the number of threads to use (0 = use all)
 * \param sorted_order reference to the CPU id affinity list
 *
 * \return the number of workers to use
 */
unsigned int configThreadGroup(int mode, int nthreads, std::vector<unsigned int> *sorted_order);

}  // namespace threading
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREADING_BACKEND_H_
