/*!
 *  Copyright (c) 2018 by Contributors
 * \file threading_backend.h
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

class ThreadGroup {
 public:
  class Impl;

   /*!
    * \brief Starts and manages threads which back a thread pool.
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

 private:
  Impl* impl_;
};

/*!
 * \brief Platform-agnostic no-op
 */
void Yield();

/*!
 * \brief Returns the maximum number of effective workers for this system.
 */
int MaxConcurrency();

}  // namespace threading
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_THREADING_BACKEND_H_
