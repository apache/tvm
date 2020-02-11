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
 * \file thread_pool.cc
 * \brief Threadpool for multi-threading runtime.
 */
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/threading_backend.h>
#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
#if TVM_THREADPOOL_USE_OPENMP
#include <omp.h>
#endif
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <memory>
#include <sstream>

const constexpr int kL1CacheBytes = 64;

namespace tvm {
namespace runtime {
namespace {

constexpr uint32_t kDefaultSpinCount = 300000;

uint32_t GetSpinCount() {
  const char* val = getenv("TVM_THREAD_POOL_SPIN_COUNT");
  if (!val) {
    return kDefaultSpinCount;
  }
  return atoi(val);
}

}  // namespace

// stride in the page, fit to cache line.
constexpr int kSyncStride = 64 / sizeof(std::atomic<int>);

/*!
 * \brief Thread local master environment.
 */
class ParallelLauncher {
 public:
  // Reset the the task request.
  void Init(FTVMParallelLambda flambda,
            void* cdata,
            int num_task,
            bool need_sync) {
    num_pending_.store(num_task);
    this->cdata = cdata;
    this->flambda = flambda;
    this->env.num_task = num_task;
    has_error_.store(false);
    // reshape
    if (static_cast<size_t>(num_task) > par_errors_.size()) {
      par_errors_.resize(num_task + 1);
      if (need_sync) {
        delete[] sync_counter_;
        sync_counter_ = new std::atomic<int>[num_task * kSyncStride];
      }
    }
    if (need_sync) {
      for (int i = 0; i < num_task; ++i) {
        sync_counter_[i * kSyncStride].store(
            0, std::memory_order_relaxed);
      }
      this->env.sync_handle = sync_counter_;
    } else {
      this->env.sync_handle = nullptr;
    }
  }
  ~ParallelLauncher() {
    delete[] sync_counter_;
  }
  // Wait n jobs to finish
  int WaitForJobs() {
    while (num_pending_.load() != 0) {
      tvm::runtime::threading::Yield();
    }
    if (!has_error_.load()) return 0;
    // the following is intended to use string due to
    // security issue raised in SGX backend
    std::string err("");
    for (size_t i = 0; i < par_errors_.size(); ++i) {
      if (par_errors_[i].length() != 0) {
        err += "Task " + std::to_string(i) + " error: " + par_errors_[i] + '\n';
        par_errors_[i].clear();
      }
    }
    TVMAPISetLastError(err.c_str());
    return -1;
  }
  // Signal that one job has finished.
  void SignalJobError(int task_id) {
    num_pending_.fetch_sub(1);
    par_errors_[task_id] = TVMGetLastError();
    has_error_.store(true);
  }
  // Signal that one job has finished.
  void SignalJobFinish() {
    num_pending_.fetch_sub(1);
  }
  // Get thread local version of the store.
  static ParallelLauncher* ThreadLocal() {
    return dmlc::ThreadLocalStore<ParallelLauncher>::Get();
  }
  // The parallel lambda
  FTVMParallelLambda flambda;
  // The closure data
  void* cdata;
  // Local env
  TVMParallelGroupEnv env;
  // Whether this thread is worker of the pool.
  // used to prevent recursive launch.
  bool is_worker{false};

 private:
  // The pending jobs.
  std::atomic<int32_t> num_pending_;
  // Whether error has been countered.
  std::atomic<bool> has_error_;
  // The counter page.
  std::atomic<int32_t>* sync_counter_{nullptr};
  // The error message
  std::vector<std::string> par_errors_;
};

/*! \brief Lock-free single-producer-single-consumer queue for each thread */
class SpscTaskQueue {
 public:
  /*! \brief The task entry */
  struct Task {
    ParallelLauncher* launcher;
    int32_t task_id;
  };

  SpscTaskQueue() :
    buffer_(new Task[kRingSize]),
    head_(0),
    tail_(0) {
  }

  ~SpscTaskQueue() {
    delete[] buffer_;
  }

  /*!
   * \brief Push a task into the queue and notify the comsumer if it is on wait.
   * \param input The task to be dequeued.
   */
  void Push(const Task& input) {
    while (!Enqueue(input)) {
      tvm::runtime::threading::Yield();
    }
    if (pending_.fetch_add(1) == -1) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.notify_one();
    }
  }

  /*!
   * \brief Pop a task out of the queue and condition wait if no tasks.
   * \param output The pointer to the task to be dequeued.
   * \param spin_count The number of iterations to spin before sleep.
   * \return Whether pop is successful (true) or we need to exit now (false).
   */
  bool Pop(Task* output, uint32_t spin_count) {
    // Busy wait a bit when the queue is empty.
    // If a new task comes to the queue quickly, this wait avoid the worker from sleeping.
    // The default spin count is set by following the typical omp convention
    for (uint32_t i = 0; i < spin_count && pending_.load() == 0; ++i) {
      tvm::runtime::threading::Yield();
    }
    if (pending_.fetch_sub(1) == 0) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
          return pending_.load() >= 0 || exit_now_.load();
        });
    }
    if (exit_now_.load(std::memory_order_relaxed)) {
      return false;
    }
    const uint32_t head = head_.load(std::memory_order_relaxed);
    // sanity check if the queue is empty
    CHECK(tail_.load(std::memory_order_acquire) != head);
    *output = buffer_[head];
    head_.store((head + 1) % kRingSize, std::memory_order_release);
    return true;
  }

  /*!
   * \brief Signal to terminate the worker.
   */
  void SignalForKill() {
    std::lock_guard<std::mutex> lock(mutex_);
    exit_now_.store(true);
    cv_.notify_all();
  }

 protected:
  /*!
   * \brief Lock-free enqueue.
   * \param input The task to be enqueued.
   * \return Whether the task is enqueued.
   */
  bool Enqueue(const Task& input) {
    if (exit_now_.load(std::memory_order_relaxed)) return false;

    const uint32_t tail = tail_.load(std::memory_order_relaxed);

    if ((tail + 1) % kRingSize != (head_.load(std::memory_order_acquire))) {
      buffer_[tail] = input;
      tail_.store((tail + 1) % kRingSize, std::memory_order_release);
      return true;
    }
    return false;
  }

  // the cache line paddings are used for avoid false sharing between atomic variables
  typedef char cache_line_pad_t[kL1CacheBytes];
  cache_line_pad_t pad0_;
  // size of the queue, the queue can host size_ - 1 items at most
  // define it as a constant for better compiler optimization
  static constexpr const int kRingSize = 2;
  // pointer to access the item
  Task* const buffer_;

  cache_line_pad_t pad1_;
  // queue head, where one gets a task from the queue
  std::atomic<uint32_t> head_;

  cache_line_pad_t pad2_;
  // queue tail, when one puts a task to the queue
  std::atomic<uint32_t> tail_;

  cache_line_pad_t pad3_;
  // pending tasks in the queue
  std::atomic<int8_t> pending_{0};

  cache_line_pad_t pad4_;
  // signal for exit now
  std::atomic<bool> exit_now_{false};

  // internal mutex
  std::mutex mutex_;
  // cv for consumer
  std::condition_variable cv_;
};

// The thread pool
class ThreadPool {
 public:
  ThreadPool(): num_workers_(tvm::runtime::threading::MaxConcurrency()) {
    for (int i = 0; i < num_workers_; ++i) {
      // The SpscTaskQueue only hosts ONE item at a time
      queues_.emplace_back(std::unique_ptr<SpscTaskQueue>(new SpscTaskQueue()));
    }
    const char* exclude_worker0 = getenv("TVM_EXCLUDE_WORKER0");
    if (exclude_worker0 && atoi(exclude_worker0) == 0) {
      exclude_worker0_ = false;
    }
    threads_ = std::unique_ptr<tvm::runtime::threading::ThreadGroup>(
        new tvm::runtime::threading::ThreadGroup(
          num_workers_, [this](int worker_id) { this->RunWorker(worker_id); },
          exclude_worker0_ /* include_main_thread */));
    num_workers_used_ = threads_->Configure(threading::ThreadGroup::kBig, 0, exclude_worker0_);
  }
  ~ThreadPool() {
    for (std::unique_ptr<SpscTaskQueue>& q : queues_) {
      q->SignalForKill();
    }
    threads_.reset();
  }
  int Launch(FTVMParallelLambda flambda,
             void* cdata,
             int num_task,
             int need_sync) {
    ParallelLauncher* launcher = ParallelLauncher::ThreadLocal();
    CHECK(!launcher->is_worker)
        << "Cannot launch parallel job inside worker, consider fuse then parallel";
    if (num_task == 0) {
      num_task = num_workers_used_;
    }
    if (need_sync != 0) {
      CHECK_LE(num_task, num_workers_used_)
          << "Request parallel sync task larger than number of threads used "
          << " workers=" << num_workers_used_ << " request=" << num_task;
    }
    launcher->Init(flambda, cdata, num_task, need_sync != 0);
    SpscTaskQueue::Task tsk;
    tsk.launcher = launcher;
    // if worker0 is taken by the master, queues_[0] is abandoned
    for (int i = exclude_worker0_; i < num_task; ++i) {
      tsk.task_id = i;
      queues_[i]->Push(tsk);
    }
    // use the master thread to run task 0
    if (exclude_worker0_) {
      TVMParallelGroupEnv* penv = &(tsk.launcher->env);
      if ((*tsk.launcher->flambda)(0, penv, cdata) == 0) {
        tsk.launcher->SignalJobFinish();
      } else {
        tsk.launcher->SignalJobError(tsk.task_id);
      }
    }
    int res = launcher->WaitForJobs();
    return res;
  }

  static ThreadPool* ThreadLocal() {
    return dmlc::ThreadLocalStore<ThreadPool>::Get();
  }

  void UpdateWorkerConfiguration(threading::ThreadGroup::AffinityMode mode, int nthreads) {
    // this will also reset the affinity of the ThreadGroup
    // may use less than the MaxConcurrency number of workers
    num_workers_used_ = threads_->Configure(mode, nthreads,
                                            exclude_worker0_);
    // if MaxConcurrency restricted the number of workers (e.g., due to
    // hyperthreading), respect the restriction
    num_workers_used_ = std::min(num_workers_, num_workers_used_);
  }

 private:
  // Internal worker function.
  void RunWorker(int worker_id) {
    SpscTaskQueue* queue = queues_[worker_id].get();
    SpscTaskQueue::Task task;
    ParallelLauncher::ThreadLocal()->is_worker = true;
    // Initialize the spin count (from envvar TVM_THREAD_POOL_SPIN_COUNT) on
    // the global first use of the ThreadPool.
    // TODO(tulloch): should we make this configurable via standard APIs?
    static size_t spin_count = GetSpinCount();
    while (queue->Pop(&task, spin_count)) {
      CHECK(task.launcher != nullptr);
      TVMParallelGroupEnv* penv = &(task.launcher->env);
      void* cdata = task.launcher->cdata;
      if ((*task.launcher->flambda)(task.task_id, penv, cdata) == 0) {
        task.launcher->SignalJobFinish();
      } else {
        task.launcher->SignalJobError(task.task_id);
      }
    }
  }
  int num_workers_;
  // number of workers used (can be restricted with affinity pref)
  int num_workers_used_;
  // if or not to exclude worker 0 and use master to run task 0
#ifndef _LIBCPP_SGX_CONFIG
  bool exclude_worker0_{true};
#else
  bool exclude_worker0_{false};
#endif
  std::vector<std::unique_ptr<SpscTaskQueue> > queues_;
  std::unique_ptr<tvm::runtime::threading::ThreadGroup> threads_;
};

TVM_REGISTER_GLOBAL("runtime.config_threadpool")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    threading::ThreadGroup::AffinityMode mode =\
    static_cast<threading::ThreadGroup::AffinityMode>(\
    static_cast<int>(args[0]));
    int nthreads = args[1];
    ThreadPool::ThreadLocal()->UpdateWorkerConfiguration(mode, nthreads);
});


}  // namespace runtime
}  // namespace tvm


int TVMBackendParallelLaunch(
    FTVMParallelLambda flambda,
    void* cdata,
    int num_task) {
#if !TVM_THREADPOOL_USE_OPENMP
  int res = tvm::runtime::ThreadPool::ThreadLocal()->Launch(
      flambda, cdata, num_task, 1);
  return res;
#else
  int num_workers = tvm::runtime::threading::MaxConcurrency();
  if (num_task == 0) num_task = num_workers;
  omp_set_num_threads(num_workers);
  #pragma omp parallel num_threads(num_workers)
  {
    TVMParallelGroupEnv env;
    env.num_task = num_task;
    (*flambda)(omp_get_thread_num(), &env, cdata);
  }
  return 0;
#endif
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) {
#if TVM_THREADPOOL_USE_OPENMP
  #pragma omp barrier
#else
  using tvm::runtime::kSyncStride;
  int num_task = penv->num_task;
  std::atomic<int>* sync_counter =
      reinterpret_cast<std::atomic<int>*>(penv->sync_handle);
  int old_counter = sync_counter[task_id * kSyncStride].fetch_add(
      1, std::memory_order_release);
  for (int i = 0; i < num_task; ++i) {
    if (i != task_id) {
      while (sync_counter[i * kSyncStride].load(
                 std::memory_order_relaxed) <= old_counter) {
        tvm::runtime::threading::Yield();
      }
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
#endif
  return 0;
}
