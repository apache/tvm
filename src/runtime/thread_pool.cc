/*!
 *  Copyright (c) 2017 by Contributors
 * \file thread_pool.cc
 * \brief Threadpool for multi-threading runtime.
 */
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/threading_backend.h>
#include <dmlc/thread_local.h>
#include <dmlc/logging.h>
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
#ifndef _LIBCPP_SGX_CONFIG
#include <iostream>
#endif

const constexpr int kL1CacheBytes = 64;

namespace tvm {
namespace runtime {

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
    std::lock_guard<std::mutex> lock(mutex_);
    num_pending_ = num_task;
    this->cdata = cdata;
    this->flambda = flambda;
    this->env.num_task = num_task;
    has_error_ = false;
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
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
        return num_pending_ == 0;
      });
    if (!has_error_) return 0;
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
    std::unique_lock<std::mutex> lock(mutex_);
    --num_pending_;
    par_errors_[task_id] = TVMGetLastError();
    has_error_ = true;
    if (num_pending_ == 0) {
      lock.unlock();
      cv_.notify_one();
    }
  }
  // Signal that one job has finished.
  void SignalJobFinish() {
    std::unique_lock<std::mutex> lock(mutex_);
    --num_pending_;
    if (num_pending_ == 0) {
      lock.unlock();
      cv_.notify_one();
    }
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
  // The mutex to access local env.
  std::mutex mutex_;
  // The conditional variable.
  std::condition_variable cv_;
  // The pending jobs.
  uint32_t num_pending_;
  // Whether error has been countered.
  bool has_error_;
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
  bool Pop(Task* output, uint32_t spin_count = 300000) {
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
  ThreadPool() {
    num_workers_ = tvm::runtime::threading::MaxConcurrency();
    this->Init();
  }
  ~ThreadPool() {
    if (num_workers_ > 0) Shutdown();
  }
  int Launch(FTVMParallelLambda flambda,
             void* cdata,
             int num_task,
             int need_sync) {
    ParallelLauncher* launcher = ParallelLauncher::ThreadLocal();
    CHECK(!launcher->is_worker)
        << "Cannot launch parallel job inside worker, consider fuse then parallel";
    if (num_task == 0) {
      num_task = num_workers_;
    }
    if (need_sync != 0) {
      CHECK_LE(num_task, num_workers_)
          << "Request parallel sync task larger than number of threads available "
          << " workers=" << num_workers_ << " request=" << num_task;
    }
    launcher->Init(flambda, cdata, num_task, need_sync != 0);
    SpscTaskQueue::Task tsk;
    tsk.launcher = launcher;
    for (int i = 0; i < num_task; ++i) {
      tsk.task_id = i;
      queues_[i]->Push(tsk);
    }
    int res = launcher->WaitForJobs();
    return res;
  }

  void Shutdown() {
    for (std::unique_ptr<SpscTaskQueue>& q : queues_) {
      q->SignalForKill();
    }
    delete threads_;
    num_workers_ = 0;
  }

  static ThreadPool* Global() {
    static ThreadPool inst;
    return &inst;
  }

 private:
  // Initialize the pool.
  void Init() {
    for (int i = 0; i < num_workers_; ++i) {
      // The SpscTaskQueue only host ONE item at a time
      queues_.emplace_back(
          std::unique_ptr<SpscTaskQueue>(new SpscTaskQueue()));
    }
    std::vector<std::function<void()>> callbacks;
    for (int i = 0; i < num_workers_; ++i) {
      callbacks.push_back(std::bind(ThreadPool::RunWorker, queues_[i].get()));
    }
    threads_ = new tvm::runtime::threading::ThreadGroup();
    threads_->Launch(callbacks);
  }
  // Internal worker function.
  static void RunWorker(SpscTaskQueue* queue) {
    SpscTaskQueue::Task task;
    ParallelLauncher::ThreadLocal()->is_worker = true;
    while (queue->Pop(&task)) {
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
  // Number of workers
  int num_workers_;
  std::vector<std::unique_ptr<SpscTaskQueue> > queues_;
  tvm::runtime::threading::ThreadGroup* threads_;
};

}  // namespace runtime
}  // namespace tvm

int TVMBackendParallelLaunch(
    FTVMParallelLambda flambda,
    void* cdata,
    int num_task) {
  int res = tvm::runtime::ThreadPool::Global()->Launch(
      flambda, cdata, num_task, 1);
  return res;
}

int TVMBackendParallelBarrier(int task_id, TVMParallelGroupEnv* penv) {
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
  return 0;
}
