/*!
 *  Copyright (c) 2017 by Contributors
 * \file thread_pool.cc
 * \brief Threadpool for multi-threading runtime.
 */
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
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
    std::ostringstream os;
    for (size_t i = 0; i < par_errors_.size(); ++i) {
      if (par_errors_[i].length() != 0) {
        os << "Task " << i << " error: " << par_errors_[i] << '\n';
        par_errors_[i].clear();
      }
    }
    TVMAPISetLastError(os.str().c_str());
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

/*! \brief Working queue for each thread */
class ParallelTaskQueue {
 public:
  /*! \brief The task entry */
  struct Task {
    ParallelLauncher* launcher;
    int32_t task_id;
  };
  ParallelTaskQueue() {
    ring_.resize(2);
  }
  /*!
   * \brief Signal to kill the job.
   */
  void SignalForKill() {
    std::lock_guard<std::mutex> lock(mutex_);
    exit_now_.store(true);
    cv_.notify_all();
  }
  /*!
   * \brief Push task into the queue.
   * \param task The task to be pushed.
   */
  void Push(Task task) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (num_pending_ < ring_.size()) {
      CHECK_NE(ring_.size(), 0U);
      ring_[(head_ + num_pending_) % ring_.size()] = task;
      ++num_pending_;
    } else {
      size_t old_size = ring_.size();
      ring_.resize(old_size * 2);
      if (head_ + num_pending_ > old_size) {
        // copy the ring overflow part into the tail.
        size_t ncopy = head_ + num_pending_ - old_size;
        memcpy(&ring_[0] + old_size, &ring_[0], ncopy * sizeof(Task));
      }
      ring_[(head_ + num_pending_) % ring_.size()] = task;
      ++num_pending_;
    }
    if (nwait_consumer_ != 0) {
      lock.unlock();
      cv_.notify_one();
    }
  }
  /*!
   * \brief Pop task from the queue
   * \param task The task to be poped.
   * \param timeout The number of cycles to spin before sleep.
   * \return Whether pop is successful or we need to exit now.
   */
  bool Pop(Task* task, int timeout = 10) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (num_pending_ != 0) {
      *task = ring_[head_];
      head_ = (head_ + 1) % ring_.size();
      --num_pending_;
      if (exit_now_.load()) return false;
    } else {
      lock.unlock();
      // do a bit spin and busy waiting before sleep.
      for (int i = 0; i < timeout && num_pending_ == 0; ++i) {
        std::this_thread::yield();
      }
      lock.lock();
      ++nwait_consumer_;
      cv_.wait(lock, [this] {
          return num_pending_ != 0 || exit_now_.load();
        });
      --nwait_consumer_;
      *task = ring_[head_];
      head_ = (head_ + 1) % ring_.size();
      --num_pending_;
      if (exit_now_.load()) return false;
    }
    return true;
  }

 private:
  // Number of the elments in the queue
  uint32_t num_pending_{0};
  // Queue head
  uint32_t head_{0};
  // Number of consumers to wait.
  uint32_t nwait_consumer_{0};
  // internal mutex
  std::mutex mutex_;
  // cv for consumer
  std::condition_variable cv_;
  // signal for exit now
  std::atomic<bool> exit_now_{false};
  // The internal ring.
  std::vector<Task> ring_;
};

// The thread pool
class ThreadPool {
 public:
  ThreadPool() {
    const char *val = getenv("TVM_NUM_THREADS");
    if (val == nullptr) {
      val = getenv("OMP_NUM_THREADS");
    }
    if (val != nullptr) {
      num_workers_ = atoi(val);
    } else {
#if defined(_M_X64) || defined(__x86_64__)
      // Half to not count hyper threading.
      num_workers_ = std::thread::hardware_concurrency() / 2;
#else
      num_workers_ = std::thread::hardware_concurrency();
#endif
    }
    num_workers_ = std::max(num_workers_, 1);
    this->Init();
  }
  ~ThreadPool() {
    for (std::unique_ptr<ParallelTaskQueue>& q : queues_) {
      q->SignalForKill();
    }
    for (std::thread& t : threads_) {
      t.join();
    }
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
    ParallelTaskQueue::Task tsk;
    tsk.launcher = launcher;
    for (int i = 0; i < num_task; ++i) {
      tsk.task_id = i;
      queues_[i]->Push(tsk);
    }
    return launcher->WaitForJobs();
  }

  static ThreadPool* Global() {
    static ThreadPool inst;
    return &inst;
  }

 private:
  // Initialize the pool.
  void Init() {
    for (int i = 0; i < num_workers_; ++i) {
      queues_.emplace_back(
          std::unique_ptr<ParallelTaskQueue>(new ParallelTaskQueue()));
    }
    threads_.resize(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
      threads_[i] = std::thread([this, i] {
          this->RunWorker(queues_[i].get());
        });
    }
  }
  // Internal worker function.
  void RunWorker(ParallelTaskQueue* queue) {
    ParallelTaskQueue::Task task;
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
  std::vector<std::unique_ptr<ParallelTaskQueue> > queues_;
  std::vector<std::thread> threads_;
};

}  // namespace runtime
}  // namespace tvm

int TVMBackendParallelLaunch(
    FTVMParallelLambda flambda,
    void* cdata,
    int num_task) {
  return tvm::runtime::ThreadPool::Global()->Launch(
      flambda, cdata, num_task, 1);
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
        std::this_thread::yield();
      }
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  return 0;
}
