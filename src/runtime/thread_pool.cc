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
#if defined(__linux__)
#include <sched.h>
#endif

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

/*! \brief Lock-free single-producer-single-consumer queue for each thread */
class SpscTaskQueue {
 public:
 /*! \brief The task entry */
  struct Task {
    ParallelLauncher* launcher;
    int32_t task_id;
  };

  SpscTaskQueue(char size) :
    size_(size),
    buffer_(reinterpret_cast<Task*>(new aligned_t[size_])),
    head_(0),
    tail_(0) {
  }

  ~SpscTaskQueue()
  {
    delete[] buffer_;
  }

  /*!
   * \brief Lock-free enqueue.
   * \param input The task to be enqueued.
   * \return Whether the task is enqueued.
   */
  bool Enqueue(const Task& input)
  {
    const uint8_t tail = tail_.load(std::memory_order_relaxed);

    if ((tail + 1) % size_ != (head_.load(std::memory_order_acquire))) {
      buffer_[tail] = input;
      tail_.store((tail + 1) % size_, std::memory_order_release);
      return true;
    }
    return false;
  }

  /*!
   * \brief Lock-free dequeue.
   * \param output The task to be dequeued.
   * \return Whether a task is dequeued.
   */
  bool Dequeue(Task& output)
  {
    const uint8_t head = head_.load(std::memory_order_relaxed);

    if (tail_.load(std::memory_order_acquire) != head) {
      output = buffer_[head];
      head_.store((head + 1) % size_, std::memory_order_release);
      return true;
    }
    return false;
  }

  /*!
   * \brief Push a task into the queue and notify the comsumer if it is on wait.
   * \param input The task to be dequeued.
   */
  void Push(const Task& input)
  {    
    while (!Enqueue(input)) {
      std::this_thread::yield();
    }
    if (pending_.fetch_add(1) == -1) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.notify_one();
    }
  }

  /*!
   * \brief Pop a task out of the queue and condition wait if no tasks.
   * \param output The task to be dequeued.
   * \param timeout The number of iterations to spin before sleep.
   * \return Whether pop is successful (true) or we need to exit now (false).
   */
  bool Pop(Task& output, uint32_t timeout = 100000) {
    // busy wait a bit when the queue is empty
    // if a new task comes to the queue quickly, this wait avoid the worker from sleeping
    for (uint32_t i = 0; i < timeout && pending_.load() == 0; ++i) {
        std::this_thread::yield();
      }
    if (pending_.fetch_sub(1) == 0) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] {
          return pending_.load() >= 0 || exit_now_.load();
        });
    }
    return Dequeue(output);
    }

  /*!
   * \brief Signal to terminate the worker.
   */
  void SignalForKill() {
    exit_now_.store(true);
    cv_.notify_all();
  }

 private:

  struct node_t
  {
    node_t* next;
    Task data;
  };

  typedef std::aligned_storage<sizeof(Task), std::alignment_of<Task>::value>::type \
    aligned_t;

  typedef char cache_line_pad_t[64];
  cache_line_pad_t _pad0;
  // size of the queue, the queue can host size_ - 1 items at most
  const uint8_t size_;
  // pointer to access the item
  Task* const buffer_;

  cache_line_pad_t _pad1;
  // queue head, where one gets a task from the queue
  std::atomic<uint8_t> head_;

  cache_line_pad_t _pad2;
  // queue tail, when one puts a task to the queue
  std::atomic<uint8_t> tail_;

  cache_line_pad_t _pad3;
  // pending tasks in the queue
  std::atomic<int8_t> pending_{0};

  cache_line_pad_t _pad4;
  // signal for exit now
  std::atomic<bool> exit_now_{false};
  
  cache_line_pad_t _pad5;
  // internal mutex
  std::mutex mutex_;
  // cv for consumer
  std::condition_variable cv_;
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
    for (std::unique_ptr<SpscTaskQueue>& q : queues_) {
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
    SpscTaskQueue::Task tsk;
    tsk.launcher = launcher;
    for (int i = 0; i < num_task; ++i) {
      tsk.task_id = i;
      queues_[i]->Push(tsk);
    }
    int res = launcher->WaitForJobs();
    return res;
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
          std::unique_ptr<SpscTaskQueue>(new SpscTaskQueue(2)));
    }
    threads_.resize(num_workers_);
    for (int i = 0; i < num_workers_; ++i) {
      threads_[i] = std::thread([this, i] {
          this->RunWorker(queues_[i].get());
        });
    }
    SetThreadAffinity();
  }
  // Internal worker function.
  void RunWorker(SpscTaskQueue* queue) {
    SpscTaskQueue::Task task;
    ParallelLauncher::ThreadLocal()->is_worker = true;
    while(queue->Pop(task)) {
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
  // bind worker threads to disjoint cores
  void SetThreadAffinity() {
    for (int i=0; i < num_workers_; ++i) {
#if defined(__linux__)
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(i, &cpuset);
      pthread_setaffinity_np(threads_[i].native_handle(), 
        sizeof(cpu_set_t), &cpuset);
#endif
    }
  }
  // Number of workers
  int num_workers_;
  std::vector<std::unique_ptr<SpscTaskQueue> > queues_;
  std::vector<std::thread> threads_;
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
        std::this_thread::yield();
      }
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  return 0;
}
