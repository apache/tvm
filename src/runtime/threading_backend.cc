/*!
 *  Copyright (c) 2018 by Contributors
 * \file threading_backend.cc
 * \brief Native threading backend
 */
#include <tvm/runtime/threading_backend.h>
#include <dmlc/logging.h>
#include <thread>
#include <algorithm>
#if defined(__linux__)
#include <sched.h>
#endif

namespace tvm {
namespace runtime {
namespace threading {

class ThreadGroup::Impl {
 public:
  Impl(int num_workers,
       std::function<void(int)> worker_callback,
       bool exclude_worker0)
      : num_workers_(num_workers) {
    CHECK_GE(num_workers, 1)
      << "Requested a non-positive number of worker threads.";
    for (int i = exclude_worker0; i < num_workers_; ++i) {
      threads_.emplace_back([worker_callback, i] { worker_callback(i); });
    }
    const char *val = getenv("TVM_BIND_THREADS");
    if (val == nullptr || atoi(val) == 1) {
      if (static_cast<size_t>(num_workers_) <= std::thread::hardware_concurrency()) {
        SetAffinity(exclude_worker0);
      } else {
        LOG(WARNING)
          << "The thread affinity cannot be set when the number of workers"
          << "is larger than the number of available cores in the system.";
      }
    }
  }
  ~Impl() { Join(); }

  void Join() {
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
  }

 private:
  // bind worker threads to disjoint cores
  // if worker 0 is offloaded to master, i.e. exclude_worker0 is true,
  // the master thread is bound to core 0.
  void SetAffinity(bool exclude_worker0) {
#if defined(__ANDROID__)
#ifndef CPU_SET
#define CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof (uint64_t))
    typedef struct {
      uint64_t __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
    ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))
#define CPU_ZERO(cpusetp) \
    memset((cpusetp), 0, sizeof(cpu_set_t))
#endif
#endif
#if defined(__linux__) || defined(__ANDROID__)
    for (unsigned i = 0; i < threads_.size(); ++i) {
      unsigned core_id = i + exclude_worker0;
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(core_id, &cpuset);
#if defined(__ANDROID__)
      sched_setaffinity(threads_[i].native_handle(), sizeof(cpu_set_t), &cpuset);
#else
      pthread_setaffinity_np(threads_[i].native_handle(),
          sizeof(cpu_set_t), &cpuset);
#endif
    }
    if (exclude_worker0) {  // bind the master thread to core 0
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(0, &cpuset);
#if defined(__ANDROID__)
      sched_setaffinity(pthread_self(),
        sizeof(cpu_set_t), &cpuset);
#else
      pthread_setaffinity_np(pthread_self(),
        sizeof(cpu_set_t), &cpuset);
#endif
    }
#endif
  }

  int num_workers_;
  std::vector<std::thread> threads_;
};

ThreadGroup::ThreadGroup(int num_workers,
                         std::function<void(int)> worker_callback,
                         bool exclude_worker0)
  : impl_(new ThreadGroup::Impl(num_workers, worker_callback, exclude_worker0)) {}
ThreadGroup::~ThreadGroup() { delete impl_; }
void ThreadGroup::Join() { impl_->Join(); }

void Yield() {
  std::this_thread::yield();
}

int MaxConcurrency() {
  int max_concurrency = 1;
  const char *val = getenv("TVM_NUM_THREADS");
  if (val == nullptr) {
    val = getenv("OMP_NUM_THREADS");
  }
  if (val != nullptr) {
    max_concurrency = atoi(val);
  } else {
    max_concurrency = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    max_concurrency /= 2;  // ignore hyper-threading
#endif
  }
  return std::max(max_concurrency, 1);
}

}  // namespace threading
}  // namespace runtime
}  // namespace tvm
