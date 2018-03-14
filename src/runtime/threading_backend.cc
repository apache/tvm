/*!
 *  Copyright (c) 2018 by Contributors
 * \file threading_backend.cc
 * \brief Native threading backend
 */
#include <tvm/runtime/threading_backend.h>
#include <dmlc/logging.h>
#include <thread>
#if defined(__linux__)
#include <sched.h>
#endif
#include <iostream>

namespace tvm {
namespace runtime {
namespace threading {

class ThreadGroup::ThreadGroupImpl {
 public:
  ~ThreadGroupImpl() {
    Join();
  }

  void Launch(std::vector<std::function<void()>> task_callbacks) {
    for (const auto& cb : task_callbacks) {
      threads_.emplace_back(cb);
    }
    const char *val = getenv("TVM_BIND_THREADS");
    if (val == nullptr || atoi(val) == 1) {
      if (Size() <= std::thread::hardware_concurrency()) {
        SetAffinity();
      } else {
        LOG(WARNING)
          << "The thread affinity cannot be set when the number of workers"
          << "is larger than the number of available cores in the system.";
      }
    }
  }

  size_t Size() { return threads_.size(); }

  size_t Join() {
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
  }

 private:
    // bind worker threads to disjoint cores
  void SetAffinity() {
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
      for (unsigned i=0; i < threads_.size(); ++i) {
#if defined(__linux__) || defined(__ANDROID__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
#if defined(__ANDROID__)
        sched_setaffinity(threads_[i].native_handle(), sizeof(cpu_set_t), &cpuset);
#else
        pthread_setaffinity_np(threads_[i].native_handle(),
            sizeof(cpu_set_t), &cpuset);
#endif
#endif
      }
    }

    std::vector<std::thread> threads_;
};

ThreadGroup::ThreadGroup(): impl_(new ThreadGroup::ThreadGroupImpl()) {}
ThreadGroup::~ThreadGroup() { delete impl_; }
void ThreadGroup::Launch(std::vector<std::function<void()>> task_callbacks) {
  return impl_->Launch(task_callbacks);
}
size_t ThreadGroup::Size() { return impl_->Size(); }
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
