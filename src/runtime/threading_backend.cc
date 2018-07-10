/*!
 *  Copyright (c) 2018 by Contributors
 * \file threading_backend.cc
 * \brief Native threading backend
 */
#include <tvm/runtime/threading_backend.h>
#include <dmlc/logging.h>
#include <thread>
#include <algorithm>
#include <fstream>
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

  // bind worker threads to disjoint cores
  // if worker 0 is offloaded to master, i.e. exclude_worker0 is true,
  // the master thread is bound to core 0.
  void SetAffinity(bool exclude_worker0, const std::vector<unsigned int> *order = NULL,
                   bool reverse = false) {
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
      unsigned core_id;
      if (order != NULL && order->size() >= threads_.size()) {
        if (reverse) {
          core_id = (*order)[order->size() - (i + exclude_worker0) - 1];
        } else {
          core_id = (*order)[i + exclude_worker0];
        }
      } else {
        core_id = i + exclude_worker0;
      }
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
      if (order != NULL && order->size() >= threads_.size()) {
        if (reverse) {
          CPU_SET((*order)[order->size() - 1], &cpuset);
        } else {
          CPU_SET((*order)[0], &cpuset);
        }
      } else {
        CPU_SET(0, &cpuset);
      }
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

 private:
  int num_workers_;
  std::vector<std::thread> threads_;
};

ThreadGroup::ThreadGroup(int num_workers,
                         std::function<void(int)> worker_callback,
                         bool exclude_worker0)
  : impl_(new ThreadGroup::Impl(num_workers, worker_callback, exclude_worker0)) {}
ThreadGroup::~ThreadGroup() { delete impl_; }
void ThreadGroup::Join() { impl_->Join(); }
void ThreadGroup::SetAffinity(bool exclude_worker0, const std::vector<unsigned int> *order,
                              bool reverse) {
  impl_->SetAffinity(exclude_worker0, order, reverse);
}

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


unsigned int configThreadGroup(int mode, int nthreads, std::vector<unsigned int> *sorted_order) {
  unsigned int threads = std::thread::hardware_concurrency();
  std::vector<std::pair <unsigned int, int64_t>> max_freqs;
  int preferred_num = 0;

  // big or LITTLE
  if (mode) {
    for (unsigned int i = 0; i < threads; ++i) {
      std::ostringstream filepath;
      filepath << "/sys/devices/system/cpu/cpu"  << i << "/cpufreq/cpuinfo_max_freq";
      std::ifstream ifs(filepath.str());
      int64_t cur_freq = -1;
      if (!ifs.fail()) {
        ifs >> cur_freq;
        ifs.close();
        max_freqs.push_back(std::make_pair(i, cur_freq));
        if (cur_freq < 0) {
          LOG(WARNING) << "failed to read CPU max frequency!";
        }
      } else {
        LOG(WARNING) << "failed to read CPU max frequency!";
        break;
      }
    }

    auto max = [] (std::pair<unsigned int, int64_t> a, std::pair<unsigned int, int64_t> b) {
      return a.second > b.second;
    };
    std::sort(max_freqs.begin(), max_freqs.end(), max);
    int64_t preferred_freq;
    if (mode == 1) {
      preferred_freq = max_freqs.begin()->second;
    } else {
      preferred_freq = max_freqs.rend()->second;
    }
    for (auto it = max_freqs.begin(); it != max_freqs.end(); it++) {
      sorted_order->push_back(it->first);
      if (preferred_freq == it->second) {
        preferred_num++;
      }
    }
  }

  unsigned int num_workers_used = preferred_num;
  // if a specific number was given, use that
  if (nthreads)
    num_workers_used = nthreads;
  // use default
  if (!num_workers_used)
    num_workers_used = threading::MaxConcurrency();

  return num_workers_used;
}


}  // namespace threading
}  // namespace runtime
}  // namespace tvm
