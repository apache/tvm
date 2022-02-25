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
 * \file threading_backend.cc
 * \brief Native threading backend
 */
#include <tvm/runtime/logging.h>
#include <tvm/runtime/threading_backend.h>

#if defined(__linux__) || defined(__ANDROID__)
#include <fstream>
#include <sstream>
#else
#endif
#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__hexagon__)
#include <dlfcn.h>
#endif
#include <algorithm>
#include <thread>
#define CURRENT_THREAD_HANDLE (static_cast<std::thread::native_handle_type>(0))
namespace tvm {
namespace runtime {
namespace threading {
thread_local int max_concurrency = 0;
class ThreadGroup::Impl {
 public:
  Impl(int num_workers, std::function<void(int)> worker_callback, bool exclude_worker0)
      : num_workers_(num_workers) {
    ICHECK_GE(num_workers, 1) << "Requested a non-positive number of worker threads.";
    for (int i = exclude_worker0; i < num_workers_; ++i) {
      threads_.emplace_back([worker_callback, i] { worker_callback(i); });
    }
    InitSortedOrder();
  }
  ~Impl() { Join(); }

  void Join() {
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
  }

  int Configure(AffinityMode mode, int nthreads, bool exclude_worker0,
                std::vector<unsigned int> cpus) {
    int num_workers_used = 0;
    switch (mode) {
      case kLittle:
        num_workers_used = little_count_;
        break;
      case kBig:
        num_workers_used = big_count_;
        break;
      case kSpecifyOneCorePerThread:
      case kSpecifyThreadShareAllCore:
        num_workers_used = cpus.size();
        sorted_order_ = cpus;
        break;
      default:
        // use default
        num_workers_used = threading::MaxConcurrency();
    }
    // if a specific number was given, use that
    if (nthreads) {
      num_workers_used = nthreads;
    }
    // if MaxConcurrency restricted the number of workers (e.g., due to
    // hyperthreading), respect the restriction. On CPUs with N logical cores
    // and N/2 physical cores this will set affinity to the first N/2 logical
    // ones.
    num_workers_used = std::min(num_workers_, num_workers_used);
    SetAffinity(exclude_worker0, mode);
    return num_workers_used;
  }

 private:
  void SetThreadAffinity(std::thread::native_handle_type thread,
                         const std::vector<unsigned int>& ids) {
#if defined(__linux__) || defined(__ANDROID__)
    if (pthread_equal(thread, CURRENT_THREAD_HANDLE)) {
      thread = pthread_self();
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (auto id : ids) {
      CPU_SET(id, &cpuset);
    }
#if defined(__ANDROID__)
    sched_setaffinity(thread, sizeof(cpu_set_t), &cpuset);
#else
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#endif
#endif
  }

  // bind worker threads to disjoint cores
  // if worker 0 is offloaded to main, i.e. exclude_worker0 is true,
  // the main thread is bound to core 0.
  void SetAffinity(bool exclude_worker0, AffinityMode mode) {
    const char* val = getenv("TVM_BIND_THREADS");
    if (val != nullptr && atoi(val) != 1) {
      return;
    }
    // Do not set affinity if there are more workers than found cores and mode is not kSpecify*.
    if (sorted_order_.size() < static_cast<unsigned int>(num_workers_)) {
      switch (mode) {
        // When the mode is kSpecifyOneCorePerThread or kSpecifyThreadShareAllCore, we should
        // let the threads share all the cpu cores.
        case kSpecifyOneCorePerThread:
        case kSpecifyThreadShareAllCore:
          for (unsigned i = 0; i < threads_.size(); ++i) {
            SetThreadFullCpuAffinity(threads_[i].native_handle(), mode);
          }
          if (exclude_worker0) {  // main thread run task
            SetMasterThreadFullCpuAffinity(mode);
          }
          break;
        case kLittle:
        case kBig:
          LOG(WARNING) << "The thread affinity cannot be set when the number of workers"
                       << "is larger than the number of available cores in the system.";
          break;
      }
    } else {
      ICHECK_GE(sorted_order_.size(), num_workers_);
      switch (mode) {
        case kSpecifyThreadShareAllCore:
          for (unsigned i = 0; i < threads_.size(); ++i) {
            SetThreadFullCpuAffinity(threads_[i].native_handle(), mode);
          }
          break;
        case kLittle:
        case kBig:
        case kSpecifyOneCorePerThread:
          for (unsigned i = 0; i < threads_.size(); ++i) {
            bool reverse = mode == kLittle;
            unsigned core_id;
            if (reverse) {
              core_id = sorted_order_[sorted_order_.size() - (i + exclude_worker0) - 1];
            } else {
              core_id = sorted_order_[i + exclude_worker0];
            }
            SetThreadAffinity(threads_[i].native_handle(), {core_id});
          }
          break;
      }
      if (exclude_worker0) {  // main thread run task
        // Master thread will have free migration on needed cores.
        // Typically, the OS will schedule the main thread to run at core 0,
        // which is idle, when other workers are running.
        // See the comment inside SetMasterThreadFullCpuAffinity function to get more detail.
        SetMasterThreadFullCpuAffinity(mode);
      }
    }
  }

  void SetThreadFullCpuAffinity(std::thread::native_handle_type thread, AffinityMode mode) {
    // For example, we have 2xA72 + 4xA53 (id is 0 - 5, 4, 5 is A72 big core)
    // And we use config_threadpool API to set we will only use 4xA53.
    // The sorted_order will be [4, 5, 0, 1, 2, 3].
    // When to call this API, we have spawn threads on little cores for other workers
    // in SetAffinity function. And for tvm main thread, it should also run on little cores,
    // not big cores (4, 5).

    // Note: this works well on x86 too. Because x86 doesn't have BIG.LITTLE,
    // our implementation will use kBig mode by default and will let main thread
    // run on intended cores.
    std::vector<unsigned> ids;
    switch (mode) {
      case kSpecifyOneCorePerThread:
      case kSpecifyThreadShareAllCore:
        for (size_t i = 0; i < sorted_order_.size(); ++i) {
          ids.push_back(sorted_order_[i]);
        }
        break;
      case kLittle:
        for (int i = 0; i < little_count_; ++i) {
          ids.push_back(sorted_order_[sorted_order_.size() - i - 1]);
        }
        break;
      case kBig:
        int num_cpu_workers = std::min(MaxConcurrency(), big_count_);
        for (int i = 0; i < num_cpu_workers; ++i) {
          ids.push_back(sorted_order_[i]);
        }
        break;
    }
    SetThreadAffinity(thread, ids);
  }

  void SetMasterThreadFullCpuAffinity(AffinityMode mode) {
    SetThreadFullCpuAffinity(CURRENT_THREAD_HANDLE, mode);
  }

  void InitSortedOrder() {
    unsigned int threads = std::thread::hardware_concurrency();
#if defined(__hexagon__)
    // With unsigned PDs, getting the number of available hardware threads
    // is not supported in earlier versions of QuRT. In such cases assume 4.
    if (threads == 0) threads = 4;
#endif
    std::vector<std::pair<unsigned int, int64_t> > max_freqs;

    for (unsigned int i = 0; i < threads; ++i) {
      int64_t cur_freq = 0;
#if defined(__linux__) || defined(__ANDROID__)
      std::ostringstream filepath;
      filepath << "/sys/devices/system/cpu/cpu" << i << "/cpufreq/scaling_max_freq";
      std::ifstream ifs(filepath.str());
      if (!ifs.fail()) {
        if (!(ifs >> cur_freq)) {
          cur_freq = -1;
        }
        ifs.close();
      }
#endif
      max_freqs.push_back(std::make_pair(i, cur_freq));
    }

    auto fcmpbyfreq = [](const std::pair<unsigned int, int64_t>& a,
                         const std::pair<unsigned int, int64_t>& b) {
      return a.second == b.second ? a.first < b.first : a.second > b.second;
    };
    std::sort(max_freqs.begin(), max_freqs.end(), fcmpbyfreq);
    int64_t big_freq = max_freqs.begin()->second;
    int64_t little_freq = max_freqs.rbegin()->second;
    for (auto it = max_freqs.begin(); it != max_freqs.end(); it++) {
      sorted_order_.push_back(it->first);
      if (big_freq == it->second) {
        big_count_++;
      }
      if (big_freq != little_freq && little_freq == it->second) {
        little_count_++;
      }
    }
    if (big_count_ + little_count_ != static_cast<int>(sorted_order_.size())) {
      LOG(WARNING) << "more than two frequencies detected!";
    }
  }

  int num_workers_;
  std::vector<std::thread> threads_;
  std::vector<unsigned int> sorted_order_;
  int big_count_ = 0;
  int little_count_ = 0;
};

ThreadGroup::ThreadGroup(int num_workers, std::function<void(int)> worker_callback,
                         bool exclude_worker0)
    : impl_(new ThreadGroup::Impl(num_workers, worker_callback, exclude_worker0)) {}
ThreadGroup::~ThreadGroup() { delete impl_; }
void ThreadGroup::Join() { impl_->Join(); }

int ThreadGroup::Configure(AffinityMode mode, int nthreads, bool exclude_worker0,
                           std::vector<unsigned int> cpus) {
  return impl_->Configure(mode, nthreads, exclude_worker0, cpus);
}

void Yield() { std::this_thread::yield(); }
/*!
 * \bief Set the maximum number of available cores.
 */
void SetMaxConcurrency(int value) {
  if (value > 0) {
    max_concurrency = value;
  }
}
int MaxConcurrency() {
  int max_concurrency = 1;
  if (tvm::runtime::threading::max_concurrency != 0) {
    max_concurrency = tvm::runtime::threading::max_concurrency;
  } else {
    const char* val = getenv("TVM_NUM_THREADS");
    if (val == nullptr) {
      val = getenv("OMP_NUM_THREADS");
    }
    if (val != nullptr) {
      max_concurrency = atoi(val);
    } else {
      max_concurrency = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
      max_concurrency /= 2;  // ignore hyper-threading
#elif defined(__hexagon__)
      // With unsigned PDs, getting the number of available hardware threads
      // is not supported in earlier versions of QuRT. In such cases assume 4.
      // If running on simulator, set max_concurrency to 1.
      if (max_concurrency == 0) {
        if (dlsym(RTLD_DEFAULT, "running_in_sim_dev_17bc90206f6cf5a7")) {
          max_concurrency = 1;
        } else {
          max_concurrency = 4;
        }
      }
#endif
    }
  }
  return std::max(max_concurrency, 1);
}

}  // namespace threading
}  // namespace runtime
}  // namespace tvm
