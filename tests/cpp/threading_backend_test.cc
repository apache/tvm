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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/threading_backend.h>

#include <atomic>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

constexpr size_t N = 128;
void AtomicCompute(int task_id, size_t n, std::atomic<size_t>* acc, TVMParallelGroupEnv* penv) {
  const size_t N_per_task = (n + penv->num_task - 1) / penv->num_task;
  for (size_t i = task_id * N_per_task; i < n && i < (task_id + 1) * N_per_task; ++i) {
    acc->fetch_add(i, std::memory_order_relaxed);
  }
  return;
}

class AffinityCheck {
 public:
  AffinityCheck(uint32_t parent_id, int max_concurrency, std::atomic<size_t>* acc)
      : id_(parent_id), max_concurrency_(max_concurrency), acc_(acc) {}

  void Compute(int task_id, size_t n, TVMParallelGroupEnv* penv) {
    AtomicCompute(task_id, n, acc_, penv);
  }

  int GetComputeResult() { return acc_->load(std::memory_order_relaxed); }

  void GetAffinity(int task_id) {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    std::lock_guard<std::mutex> lock(mutex_);
    thread_affinity_[task_id] = cpuset;
    // Printing the current thread CPU affinity.
    std::ostringstream str;
    for (int i = 0; i < max_concurrency_; i++) {
      if (CPU_ISSET(i, &cpuset)) {
        str << i << ",";
      }
    }
    LOG(INFO) << "id:" << id_ << " taskid:" << task_id << " affinity:" << str.str() << std::endl;
#endif
  }

  bool VerifyAffinity(const std::vector<uint32_t>& cpus) {
#if defined(__linux__)
    std::unordered_set<uint32_t> uset;
    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
    for (auto x : cpus) {
      CPU_SET(x, &cpu_mask);
      uset.insert(x);
    }

    for (auto x : thread_affinity_) {
      if (!CPU_EQUAL(&cpu_mask, &x.second)) {
        bool cpu_find = false;
        for (auto cpu : uset) {
          CPU_ISSET(cpu, &x.second);
          uset.erase(cpu);
          cpu_find = true;
          break;
        }
        if (!cpu_find) return false;
      }
    }
#endif
    return true;
  }

 private:
  uint32_t id_;
  int max_concurrency_;
  std::atomic<size_t>* acc_;
  std::mutex mutex_;
#if defined(__linux__)
  std::unordered_map<int, cpu_set_t> thread_affinity_;
#endif
};

static FTVMParallelLambda atomic_add_task_id = [](int task_id, TVMParallelGroupEnv* penv,
                                                  void* cdata) -> int {
  auto* data = reinterpret_cast<std::atomic<size_t>*>(cdata);
  AtomicCompute(task_id, N, data, penv);
  return 0;
};

static FTVMParallelLambda affinity_check_task_id = [](int task_id, TVMParallelGroupEnv* penv,
                                                      void* cdata) -> int {
  auto* data = reinterpret_cast<AffinityCheck*>(cdata);
  data->Compute(task_id, N, penv);
  data->GetAffinity(task_id);
  return 0;
};

TEST(ThreadingBackend, TVMBackendParallelLaunch) {
  std::atomic<size_t> acc(0);
  TVMBackendParallelLaunch(atomic_add_task_id, &acc, 0);
  EXPECT_EQ(acc.load(std::memory_order_relaxed), N * (N - 1) / 2);
}

TEST(ThreadingBackend, TVMBackendParallelLaunchMultipleThreads) {
  // TODO(tulloch) use parameterised tests when available.
  size_t num_jobs_per_thread = 3;
  size_t max_num_threads = 2;

  for (size_t num_threads = 1; num_threads < max_num_threads; ++num_threads) {
    std::vector<std::unique_ptr<std::thread>> ts;
    for (size_t i = 0; i < num_threads; ++i) {
      ts.emplace_back(new std::thread([&]() {
        for (size_t j = 0; j < num_jobs_per_thread; ++j) {
          std::atomic<size_t> acc(0);
          TVMBackendParallelLaunch(atomic_add_task_id, &acc, 0);
          EXPECT_EQ(acc.load(std::memory_order_relaxed), N * (N - 1) / 2);
        }
      }));
    }
    for (auto& t : ts) {
      t->join();
    }
  }
}

TEST(ThreadingBackend, TVMBackendAffinityConfigure) {
  int max_concurrency = tvm::runtime::threading::MaxConcurrency();
  std::vector<std::unique_ptr<std::thread>> ts;
  // Returning as there is only one CPU available.
  if (max_concurrency <= 1) {
    return;
  }
  // Creating two threads to test the 'CPU list affinity' feature.
  const int threads_num = 2;
  // Getting the maximum number of CPUs which are available to each thread.
  const int cpus_num_per_thread = max_concurrency / threads_num;
  // Testing two mode of affinity.,
  std::vector<tvm::runtime::threading::ThreadGroup::AffinityMode> modes = {
      tvm::runtime::threading::ThreadGroup::kSpecifyOneCorePerThread,
      tvm::runtime::threading::ThreadGroup::kSpecifyThreadShareAllCore};
  for (auto mode : modes) {
    for (int thread_pool_idx = 0; thread_pool_idx < threads_num; thread_pool_idx++) {
      ts.emplace_back(new std::thread(
          [&](int thread_pool_index, int sys_max_concurrency,
              tvm::runtime::threading::ThreadGroup::AffinityMode affinity_mode) {
            std::atomic<size_t> acc(0);
            AffinityCheck ac(thread_pool_index, sys_max_concurrency, &acc);
            std::vector<unsigned int> cpus;
            LOG(INFO) << affinity_mode << std::endl;
            for (int k = 0; k < cpus_num_per_thread; k++) {
              cpus.push_back(thread_pool_index * cpus_num_per_thread + k);
            }
            tvm::runtime::threading ::Configure(affinity_mode, 0, cpus);
            TVMBackendParallelLaunch(affinity_check_task_id, &ac, 0);
            EXPECT_EQ(ac.GetComputeResult(), N * (N - 1) / 2);
            EXPECT_EQ(ac.VerifyAffinity(cpus), true);
          },
          thread_pool_idx, max_concurrency, mode));
    }
  }
  for (auto& t : ts) {
    t->join();
  }
}

TEST(ThreadingBackend, TVMBackendParallelForWithThreadingBackend) {
  int n = 100;
  std::vector<int> vec(/*size=*/n, /*value=*/0);
  tvm::runtime::parallel_for_with_threading_backend([&vec](int i) { vec[i] = i; }, 0, n);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(vec[i], i);
  }
}
