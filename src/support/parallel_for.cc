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
 * \file parallel_for.cc
 * \brief An implementation to run loop in parallel.
 */
#include <tvm/runtime/logging.h>
#include <tvm/support/parallel_for.h>

#include <future>
#include <thread>
#include <utility>
#include <vector>

namespace tvm {
namespace support {

std::vector<std::vector<int>> rr_partitioner(int begin, int end, int step, int num_threads) {
  int total_task_count = (end - begin) / step;
  ICHECK_GE(total_task_count, 0) << "Infinite loop condition with begin: " << begin
                                 << " end: " << end << " step: " << step;
  std::vector<std::vector<int>> ret;
  ret.reserve(num_threads);
  for (size_t thread = 0; begin < end; begin += step, thread = (thread + 1) % num_threads) {
    if (thread >= ret.size()) {
      ret.push_back(std::vector<int>());
    }
    ret[thread].push_back(begin);
  }
  return ret;
}

void parallel_for(int begin, int end, const std::function<void(int)>& f, int step,
                  const PartitionerFuncType partitioner) {
  static bool GLOBAL_PARALLEL_FOR_FLAG{false};
  static std::mutex M_GLOBAL_PARALLEL_FOR_FLAG;
  {
    std::unique_lock<std::mutex> l(M_GLOBAL_PARALLEL_FOR_FLAG);
    ICHECK(!GLOBAL_PARALLEL_FOR_FLAG) << "There's another parallel_for running. Maybe you're "
                                      << "currently inside another parallel_for loop.";
    GLOBAL_PARALLEL_FOR_FLAG = true;
  }

  int default_num_threads = std::thread::hardware_concurrency();
  const auto& run_partitions = partitioner(begin, end, step, default_num_threads);

  std::vector<std::thread> threads;
  threads.reserve(run_partitions.size());
  std::vector<std::future<void>> res_vec;
  res_vec.reserve(run_partitions.size());
  for (const auto& run_partition : run_partitions) {
    std::packaged_task<void(const std::vector<int>&, const std::function<void(int)>&)> task(
        [](const std::vector<int>& run_partition, const std::function<void(int)>& f) {
          for (const auto& i : run_partition) {
            f(i);
          }
        });
    res_vec.emplace_back(task.get_future());
    threads.emplace_back(std::move(task), run_partition, f);
  }

  for (auto&& thread : threads) {
    thread.join();
  }
  {
    std::unique_lock<std::mutex> l(M_GLOBAL_PARALLEL_FOR_FLAG);
    ICHECK(GLOBAL_PARALLEL_FOR_FLAG);
    GLOBAL_PARALLEL_FOR_FLAG = false;
  }
  try {
    for (auto&& i : res_vec) {
      i.get();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "Parallel_for error with " << e.what();
  }
}

void parallel_for_dynamic(int begin, int end, int num_threads,
                          const std::function<void(int thread_id, int task_id)>& f) {
  // Step 1. Sanity checks
  if (begin == end) {
    return;
  }
  CHECK_LE(begin, end) << "ValueError: The interval [begin, end) requires `begin <= end`";
  CHECK_GT(num_threads, 0) << "ValueError: `num_threads` should be positive";
  // Step 2. Launch threads
  // Step 2.1. Launch worker 1 to worker `num_threads - 1`
  std::atomic<int> counter{begin};
  std::vector<std::future<void>> futures;
  std::vector<std::thread> threads;
  futures.reserve(num_threads - 1);
  threads.reserve(num_threads - 1);
  auto worker = [end, &counter, &f](int thread_id) -> void {
    for (int task_id; (task_id = counter++) < end;) {
      f(thread_id, task_id);
    }
  };
  for (int thread_id = 1; thread_id < num_threads; ++thread_id) {
    std::packaged_task<void(int)> task(worker);
    futures.emplace_back(task.get_future());
    threads.emplace_back(std::move(task), thread_id);
  }
  // Step 2.2. Launch worker 0 inplace
  try {
    worker(0);
  } catch (const std::exception& e) {
    for (auto&& thread : threads) {
      thread.join();
    }
    LOG(FATAL) << "RuntimeError: parallel_for_dynamic error with " << e.what();
  }
  // Step 3. Join threads and check exceptions
  for (auto&& thread : threads) {
    thread.join();
  }
  try {
    for (auto&& future : futures) {
      future.get();
    }
  } catch (const std::exception& e) {
    LOG(FATAL) << "RuntimeError: parallel_for_dynamic error with " << e.what();
  }
}

}  // namespace support
}  // namespace tvm
