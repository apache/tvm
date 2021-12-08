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

#include <gtest/gtest.h>
#include <tvm/runtime/c_backend_api.h>

#include <atomic>
#include <memory>
#include <thread>

constexpr size_t N = 128;

static FTVMParallelLambda atomic_add_task_id = [](int task_id, TVMParallelGroupEnv* penv,
                                                  void* cdata) -> int {
  auto* data = reinterpret_cast<std::atomic<size_t>*>(cdata);
  const size_t N_per_task = (N + penv->num_task - 1) / penv->num_task;
  for (size_t i = task_id * N_per_task; i < N && i < (task_id + 1) * N_per_task; ++i) {
    data->fetch_add(i, std::memory_order_relaxed);
  }
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
