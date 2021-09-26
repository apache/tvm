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
 * \file parallel_for.h
 * \brief An implementation to run loop in parallel.
 */
#ifndef TVM_SUPPORT_PARALLEL_FOR_H_
#define TVM_SUPPORT_PARALLEL_FOR_H_

#include <tvm/runtime/c_runtime_api.h>

#include <functional>
#include <vector>

namespace tvm {
namespace support {

using PartitionerFuncType = std::function<std::vector<std::vector<int>>(int, int, int, int)>;

/*!
 * \brief A partitioner to split the task to each thread in Round-robin manner.
 * \param begin The start index of this parallel loop(inclusive).
 * \param end The end index of this parallel loop(exclusive).
 * \param step The traversal step to the index.
 * \param num_threads The number of threads(the number of tasks to be partitioned to).
 * \return A list with `num_threads` elements, and each is a list of integers indicating the loop
 * indexes for the corresponding thread to process.
 */
TVM_DLL std::vector<std::vector<int>> rr_partitioner(int begin, int end, int step, int num_threads);

/*!
 * \brief A runtime api provided to run the task function in parallel.
 * e.g. A for loop:
 *   for (int i = 0; i < 10; i++) {
 *     a[i] = i;
 *   }
 * should work the same as:
 *   parallel_for(0, 10, [&a](int index) {
 *     a[i] = i;
 *   });
 * \param begin The start index of this parallel loop(inclusive).
 * \param end The end index of this parallel loop(exclusive).
 * \param f The task function to be executed. Assert to take an int index as input with no output.
 * \param step The traversal step to the index.
 * \param partitioner A partition function to split tasks to different threads. Use Round-robin
 * partitioner by default.
 * \note 1. Currently do not support nested parallel_for; 2. The order of execution in each thread
 * is not guaranteed, the for loop task should be thread independent and thread safe.
 */
TVM_DLL void parallel_for(int begin, int end, const std::function<void(int)>& f, int step = 1,
                          const PartitionerFuncType partitioner = rr_partitioner);

/*!
 * \brief An API to launch fix amount of threads to run the specific functor in parallel.
 * Different from `parallel_for`, the partition is determined dynamically on the fly,
 * i.e. any time when a thread is idle, it fetches the next task to run.
 * The behavior is similar to dynamic scheduling in OpenMP:
 *
 *   \#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
 *   for (int i = 0; i < 10; i++) {
 *     a[i] = i;
 *   }
 *
 * \param begin The start index of this parallel loop (inclusive).
 * \param end The end index of this parallel loop (exclusive).
 * \param num_threads The number of threads to be used.
 * \param f The task function to be executed. Takes the thread index and the task index as
 * input with no output.
 * \note `step` support is left for future work.
 */
TVM_DLL void parallel_for_dynamic(int begin, int end, int num_threads,
                                  const std::function<void(int thread_id, int task_id)>& f);
}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_PARALLEL_FOR_H_
