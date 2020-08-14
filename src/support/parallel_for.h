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

#include <functional>

namespace tvm {
namespace support {

/*!
 * \brief A runtime api provided to run the task function in parallel.
 * e.g. A for loop:
 *   for (int i = 0; i < 10; i++) {
 *     std::cout << index << "\n";
 *   }
 * should work the same as:
 *   parallel_for(0, 10, [](int index) {
 *     std::cout << index << "\n";
 *   });
 * \param begin The start index of this parallel loop(inclusive).
 * \param end The end index of this parallel loop(exclusive).
 * \param f The task function to be excuted. Assert to take an int index as input with no output.
 * \param step The traversal step to the index.
 */
void parallel_for(int begin, int end, const std::function<void(int)>& f, int step = 1);

}  // namespace support
}  // namespace tvm

#endif  // TVM_SUPPORT_PARALLEL_FOR_H_
