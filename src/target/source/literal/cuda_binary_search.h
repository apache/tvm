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
 * \file cuda_binary_search.h
 * \brief Binary search function definition for cuda codegen.
 */
#ifndef TVM_TARGET_SOURCE_LITERAL_CUDA_BINARY_SEARCH_H_
#define TVM_TARGET_SOURCE_LITERAL_CUDA_BINARY_SEARCH_H_

static constexpr const char* _cuda_binary_search_def = R"(
template <typename DType>
__forceinline__ __device__ int32_t __lower_bound(
    const DType* __restrict__ arr,
    DType val,
    int32_t l,
    int32_t r) {
  int32_t low = l - 1, high = r;
  /* loop invariant: low < mid < high, arr[low] < val, arr[high] >= val */
  while (low + 1 < high) {
    int32_t mid = (low + high) >> 1;
    if (arr[mid] < val) {
      low = mid;
    } else {
      high = mid;
    }
  }
  // high = low + 1, arr[low] < val, arr[high] >= val
  return high;
}

template <typename DType>
__forceinline__ __device__ int32_t __upper_bound(
    const DType* __restrict__ arr,
    DType val,
    int32_t l,
    int32_t r) {
  int32_t low = l - 1, high = r;
  /* loop invariant: low < mid < high, arr[low] < val, arr[high] > val */
  while (low + 1 < high) {
    int32_t mid = (low + high) >> 1;
    if (arr[mid] > val) {
      high = mid;
    } else {
      low = mid;
    }
  }
  // high = low + 1, arr[low] <= val, arr[high] > val
  return high;
}
)";

#endif  // TVM_TARGET_SOURCE_LITERAL_CUDA_BINARY_SEARCH_H_
