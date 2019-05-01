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
 *  Copyright (c) 2017 by Contributors
 * \file Use standard C library call.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>

namespace tvm {
namespace contrib {

using namespace runtime;

template<typename DType>
bool CompareAscend(const std::pair<int32_t, DType>& lhs,
                   const std::pair<int32_t, DType>& rhs) {
  return lhs.second < rhs.second;
}

template<typename DType>
bool CompareDescend(const std::pair<int32_t, DType>& lhs,
                    const std::pair<int32_t, DType>& rhs) {
  return lhs.second > rhs.second;
}


// Argsort implemented C library sort for nms.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
TVM_REGISTER_GLOBAL("tvm.contrib.sort.argsort_nms")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *input = args[0];
  DLTensor *sort_num = args[1];
  DLTensor *output = args[2];
  int32_t axis = args[3];
  bool is_ascend = args[4];

  auto dtype = input->dtype;
  auto data_ptr = static_cast<float *>(input->data);
  auto sort_num_ptr = static_cast<int32_t *>(sort_num->data);
  std::vector<std::pair<int32_t, float>> sorter;
  int64_t axis_mul_before = 1;
  int64_t axis_mul_after = 1;

  if (axis < 0) {
    axis = input->ndim + axis;
  }

  // Currently only supports input dtype to be float32.
  CHECK_EQ(dtype.code, 2) << "Currently only supports input dtype "
      "to be float32.";
  CHECK_EQ(dtype.bits, 32) << "Currently only supports input dtype "
      "to be float32.";
  CHECK_LT(axis, input->ndim) << "Axis out of boundary for "
      "input ndim " << input->ndim;

  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }

  for (int64_t i = 0 ; i < axis_mul_before; ++i) {
    for (int64_t j = 0 ; j < axis_mul_after; ++j) {
      sorter.clear();
      int32_t current_sort_num = *(sort_num_ptr + i * axis_mul_after + j);
      int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
      for (int64_t k = 0; k < current_sort_num; ++k) {
        int64_t full_idx = base_idx + k * axis_mul_after;
        sorter.emplace_back(std::make_pair(k, *(data_ptr + full_idx)));
      }
      if (is_ascend) {
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<float>);
      } else {
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<float>);
      }
      for (int32_t k = 0; k < input->shape[axis]; ++k) {
        *(static_cast<int32_t *>(output->data) + base_idx + k * axis_mul_after)
            = k < static_cast<int32_t>(sorter.size()) ? sorter[k].first : k;
      }
    }
  }
});


// Argsort implemented C library sort.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
TVM_REGISTER_GLOBAL("tvm.contrib.sort.argsort")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *input = args[0];
  DLTensor *output = args[1];
  int32_t axis = args[2];
  bool is_ascend = args[3];

  auto dtype = input->dtype;
  auto data_ptr = static_cast<float *>(input->data);
  std::vector<std::pair<float, float>> sorter;
  int64_t axis_mul_before = 1;
  int64_t axis_mul_after = 1;

  if (axis < 0) {
    axis = input->ndim + axis;
  }

  // Currently only supports input dtype to be float32.
  CHECK_EQ(dtype.code, 2) << "Currently only supports input dtype "
      "to be float32.";
  CHECK_EQ(dtype.bits, 32) << "Currently only supports input dtype "
      "to be float32.";
  CHECK_LT(axis, input->ndim) << "Axis out of boundary for "
      "input ndim " << input->ndim;

  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }

  int32_t current_sort_num = input->shape[axis];
  for (int64_t i = 0 ; i < axis_mul_before; ++i) {
    for (int64_t j = 0 ; j < axis_mul_after; ++j) {
      sorter.clear();
      int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
      for (int64_t k = 0; k < current_sort_num; ++k) {
        int64_t full_idx = base_idx + k * axis_mul_after;
        sorter.emplace_back(std::make_pair(k, *(data_ptr + full_idx)));
      }
      if (is_ascend) {
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<float>);
      } else {
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<float>);
      }
      for (int32_t k = 0; k < input->shape[axis]; ++k) {
        *(static_cast<float *>(output->data) + base_idx + k * axis_mul_after)
            = k < static_cast<float>(sorter.size()) ? sorter[k].first : k;
      }
    }
  }
});

}  // namespace contrib
}  // namespace tvm
