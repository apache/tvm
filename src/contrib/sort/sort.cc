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
bool CompareAscend(const std::pair<int64_t, DType>& lhs,
                   const std::pair<int64_t, DType>& rhs) {
  return lhs.second < rhs.second;
}

template<typename DType>
bool CompareDescend(const std::pair<int64_t, DType>& lhs,
                    const std::pair<int64_t, DType>& rhs) {
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
      "to be float.";
#if (__ARM_FP16_FORMAT_IEEE != 1)
  CHECK_EQ(dtype.bits, 32) << "Currently only supports input dtype "
      "to be float32.";
#endif
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
#if (__ARM_FP16_FORMAT_IEEE == 1)
        if (dtype.bits == 16) {
          std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<__fp16>);
        } else {
#endif
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<float>);
#if (__ARM_FP16_FORMAT_IEEE == 1)
        }
#endif
      } else {
#if (__ARM_FP16_FORMAT_IEEE == 1)
        if (dtype.bits == 16) {
          std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<__fp16>);
        } else {
#endif
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<float>);
#if (__ARM_FP16_FORMAT_IEEE == 1)
        }
#endif
      }
      for (int32_t k = 0; k < input->shape[axis]; ++k) {
        *(static_cast<int32_t *>(output->data) + base_idx + k * axis_mul_after)
            = k < static_cast<int32_t>(sorter.size()) ? sorter[k].first : k;
      }
    }
  }
});

template<typename DataType, typename OutType>
void argsort(DLTensor* input, DLTensor* output, int32_t axis, bool is_ascend) {
  auto data_ptr = static_cast<DataType *>(input->data);
  auto out_ptr = static_cast<OutType *>(output->data);
  std::vector<std::pair<int64_t, DataType> > sorter;

  int axis_mul_before = 1;
  int axis_mul_after = 1;
  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }

  for (int i = 0 ; i < axis_mul_before; ++i) {
    for (int j = 0 ; j < axis_mul_after; ++j) {
      sorter.clear();
      int64_t base_idx = i * input->shape[axis] * axis_mul_after + j;
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        int64_t full_idx = base_idx + k * axis_mul_after;
        sorter.emplace_back(std::make_pair(k, data_ptr[full_idx]));
      }
      if (is_ascend) {
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<DataType>);
      } else {
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<DataType>);
      }
      for (int64_t k = 0; k < input->shape[axis]; ++k) {
        out_ptr[base_idx + k * axis_mul_after] = static_cast<OutType>(sorter[k].first);
      }
    }
  }
}

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
  if (axis < 0) {
    axis = input->ndim + axis;
  }
  CHECK_LT(axis, input->ndim) << "Axis out of boundary for "
                                 "input ndim " << input->ndim;

  auto data_dtype = TVMType2String(input->dtype);
  auto out_dtype = TVMType2String(output->dtype);

  if (data_dtype == "float32") {
    if (out_dtype == "int32") {
      argsort<float, int32_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "int64") {
      argsort<float, int64_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "float32") {
      argsort<float, float>(input, output, axis, is_ascend);
    } else if (out_dtype == "float64") {
      argsort<float, double>(input, output, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "float64") {
    if (out_dtype == "int32") {
      argsort<double, int32_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "int64") {
      argsort<double, int64_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "float32") {
      argsort<double, float>(input, output, axis, is_ascend);
    } else if (out_dtype == "float64") {
      argsort<double, double>(input, output, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
#if (__ARM_FP16_FORMAT_IEEE == 1)
  } else if (data_dtype == "float16") {
    if (out_dtype == "float16") {
      argsort<__fp16, __fp16>(input, output, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
#endif
  } else if (data_dtype == "int32") {
    if (out_dtype == "int32") {
      argsort<int32_t, int32_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "int64") {
      argsort<int32_t, int64_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "float32") {
      argsort<int32_t, float>(input, output, axis, is_ascend);
    } else if (out_dtype == "float64") {
      argsort<int32_t, double>(input, output, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  }  else if (data_dtype == "int64") {
    if (out_dtype == "int32") {
      argsort<int64_t, int32_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "int64") {
      argsort<int64_t, int64_t>(input, output, axis, is_ascend);
    } else if (out_dtype == "float32") {
      argsort<int64_t, float>(input, output, axis, is_ascend);
    } else if (out_dtype == "float64") {
      argsort<int64_t, double>(input, output, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else {
    LOG(FATAL) << "Unsupported input dtype: " << data_dtype;
  }
});

template<typename DataType, typename IndicesType>
void topk(DLTensor* input,
          DLTensor* out_values,
          DLTensor* out_indices,
          int k,
          int axis,
          bool is_ascend) {
  DataType* data_ptr = static_cast<DataType *>(input->data);
  DataType* values_ptr = (out_values == nullptr) ? nullptr :
          static_cast<DataType *>(out_values->data);
  IndicesType* indices_ptr = (out_indices == nullptr) ? nullptr :
          static_cast<IndicesType *>(out_indices->data);
  std::vector<std::pair<int64_t, DataType> > sorter;

  int axis_mul_before = 1;
  int axis_mul_after = 1;
  for (int i = 0; i < input->ndim; ++i) {
    if (i < axis) {
      axis_mul_before *= input->shape[i];
    } else if (i > axis) {
      axis_mul_after *= input->shape[i];
    }
  }
  if (k < 1) {
    k = input->shape[axis];
  }

  for (int i = 0 ; i < axis_mul_before; ++i) {
    for (int j = 0 ; j < axis_mul_after; ++j) {
      sorter.clear();
      int64_t src_base_idx = i * input->shape[axis] * axis_mul_after + j;
      int64_t dst_base_idx = i * k * axis_mul_after + j;
      for (int64_t kk = 0; kk < input->shape[axis]; ++kk) {
        int64_t full_idx = src_base_idx + kk * axis_mul_after;
        sorter.emplace_back(std::make_pair(kk, data_ptr[full_idx]));
      }
      if (is_ascend) {
        std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<DataType>);
      } else {
        std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<DataType>);
      }
      int64_t cnt = k > 0 ? k : input->shape[axis];
      for (int64_t kk = 0; kk < cnt; ++kk) {
        if (indices_ptr != nullptr) {
          indices_ptr[dst_base_idx + kk * axis_mul_after] =
                  static_cast<IndicesType>(sorter[kk].first);
        }
        if (values_ptr != nullptr) {
          values_ptr[dst_base_idx + kk * axis_mul_after] =
                  static_cast<DataType>(sorter[kk].second);
        }
      }
    }
  }
}

// Argsort implemented C library sort.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
TVM_REGISTER_GLOBAL("tvm.contrib.sort.topk")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  DLTensor* input = args[0];
  DLTensor* values_out = nullptr;
  DLTensor* indices_out = nullptr;
  int k = args[args.num_args - 4];
  int axis = args[args.num_args - 3];
  std::string ret_type = args[args.num_args - 2];
  bool is_ascend = args[args.num_args - 1];
  if (ret_type == "both") {
    values_out = args[1];
    indices_out = args[2];
  } else if (ret_type == "values") {
    values_out = args[1];
  } else if (ret_type == "indices") {
    indices_out = args[1];
  } else {
    LOG(FATAL) << "Unsupported ret type: " << ret_type;
  }
  if (axis < 0) {
    axis = input->ndim + axis;
  }
  CHECK(axis >= 0 && axis < input->ndim) << "Axis out of boundary for input ndim " << input->ndim;

  auto data_dtype = TVMType2String(input->dtype);
  auto out_dtype = (indices_out == nullptr) ? "int64" : TVMType2String(indices_out->dtype);

  if (data_dtype == "float32") {
    if (out_dtype == "int32") {
      topk<float, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "int64") {
      topk<float, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float32") {
      topk<float, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float64") {
      topk<float, double>(input, values_out, indices_out, k, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "float64") {
    if (out_dtype == "int32") {
      topk<double, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "int64") {
      topk<double, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float32") {
      topk<double, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float64") {
      topk<double, double>(input, values_out, indices_out, k, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "int32") {
    if (out_dtype == "int32") {
      topk<int32_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "int64") {
      topk<int32_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float32") {
      topk<int32_t, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float64") {
      topk<int32_t, double>(input, values_out, indices_out, k, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  }  else if (data_dtype == "int64") {
    if (out_dtype == "int32") {
      topk<int64_t, int32_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "int64") {
      topk<int64_t, int64_t>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float32") {
      topk<int64_t, float>(input, values_out, indices_out, k, axis, is_ascend);
    } else if (out_dtype == "float64") {
      topk<int64_t, double>(input, values_out, indices_out, k, axis, is_ascend);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else {
    LOG(FATAL) << "Unsupported input dtype: " << data_dtype;
  }
});

}  // namespace contrib
}  // namespace tvm
