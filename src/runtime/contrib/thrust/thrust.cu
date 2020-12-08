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
 * \file Use external Thrust library call
 */

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <tvm/runtime/registry.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>
#include <functional>

namespace tvm {
namespace contrib {

using namespace runtime;

// Performs sorting along axis -1 and returns both sorted values and indices.
template<typename DataType, typename IndicesType>
void thrust_sort(DLTensor* input,
                 DLTensor* out_values,
                 DLTensor* out_indices,
                 bool is_ascend,
                 const std::function<int(int)> &get_sort_len) {
  thrust::device_ptr<DataType> data_ptr(static_cast<DataType *>(input->data));
  thrust::device_ptr<DataType> values_ptr(static_cast<DataType *>(out_values->data));
  thrust::device_ptr<IndicesType> indices_ptr(static_cast<IndicesType *>(out_indices->data));

  int n_values = input->shape[input->ndim - 1];
  int n_iter = 1;
  for (int i = 0; i < input->ndim - 1; ++i) {
    n_iter *= input->shape[i];
  }

  thrust::copy(data_ptr, data_ptr + n_iter * n_values, values_ptr);

  for (int i = 0 ; i < n_iter; ++i) {
    n_values = get_sort_len(i);
    thrust::sequence(indices_ptr, indices_ptr + n_values);
    if (is_ascend) {
      thrust::sort_by_key(values_ptr, values_ptr + n_values, indices_ptr);
    } else {
      thrust::sort_by_key(values_ptr, values_ptr + n_values, indices_ptr,
                          thrust::greater<DataType>());
    }
    values_ptr += n_values;
    indices_ptr += n_values;
  }
}

void thrust_sort_common(DLTensor* input,
                        DLTensor* values_out,
                        DLTensor* indices_out,
                        bool is_ascend,
                        const std::function<int(int)> &get_sort_len,
                        std::string data_dtype,
                        std::string out_dtype) {
  if (data_dtype == "float32") {
    if (out_dtype == "int32") {
      thrust_sort<float, int32_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<float, int64_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<float, float>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<float, double>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "float64") {
    if (out_dtype == "int32") {
      thrust_sort<double, int32_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<double, int64_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<double, float>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<double, double>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "int32") {
    if (out_dtype == "int32") {
      thrust_sort<int32_t, int32_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<int32_t, int64_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<int32_t, float>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<int32_t, double>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  }  else if (data_dtype == "int64") {
    if (out_dtype == "int32") {
      thrust_sort<int64_t, int32_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<int64_t, int64_t>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<int64_t, float>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<int64_t, double>(input, values_out, indices_out, is_ascend, get_sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else {
    LOG(FATAL) << "Unsupported input dtype: " << data_dtype;
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.thrust.sort_nms")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_GE(args.num_args, 5);
  DLTensor* input = args[0];
  DLTensor* valid_count = args[1];
  DLTensor* values_out = args[2];
  DLTensor* indices_out = args[3];
  bool is_ascend = args[4];

  auto data_dtype = DLDataType2String(input->dtype);
  auto out_dtype = DLDataType2String(indices_out->dtype);

  thrust::device_ptr<int> valid_count_ptr(static_cast<int *>(valid_count->data));
  auto get_sort_len = [&valid_count_ptr](int i) { return valid_count_ptr[i]; };
  thrust_sort_common(input, values_out, indices_out, is_ascend, get_sort_len,
                     data_dtype, out_dtype);
});


TVM_REGISTER_GLOBAL("tvm.contrib.thrust.sort")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_GE(args.num_args, 4);
  DLTensor* input = args[0];
  DLTensor* values_out = args[1];
  DLTensor* indices_out = args[2];
  bool is_ascend = args[3];

  auto data_dtype = DLDataType2String(input->dtype);
  auto out_dtype = DLDataType2String(indices_out->dtype);

  int n_values = input->shape[input->ndim - 1];
  auto get_sort_len = [=](int i) { return n_values; };
  thrust_sort_common(input, values_out, indices_out, is_ascend, get_sort_len,
                     data_dtype, out_dtype);
});

template<typename KeyType, typename ValueType>
void thrust_stable_sort_by_key(DLTensor* keys_in,
                               DLTensor* values_in,
                               DLTensor* keys_out,
                               DLTensor* values_out,
                               bool for_scatter) {
  const auto size = keys_in->shape[0];
  thrust::device_ptr<KeyType> keys_in_ptr(static_cast<KeyType *>(keys_in->data));
  thrust::device_ptr<ValueType> values_in_ptr(static_cast<ValueType *>(values_in->data));
  thrust::device_ptr<KeyType> keys_out_ptr(static_cast<KeyType *>(keys_out->data));
  thrust::device_ptr<ValueType> values_out_ptr(static_cast<ValueType *>(values_out->data));

  if (for_scatter) {
    thrust::transform(keys_in_ptr, keys_in_ptr + size, keys_out_ptr, [size] __device__(KeyType k) {
      if (k < 0) return k + static_cast<KeyType>(size);
      return k;
    });
  } else {
    thrust::copy(keys_in_ptr, keys_in_ptr + size, keys_out_ptr);
  }
  thrust::copy(values_in_ptr, values_in_ptr + size, values_out_ptr);

  thrust::stable_sort_by_key(keys_out_ptr, keys_out_ptr + size, values_out_ptr);
}

TVM_REGISTER_GLOBAL("tvm.contrib.thrust.stable_sort_by_key")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_GE(args.num_args, 5);
  DLTensor* keys_in = args[0];
  DLTensor* values_in = args[1];
  DLTensor* keys_out = args[2];
  DLTensor* values_out = args[3];
  bool for_scatter = args[4];

  auto key_dtype = DLDataType2String(keys_in->dtype);
  auto value_dtype = DLDataType2String(values_in->dtype);

  if (key_dtype == "int32") {
    if (value_dtype == "int32") {
      thrust_stable_sort_by_key<int, int>(keys_in, values_in, keys_out, values_out,
                                          for_scatter);
    } else if (value_dtype == "float32") {
      thrust_stable_sort_by_key<int, float>(keys_in, values_in, keys_out, values_out,
                                            for_scatter);
    } else {
      LOG(FATAL) << "Unsupported value dtype: " << value_dtype;
    }
  } else if (key_dtype == "int64") {
    if (value_dtype == "int32") {
      thrust_stable_sort_by_key<int64_t, int>(keys_in, values_in, keys_out, values_out,
                                              for_scatter);
    } else if (value_dtype == "float32") {
      thrust_stable_sort_by_key<int64_t, float>(keys_in, values_in, keys_out, values_out,
                                                for_scatter);
    } else {
      LOG(FATAL) << "Unsupported value dtype: " << value_dtype;
    }
  } else if (key_dtype == "float32") {
    if (value_dtype == "int32") {
      thrust_stable_sort_by_key<float, int>(keys_in, values_in, keys_out, values_out,
                                            for_scatter);
    } else if (value_dtype == "float32") {
      thrust_stable_sort_by_key<float, float>(keys_in, values_in, keys_out, values_out,
                                              for_scatter);
    } else {
      LOG(FATAL) << "Unsupported value dtype: " << value_dtype;
    }
  } else {
    LOG(FATAL) << "Unsupported key dtype: " << key_dtype;
  }
});

}  // namespace contrib
}  // namespace tvm
