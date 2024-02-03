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

#include <dlpack/dlpack.h>
#include <thrust/detail/caching_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <functional>
#include <vector>

#include "../../cuda/cuda_common.h"

namespace tvm {
namespace contrib {

using namespace runtime;

auto get_thrust_exec_policy() {
  return thrust::cuda::par_nosync(thrust::detail::single_device_tls_caching_allocator())
      .on(GetCUDAStream());
}

// Performs sorting along axis -1 and returns both sorted values and indices.
template <typename DataType, typename IndicesType>
void thrust_sort(DLTensor* input, DLTensor* out_values, DLTensor* out_indices, bool is_ascend,
                 int n_values) {
  thrust::device_ptr<DataType> data_ptr(static_cast<DataType*>(input->data));
  thrust::device_ptr<DataType> values_ptr(static_cast<DataType*>(out_values->data));
  thrust::device_ptr<IndicesType> indices_ptr(static_cast<IndicesType*>(out_indices->data));

  auto policy = get_thrust_exec_policy();

  size_t size = 1;
  for (int i = 0; i < input->ndim; ++i) {
    size *= input->shape[i];
  }
  thrust::copy(policy, data_ptr, data_ptr + size, values_ptr);

  if (size == static_cast<size_t>(input->shape[input->ndim - 1])) {
    // A fast path for single segment case
    thrust::sequence(indices_ptr, indices_ptr + n_values);
    if (is_ascend) {
      thrust::sort_by_key(policy, values_ptr, values_ptr + n_values, indices_ptr);
    } else {
      thrust::sort_by_key(policy, values_ptr, values_ptr + n_values, indices_ptr,
                          thrust::greater<DataType>());
    }
  } else {
    // segmented sort by key
    // Follow the back-to-back stable_sort_by_key strategy explained below
    // https://groups.google.com/g/thrust-users/c/BoLsxO6b4FY
    thrust::device_vector<int64_t> argsort_order(size);
    thrust::sequence(argsort_order.begin(), argsort_order.end());

    // First, sort values and store the sorted order in argsort_order.
    if (is_ascend) {
      thrust::stable_sort_by_key(policy, values_ptr, values_ptr + size, argsort_order.begin());
    } else {
      thrust::stable_sort_by_key(policy, values_ptr, values_ptr + size, argsort_order.begin(),
                                 thrust::greater<DataType>());
    }

    // The following is to create the indices array 0, 1, 2, 0, 1, 2 ... 0, 1, 2
    // without materializing it
    auto counting_iter = thrust::counting_iterator<int64_t>(0);
    auto linear_index_to_sort_axis_index = [n_values] __host__ __device__(int64_t i) {
      return i % n_values;
    };  // NOLINT(*)
    auto init_indices_iter =
        thrust::make_transform_iterator(counting_iter, linear_index_to_sort_axis_index);

    // This will reorder indices 0, 1, 2 ... in the sorted order of values_ptr
    thrust::gather(policy, argsort_order.begin(), argsort_order.end(), init_indices_iter,
                   indices_ptr);

    thrust::device_vector<int> segment_ids(size);
    auto linear_index_to_segment_id = [n_values] __host__ __device__(int64_t i) {
      return i / n_values;
    };  // NOLINT(*)
    // We also reorder segment indices 0, 0, 0, 1, 1, 1 ... in the order of values_ptr
    thrust::transform(policy, argsort_order.begin(), argsort_order.end(), segment_ids.begin(),
                      linear_index_to_segment_id);

    // The second sort key-ed by segment_ids would bring segment_ids back to 0, 0, 0, 1, 1, 1 ...
    // values_ptr and indices_ptr will also be sorted in the order of segmend_ids above
    // Since sorting has been done in a stable way, relative orderings of values and indices
    // in the segment do not change and hence they remain sorted.
    auto key_val_zip = thrust::make_zip_iterator(thrust::make_tuple(values_ptr, indices_ptr));
    thrust::stable_sort_by_key(policy, segment_ids.begin(), segment_ids.end(), key_val_zip);
  }
}

void thrust_sort_common(DLTensor* input, DLTensor* values_out, DLTensor* indices_out,
                        bool is_ascend, int sort_len, std::string data_dtype,
                        std::string out_dtype) {
  if (data_dtype == "float32") {
    if (out_dtype == "int32") {
      thrust_sort<float, int32_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<float, int64_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<float, float>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<float, double>(input, values_out, indices_out, is_ascend, sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "float64") {
    if (out_dtype == "int32") {
      thrust_sort<double, int32_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<double, int64_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<double, float>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<double, double>(input, values_out, indices_out, is_ascend, sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "int32") {
    if (out_dtype == "int32") {
      thrust_sort<int32_t, int32_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<int32_t, int64_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<int32_t, float>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<int32_t, double>(input, values_out, indices_out, is_ascend, sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else if (data_dtype == "int64") {
    if (out_dtype == "int32") {
      thrust_sort<int64_t, int32_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "int64") {
      thrust_sort<int64_t, int64_t>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float32") {
      thrust_sort<int64_t, float>(input, values_out, indices_out, is_ascend, sort_len);
    } else if (out_dtype == "float64") {
      thrust_sort<int64_t, double>(input, values_out, indices_out, is_ascend, sort_len);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype;
    }
  } else {
    LOG(FATAL) << "Unsupported input dtype: " << data_dtype;
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.thrust.sort").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_GE(args.num_args, 4);
  DLTensor* input = args[0];
  DLTensor* values_out = args[1];
  DLTensor* indices_out = args[2];
  bool is_ascend = args[3];

  auto data_dtype = DLDataType2String(input->dtype);
  auto out_dtype = DLDataType2String(indices_out->dtype);

  int n_values = input->shape[input->ndim - 1];
  thrust_sort_common(input, values_out, indices_out, is_ascend, n_values, data_dtype, out_dtype);
});

template <typename KeyType, typename ValueType>
void thrust_stable_sort_by_key(DLTensor* keys_in, DLTensor* values_in, DLTensor* keys_out,
                               DLTensor* values_out, bool for_scatter) {
  const auto size = keys_in->shape[0];
  thrust::device_ptr<KeyType> keys_in_ptr(static_cast<KeyType*>(keys_in->data));
  thrust::device_ptr<ValueType> values_in_ptr(static_cast<ValueType*>(values_in->data));
  thrust::device_ptr<KeyType> keys_out_ptr(static_cast<KeyType*>(keys_out->data));
  thrust::device_ptr<ValueType> values_out_ptr(static_cast<ValueType*>(values_out->data));

  auto policy = get_thrust_exec_policy();

  if (for_scatter) {
    thrust::transform(policy, keys_in_ptr, keys_in_ptr + size, keys_out_ptr,
                      [size] __device__(KeyType k) {
                        if (k < 0) return k + static_cast<KeyType>(size);
                        return k;
                      });
  } else {
    thrust::copy(policy, keys_in_ptr, keys_in_ptr + size, keys_out_ptr);
  }
  thrust::copy(policy, values_in_ptr, values_in_ptr + size, values_out_ptr);

  thrust::stable_sort_by_key(policy, keys_out_ptr, keys_out_ptr + size, values_out_ptr);
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
        } else if (value_dtype == "int64") {
          thrust_stable_sort_by_key<int, int64_t>(keys_in, values_in, keys_out, values_out,
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
        } else if (value_dtype == "int64") {
          thrust_stable_sort_by_key<int64_t, int64_t>(keys_in, values_in, keys_out, values_out,
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
        } else if (value_dtype == "int64") {
          thrust_stable_sort_by_key<float, int64_t>(keys_in, values_in, keys_out, values_out,
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

template <typename InType, typename OutType>
void thrust_scan(DLTensor* data, DLTensor* output, bool exclusive) {
  thrust::device_ptr<InType> data_ptr(static_cast<InType*>(data->data));
  thrust::device_ptr<OutType> output_ptr(static_cast<OutType*>(output->data));
  const auto scan_size = data->shape[data->ndim - 1];

  if (scan_size == 0) return;

  size_t size = 1;
  for (int i = 0; i < data->ndim; ++i) size *= data->shape[i];

  const bool need_cast = std::is_same<InType, OutType>::value == false;

  auto data_cast_ptr = thrust::make_transform_iterator(
      data_ptr, [] __host__ __device__(InType v) { return static_cast<OutType>(v); });  // NOLINT(*)

  auto policy = get_thrust_exec_policy();

  if (size == static_cast<size_t>(data->shape[data->ndim - 1])) {
    if (exclusive && need_cast) {
      thrust::exclusive_scan(policy, data_cast_ptr, data_cast_ptr + scan_size, output_ptr);
    } else if (exclusive && !need_cast) {
      thrust::exclusive_scan(policy, data_ptr, data_ptr + scan_size, output_ptr);
    } else if (!exclusive && need_cast) {
      thrust::inclusive_scan(policy, data_cast_ptr, data_cast_ptr + scan_size, output_ptr);
    } else {
      thrust::inclusive_scan(policy, data_ptr, data_ptr + scan_size, output_ptr);
    }
  } else {
    // Use thrust segmented scan to compute scan on the inner most axis
    // data->shape[0] * data->shape[1] * ... * data->shape[ndim - 2] scans are
    // computed in parallel

    // This is for constructing a sequence 0, 0, 0,...,1, 1, 1,...,2, 2, 2,...,
    // without materializing the sequence vector
    auto counting_iter = thrust::counting_iterator<size_t>(0);
    // Without __host__ annotation, cub crashes
    auto linear_index_to_scan_key = [scan_size] __host__ __device__(size_t i) {
      return i / scan_size;
    };  // NOLINT(*)
    auto key_iter = thrust::make_transform_iterator(counting_iter, linear_index_to_scan_key);

    if (exclusive && need_cast) {
      thrust::exclusive_scan_by_key(policy, key_iter, key_iter + size, data_cast_ptr, output_ptr);
    } else if (exclusive && !need_cast) {
      thrust::exclusive_scan_by_key(policy, key_iter, key_iter + size, data_ptr, output_ptr);
    } else if (!exclusive && need_cast) {
      thrust::inclusive_scan_by_key(policy, key_iter, key_iter + size, data_cast_ptr, output_ptr);
    } else {
      thrust::inclusive_scan_by_key(policy, key_iter, key_iter + size, data_ptr, output_ptr);
    }
  }
}

TVM_REGISTER_GLOBAL("tvm.contrib.thrust.sum_scan")
.set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK(args.num_args == 3 || args.num_args == 2);
  DLTensor* data = args[0];
  DLTensor* output = args[1];
  bool exclusive = false;

  if (args.num_args == 3) {
    exclusive = args[2];
  }

  auto in_dtype = DLDataType2String(data->dtype);
  auto out_dtype = DLDataType2String(output->dtype);

  if (in_dtype == "bool") {
    if (out_dtype == "int32") {
      thrust_scan<bool, int>(data, output, exclusive);
    } else if (out_dtype == "int64") {
      thrust_scan<bool, int64_t>(data, output, exclusive);
    } else if (out_dtype == "float32") {
      thrust_scan<bool, float>(data, output, exclusive);
    } else if (out_dtype == "float64") {
      thrust_scan<bool, double>(data, output, exclusive);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype
                 << ". Supported output dtypes are int32, int64, float32, and float64";
    }
  } else if (in_dtype == "int32") {
    if (out_dtype == "int32") {
      thrust_scan<int, int>(data, output, exclusive);
    } else if (out_dtype == "int64") {
      thrust_scan<int, int64_t>(data, output, exclusive);
    } else if (out_dtype == "float32") {
      thrust_scan<int, float>(data, output, exclusive);
    } else if (out_dtype == "float64") {
      thrust_scan<int, double>(data, output, exclusive);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype
                 << ". Supported output dtypes are int32, int64, float32, and float64";
    }
  } else if (in_dtype == "int64") {
    if (out_dtype == "int64") {
      thrust_scan<int64_t, int64_t>(data, output, exclusive);
    } else if (out_dtype == "float32") {
      thrust_scan<int64_t, float>(data, output, exclusive);
    } else if (out_dtype == "float64") {
      thrust_scan<int64_t, double>(data, output, exclusive);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype
                 << ". Supported output dtypes are int64, float32, and float64";
    }
  } else if (in_dtype == "float32") {
    if (out_dtype == "float32") {
      thrust_scan<float, float>(data, output, exclusive);
    } else if (out_dtype == "float64") {
      thrust_scan<float, double>(data, output, exclusive);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype
                 << ". Supported output dtypes are float32, and float64";
    }
  } else if (in_dtype == "float64") {
    if (out_dtype == "float64") {
      thrust_scan<double, double>(data, output, exclusive);
    } else {
      LOG(FATAL) << "Unsupported output dtype: " << out_dtype
                 << ". Supported output dtype is float64";
    }
  } else {
    LOG(FATAL) << "Unsupported input dtype: " << in_dtype
               << ". Supported input dtypes are bool, int32, int64, float32, and float64";
  }
});

}  // namespace contrib
}  // namespace tvm
