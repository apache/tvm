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
  DLTensor *sort_num = args[1];
  DLTensor *output = args[2];
  int32_t axis = args[3];
  bool is_descend = args[4];

  auto dtype = input->dtype;
  auto data_ptr = static_cast<float *>(input->data);
  auto sort_num_ptr = static_cast<int32_t *>(sort_num->data);
  std::vector<std::pair<int32_t, float>> sorter;
  std::vector<int64_t> pos_multiplier {1};
  int64_t total_num_sort = 1;

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

  for (int i = input->ndim - 1; i >= 0; --i) {
    if (i > 0) {
      pos_multiplier.insert(pos_multiplier.begin(),
                            pos_multiplier.front() * input->shape[i]);
    }
    if (i != axis) {
      total_num_sort *= input->shape[i];
    }
  }

  for (int64_t i = 0; i < total_num_sort; ++i) {
    sorter.clear();
    int32_t current_sort_num = *(sort_num_ptr + i);
    /*
       Store current_sort_num elements into sorter.
       Given a flatten index i of reduced shape (d0, d1, ..., dk, d(k+1), ...,
       d(n-1)) and the index j of sorting axis, calculate the corresponding flatten
       index in full shape (d0, d1, ..., dk, sort_axis, d(k+1), ..., d(n-1)).
       First, get the normal index (i0, i1, ..., i(n-1)) of i in reduced shape. The
       normal index of j in full shape would be (i0, i1, ..., ik, j, i(k+1), ...,
       i(n+1)). Multiply with pos_multiplier, we can get flatten index.
    */
    int64_t reduced_normal_idx;
    int64_t reduced_flatten_idx = i;
    int64_t full_flatten_base_idx = 0;
    // Restore normal index for full shape except sorting axis.
    for (int32_t j = 0; j < current_sort_num; ++j) {
      for (int32_t k = 0; k < pos_multiplier.size(); ++k) {
        if (k == axis) {
          continue;
        }
        int64_t current_multiplier = k < axis ? pos_multiplier[k] /
          input->shape[axis] : pos_multiplier[k];
        reduced_normal_idx = reduced_flatten_idx / current_multiplier;
        reduced_flatten_idx %= current_multiplier;
        full_flatten_base_idx += reduced_normal_idx * pos_multiplier[k];
      }
    }
    // Restore complete normal index for full shape and fill sorter.
    for (int32_t j = 0; j < current_sort_num; ++j) {
        auto current_pos = full_flatten_base_idx + j * pos_multiplier[axis];
        sorter.emplace_back(std::pair<int32_t, float>(j, *(data_ptr
                                                           + current_pos)));
    }
    if (is_descend) {
      std::stable_sort(sorter.begin(), sorter.end(), CompareDescend<float>);
    } else {
      std::stable_sort(sorter.begin(), sorter.end(), CompareAscend<float>);
    }
    for (int32_t j = 0; j < input->shape[axis]; ++j) {
      *(static_cast<int32_t *>(output->data) + full_flatten_base_idx +
        j * pos_multiplier[axis]) = j < sorter.size() ? sorter[j].first : j;
    }
  }
});

}  // namespace contrib
}  // namespace tvm
