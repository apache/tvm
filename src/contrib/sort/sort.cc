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
struct SortElem {
  DType value;
  int32_t index;

  SortElem(DType v, int32_t i) {
    value = v;
    index = i;
  }
};

template<typename DType>
bool compare_ascend(SortElem<DType> lhs, SortElem<DType> rhs) {
  return lhs.value < rhs.value;
}

template<typename DType>
bool compare_descend(SortElem<DType> lhs, SortElem<DType> rhs) {
  return lhs.value > rhs.value;
}


// Argsort implemented C library sort.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d1, d2, ..., d(k-1), dk, d(k+1), ..., dn)
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
  int64_t num_sort_vec = 1;
  // Currently only supports input dtype to be float32.
  CHECK_EQ(dtype.code, 2) << "Currently only supports input dtype "
      "to be float32.";
  CHECK_EQ(dtype.bits, 32) << "Currently only supports input dtype "
      "to be float32.";
  std::vector<SortElem<float>> sorter;
  std::vector<int64_t> non_sort_axis;
  if (axis == -1) {
    axis = 0;
  }
  for (int i = 0; i < input->ndim; ++i) {
    if (i != axis) {
      num_sort_vec *= input->shape[i];
      non_sort_axis.push_back(input->shape[i]);
    }
  }

  for (int i = 0; i < non_sort_axis.size(); ++i) {
    CHECK_EQ(non_sort_axis[i], sort_num->shape[i])
      << "num_sort shape inconsistent";
  }

  for (int64_t i = 0; i < num_sort_vec; ++i) {
    sorter.clear();
    std::vector<int64_t> position;
    auto current_sort_num = *(sort_num_ptr + i);
    auto pos_mul = i;
    auto pos_len = num_sort_vec;

    for (int j = 0; j < non_sort_axis.size(); ++j) {
      pos_len /= non_sort_axis[j];
      position.push_back(pos_mul / pos_len);
      pos_mul %= pos_len;
    }
    position.insert(position.begin() + axis, 0);
    int64_t tensor_base_pos = 0;
    int64_t sort_axis_base_pos = 0;
    pos_len = num_sort_vec * input->shape[axis];
    for (int j = 0; j < position.size(); ++j) {
      pos_len /= input->shape[j];
      tensor_base_pos += position[j] * pos_len;
      if (j == axis) {
        sort_axis_base_pos = pos_len;
      }
    }
    for (int32_t j = 0; j < current_sort_num; ++j) {
        auto current_pos = tensor_base_pos + j * sort_axis_base_pos;
        sorter.emplace_back(SortElem<float>(*(data_ptr + current_pos), j));
    }
    if (is_descend) {
      std::stable_sort(sorter.begin(), sorter.end(), compare_descend<float>);
    } else {
      std::stable_sort(sorter.begin(), sorter.end(), compare_ascend<float>);
    }
    for (int32_t j = 0; j < input->shape[axis]; ++j) {
      *(static_cast<int32_t *>(output->data) + tensor_base_pos +
        j * sort_axis_base_pos) = j < sorter.size() ? sorter[j].index : j;
    }
  }
});

}  // namespace contrib
}  // namespace tvm
