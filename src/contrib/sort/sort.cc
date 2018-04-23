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
// Sort a 3-D tensor for axis 1.
// Number of element to be sorted for each sample can be specified
// by num_elem tensor.
// Index of value to be used for sorting is defined by sort_value_index.
// Return index of sorted elements in each sample.
TVM_REGISTER_GLOBAL("tvm.contrib.sort.argsort")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *input = args[0];
  DLTensor *output = args[1];
  int32_t axis = args[2];
  bool is_descend = args[3];

  auto dtype = input->dtype;
  auto data_ptr = static_cast<float *>(input->data);
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

  for (int64_t i = 0; i < num_sort_vec; ++i) {
    sorter.clear();
    std::vector<int64_t> position;
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
    for (int32_t j = 0; j < input->shape[axis]; ++j) {
        auto current_pos = tensor_base_pos + j * sort_axis_base_pos;
        sorter.emplace_back(SortElem<float>(*(data_ptr + current_pos), j));
    }
    if (is_descend) {
      std::stable_sort(sorter.begin(), sorter.end(), compare_descend<float>);
    } else {
      std::stable_sort(sorter.begin(), sorter.end(), compare_ascend<float>);
    }
    for (int j = 0; j < sorter.size(); ++j) {
      *(static_cast<int32_t *>(output->data) + tensor_base_pos +
        j * sort_axis_base_pos) = sorter[j].index;
    }
  }
});

}  // namespace contrib
}  // namespace tvm
