/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use standard C library call.
 */

#include "sort.h"
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include <algorithm>
#include <vector>

namespace tvm {
namespace contrib {

using namespace runtime;

template<>
bool SortElem<float>::is_descend = true;

// C library sort.
// Sort a 3-D tensor for axis 1.
// Number of element to be sorted for each sample can be specified
// by num_elem tensor.
// Index of value to be used for sorting is defined by sort_value_index.
// Return index of sorted elements in each sample.
TVM_REGISTER_GLOBAL("tvm.contrib.sort.stable_sort")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *input = args[0];
  DLTensor *num_elem = args[1];
  DLTensor *output = args[2];
  int32_t batch_size = args[3];
  int32_t sort_value_index = args[4];
  bool is_descend = args[5];

  CHECK_EQ(input->ndim, 3) << "Currently only support sorting 3-D tensor. "
      "The first dimension should be batch axis.";
  CHECK_EQ(output->ndim, 2) << "Ouput should be a 2-D tensor contains index "
      "of sorted elements. The first dimension should be batch axis.";
  CHECK_EQ(num_elem->ndim, 1) << "num_elem should be a 1-D tensor.";

  auto data_ptr = static_cast<float *>(input->data);
  auto num_elem_ptr = static_cast<int32_t *>(num_elem->data);
  SortElem<float>::is_descend = is_descend;
  std::vector<SortElem<float>> sorter;

  for (int32_t i = 0; i < batch_size; ++i) {
    sorter.clear();
    int32_t num_elem_val = *(num_elem_ptr + i);
    for (int32_t j = 0; j < num_elem_val; ++j) {
      sorter.emplace_back(SortElem<float>(*(data_ptr +
        i * input->shape[1] * input->shape[2] +
        j * input->shape[2] + sort_value_index), j));
    }
    std::stable_sort(sorter.begin(), sorter.end());

    for (int32_t j = 0; j < num_elem_val; ++j) {
      *(static_cast<int32_t *>(output->data) + i * input->shape[1] + j) =
        sorter[j].index;
    }
  }
});

}  // namespace contrib
}  // namespace tvm
