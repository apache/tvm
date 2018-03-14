/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use standard C library call.
 */
#include <algorithm>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dmlc/logging.h>
#include "utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

template <>
bool SortElem<float>::is_descend = true;

// C library sort
TVM_REGISTER_GLOBAL("tvm.contrib.generic.utils.stable_sort")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor* input = args[0];
  DLTensor* output = args[1];
  int32_t num_elem = args[2];
  int32_t sort_value_index = args[3];
  bool is_descend = args[4];

  CHECK_EQ(input->ndim, 2) << "Currently only support sorting 2-D tensor.";
  CHECK_EQ(output->ndim, 1) << "Ouput should be a 1-D tensor contains index of sorted elements.";

  float* data_ptr = static_cast<float*>(input->data);
  SortElem<float>::is_descend = is_descend;
  std::vector<SortElem<float>> sorter;

  for( int32_t i = 0; i < num_elem; ++i) {
    sorter.emplace_back(SortElem<float>(*(data_ptr + i * input->shape[1] + sort_value_index), i));
  }

  std::stable_sort(sorter.begin(), sorter.end());
  for( int32_t i = 0; i < num_elem; ++i) {
    *(static_cast<int32_t*>(output->data) + i) = sorter[i].index;
  }
});


} // namespace contrib
} // namespace tvm