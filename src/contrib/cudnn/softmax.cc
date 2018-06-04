/*!
 *  Copyright (c) 2018 by Contributors
 * \file Use external cudnn utils function
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.softmax.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  float alpha = 1.0f;
  float beta = 0.0f;
  int alg = args[0];
  int mode = args[1];
  DLTensor *src = args[2];
  DLTensor *dst = args[3];
  int dshape[4];

  if (mode == 0) { // SOFTMAX_MODE_INSTANCE
    CHECK_EQ(src->ndim, 2)
        << "cudnn.softmax only support 2 dimensions tensor when mode=instance";
    dshape[0] = static_cast<int>(src->shape[0]);
    dshape[1] = static_cast<int>(src->shape[1]);
    dshape[2] = 1;
    dshape[3] = 1;
  } else { // SOFTMAX_MODE_CHANNEL
    CHECK_GE(src->ndim, 3)
        << "cudnn.softmax need only support greater than 3 dimensions tensor when mode=channel";
    int i = 0;
    int size_left = 0;
    for (; i < 3; i++) {
      if (i < src->ndim) {
        dshape[i] = static_cast<int>(src->shape[i]);
      } else {
        dshape[i] = 1;
      }
    }

    if (i < src->ndim) {
      for (; i < src->ndim; i++) {
        size_left += src->shape[i];
      }
      dshape[3] = size_left;
    } else {
      dshape[3] = 1;
    }
  }
  
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Algorithm
  entry_ptr->softmax_entry.alg = static_cast<cudnnSoftmaxAlgorithm_t>(alg);  
  // Set Mode
  entry_ptr->softmax_entry.mode = static_cast<cudnnSoftmaxMode_t>(mode);
  // Set Data Type
  entry_ptr->softmax_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(src->dtype);  
  // Set Input
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->softmax_entry.input_desc,
                                        CUDNN_TENSOR_NCHW,
                                        entry_ptr->softmax_entry.data_type,
                                        dshape[0],
                                        dshape[1],
                                        dshape[2],
                                        dshape[3]));
  // Set Output
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->softmax_entry.output_desc,
                                        CUDNN_TENSOR_NCHW,
                                        entry_ptr->softmax_entry.data_type,
                                        dshape[0],
                                        dshape[1],
                                        dshape[2],
                                        dshape[3]));

  CUDNN_CALL(cudnnSoftmaxForward(entry_ptr->handle,
                                 entry_ptr->softmax_entry.alg, //softmax algorithm
                                 entry_ptr->softmax_entry.mode, //mode
                                 &alpha,
                                 entry_ptr->softmax_entry.input_desc,
                                 src->data,
                                 &beta,
                                 entry_ptr->softmax_entry.output_desc,
                                 dst->data));
});

}  // namespace contrib
}  // namespace tvm
