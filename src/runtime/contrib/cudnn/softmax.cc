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
 * \file Use external cudnn utils function
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

/*
  cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t                    handle,
    cudnnSoftmaxAlgorithm_t          algorithm,
    cudnnSoftmaxMode_t               mode,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    xDesc,
    const void                      *x,
    const void                      *beta,
    const cudnnTensorDescriptor_t    yDesc,
    void                            *y)

2.62. cudnnSoftmaxAlgorithm_t

CUDNN_SOFTMAX_FAST
This implementation applies the straightforward softmax operation.

CUDNN_SOFTMAX_ACCURATE
This implementation scales each point of the softmax input domain by its maximum value to avoid potential floating point overflows in the softmax evaluation.

CUDNN_SOFTMAX_LOG
This entry performs the log softmax operation, avoiding overflows by scaling each point in the input domain as in CUDNN_SOFTMAX_ACCURATE.

2.63. cudnnSoftmaxMode_t
cudnnSoftmaxMode_t is used to select over which data the cudnnSoftmaxForward() and cudnnSoftmaxBackward() are computing their results.

Values
CUDNN_SOFTMAX_MODE_INSTANCE
The softmax operation is computed per image (N) across the dimensions C,H,W.

CUDNN_SOFTMAX_MODE_CHANNEL
The softmax operation is computed per spatial location (H,W) per image (N) across the dimension C.

*/
TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.softmax.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor* x = args[0];
  DLTensor* y = args[1];
  int axis = args[2];
  int ndim = x->ndim;
  int64_t* shape = x->shape;
  if (axis < 0) axis += ndim;
  CHECK(axis >= 0 && axis < ndim);
  CHECK(axis == ndim - 1) << "Currently only support axis=-1 for cudnn softmax";
  int64_t N = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    N *= shape[i];
  }
  
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  entry_ptr->softmax_entry.mode = CUDNN_SOFTMAX_MODE_INSTANCE;
  entry_ptr->softmax_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

  // Set shape descriptor
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->softmax_entry.shape_desc,
                                        CUDNN_TENSOR_NCHW,
                                        entry_ptr->softmax_entry.data_type,
                                        static_cast<int>(N),
                                        static_cast<int>(shape[ndim - 1]),
                                        1,
                                        1));
  auto alpha = CuDNNDataType::GetConst<1>(entry_ptr->softmax_entry.data_type);
  auto beta = CuDNNDataType::GetConst<0>(entry_ptr->softmax_entry.data_type);
  CUDNN_CALL(cudnnSoftmaxForward(entry_ptr->handle,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 entry_ptr->softmax_entry.mode,
                                 alpha,
                                 entry_ptr->softmax_entry.shape_desc,
                                 x->data,
                                 beta,
                                 entry_ptr->softmax_entry.shape_desc,
                                 y->data));
});

}  // namespace contrib
}  // namespace tvm
