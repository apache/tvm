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
 * \file src/runtime/contrib/cudnn/softmax.cc
 * \brief Use external cudnn softmax function
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

void softmax_impl(cudnnSoftmaxAlgorithm_t alg, TVMArgs args, TVMRetValue* ret) {
  DLTensor* x = args[0];
  DLTensor* y = args[1];
  int axis = args[2];
  int ndim = x->ndim;
  int64_t* shape = x->shape;
  if (axis < 0) axis += ndim;
  ICHECK(axis >= 0 && axis < ndim);

  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  entry_ptr->softmax_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);

  // Set mode and shape descriptor
  if (axis == ndim - 1) {
    int64_t N = 1;
    for (int i = 0; i < ndim - 1; ++i) {
      N *= shape[i];
    }
    entry_ptr->softmax_entry.mode = CUDNN_SOFTMAX_MODE_INSTANCE;
    CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->softmax_entry.shape_desc, CUDNN_TENSOR_NCHW,
                                          entry_ptr->softmax_entry.data_type, static_cast<int>(N),
                                          static_cast<int>(shape[ndim - 1]), 1, 1));
  } else {
    int64_t pre_axis_dim = 1;
    int64_t post_axis_dim = 1;
    for (int i = 0; i < ndim; ++i) {
      if (i < axis) {
        pre_axis_dim *= shape[i];
      } else if (i > axis) {
        post_axis_dim *= shape[i];
      }
    }
    entry_ptr->softmax_entry.mode = CUDNN_SOFTMAX_MODE_CHANNEL;
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        entry_ptr->softmax_entry.shape_desc, CUDNN_TENSOR_NCHW, entry_ptr->softmax_entry.data_type,
        static_cast<int>(pre_axis_dim), static_cast<int>(shape[axis]),
        static_cast<int>(post_axis_dim), 1));
  }

  auto alpha = CuDNNDataType::GetConst<1>(entry_ptr->softmax_entry.data_type);
  auto beta = CuDNNDataType::GetConst<0>(entry_ptr->softmax_entry.data_type);
  CUDNN_CALL(cudnnSoftmaxForward(entry_ptr->handle, alg, entry_ptr->softmax_entry.mode, alpha,
                                 entry_ptr->softmax_entry.shape_desc, x->data, beta,
                                 entry_ptr->softmax_entry.shape_desc, y->data));
}

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      softmax_impl(CUDNN_SOFTMAX_ACCURATE, args, ret);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.log_softmax.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) { softmax_impl(CUDNN_SOFTMAX_LOG, args, ret); });

}  // namespace contrib
}  // namespace tvm
