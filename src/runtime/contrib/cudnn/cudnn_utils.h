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

#ifndef TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_
#define TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_

#include <dmlc/logging.h>
#include <cudnn.h>
#include <tvm/runtime/device_api.h>
#include "../../cuda/cuda_common.h"


namespace tvm {
namespace contrib {

#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    cudnnStatus_t e = (func);                                                 \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e); \
  }

/*! breif Convert DLTensor type to CuDNN type */
struct CuDNNDataType {
  static cudnnDataType_t DLTypeToCuDNNType(const DLDataType &dtype);
  template<int v>
  static const void* GetConst(cudnnDataType_t type);
};  // struct CuDNNDataType

inline void GetStride(int nbdim, const int *dims, int *strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >=0; --i) {
    mul *= dims[i];
    strides[i] = mul;
  }
}

inline void GetCudnnStride(int nbdim,
                           const int* dims,
                           int* strides) {
  int mul = 1;
  for (int i = nbdim - 1; i >=0; --i) {
    strides[i] = mul;
    mul *= dims[i];
  }
}

struct ConvEntry {
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnConvolutionMode_t mode;
  cudnnFilterDescriptor_t filter_desc;
  cudnnDataType_t data_type;
  cudnnTensorFormat_t tensor_format;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  // cudnnMathType_t math_type;
  TVMContext ctx;
  runtime::DeviceAPI *cuda_api;
  void *workspace{nullptr};
  size_t workspace_size{0};
  ConvEntry();
  ~ConvEntry();
  void UpdateWorkspace(const size_t wsize);
  void CleanWorkspace();
};  // ConvThreadEntry

struct SoftmaxEntry {
  cudnnSoftmaxMode_t mode;
  cudnnDataType_t data_type;
  cudnnTensorDescriptor_t shape_desc;
  SoftmaxEntry();
  ~SoftmaxEntry();
};  // SoftmaxEntry

struct CuDNNThreadEntry {
  CuDNNThreadEntry();
  ~CuDNNThreadEntry();
  cudnnHandle_t handle{nullptr};
  ConvEntry conv_entry;
  SoftmaxEntry softmax_entry;
  runtime::DeviceAPI *cuda_api{nullptr};
  static CuDNNThreadEntry* ThreadLocal();
};  // CuDNNThreadEntry

}  // namespace contrib
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CUDNN_CUDNN_UTILS_H_
