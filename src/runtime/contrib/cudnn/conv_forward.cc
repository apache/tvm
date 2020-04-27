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
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;

void ConvolutionForward(
  int mode,
  int format,
  int algo,
  int dims,
  int groups,
  const int pad[],
  const int stride[],
  const int dilation[],
  DLTensor* x,
  DLTensor* w,
  DLTensor* y,
  const std::string& conv_dtype) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<cudnnConvolutionMode_t>(mode);
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Set Algo
  entry_ptr->conv_entry.fwd_algo = static_cast<cudnnConvolutionFwdAlgo_t>(algo);
  // Set Ctx
  entry_ptr->conv_entry.ctx = x->ctx;
  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Dims includes N and C
  int full_dims = dims + 2;

  std::vector<int> dim(full_dims);
  std::vector<int> tensor_stride(full_dims);

  // Note: For 2D tenor, using ND setters causes CUDNN_STATUS_NOT_SUPPORTED error
  // in following cudnnGetConvolutionForwardWorkspaceSize() when data type is fp16, int

  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  if (dims == 2) {
    // Set Desc
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(entry_ptr->conv_entry.conv_desc,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilation[0],
                                               dilation[1],
                                               entry_ptr->conv_entry.mode,
                                               entry_ptr->conv_entry.data_type));
    int ni, ci, hi, wi;
    if (entry_ptr->conv_entry.tensor_format ==  CUDNN_TENSOR_NHWC) {
      ni = 0;
      ci = 3;
      hi = 1;
      wi = 2;
    } else {
      ni = 0;
      ci = 1;
      hi = 2;
      wi = 3;
    }

    // Set Filter
    CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                          data_type,
                                          entry_ptr->conv_entry.tensor_format,
                                          static_cast<int>(w->shape[ni]),
                                          static_cast<int>(w->shape[ci]),
                                          static_cast<int>(w->shape[hi]),
                                          static_cast<int>(w->shape[wi])));
    // Set Input
    CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                          entry_ptr->conv_entry.tensor_format,
                                          data_type,
                                          static_cast<int>(x->shape[ni]),
                                          static_cast<int>(x->shape[ci]),
                                          static_cast<int>(x->shape[hi]),
                                          static_cast<int>(x->shape[wi])));
    // Set Output
    CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.output_desc,
                                          entry_ptr->conv_entry.tensor_format,
                                          data_type,
                                          static_cast<int>(y->shape[ni]),
                                          static_cast<int>(y->shape[ci]),
                                          static_cast<int>(y->shape[hi]),
                                          static_cast<int>(y->shape[wi])));
  } else {
    CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc,
                                               dims,
                                               pad,
                                               stride,
                                               dilation,
                                               entry_ptr->conv_entry.mode,
                                               entry_ptr->conv_entry.data_type));

    // Set Filter
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(w->shape[i]);
    }
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc,
                                          data_type,
                                          entry_ptr->conv_entry.tensor_format,
                                          full_dims,
                                          dim.data()));
    // Set Input
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(x->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc,
                                          data_type,
                                          full_dims,
                                          dim.data(),
                                          tensor_stride.data()));
    // Set Output
    for (int i = 0; i < full_dims; i++) {
      dim[i] = static_cast<int>(y->shape[i]);
    }
    GetCudnnStride(full_dims, dim.data(), tensor_stride.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc,
                                          data_type,
                                          full_dims,
                                          dim.data(),
                                          tensor_stride.data()));
  }

  if (cudnnGetVersion() > 7000) {
    CUDNN_CALL(cudnnSetConvolutionMathType(entry_ptr->conv_entry.conv_desc, CUDNN_TENSOR_OP_MATH))
  }

  // Set workspace
  size_t workspace_size = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(entry_ptr->handle,
                                                     entry_ptr->conv_entry.input_desc,
                                                     entry_ptr->conv_entry.filter_desc,
                                                     entry_ptr->conv_entry.conv_desc,
                                                     entry_ptr->conv_entry.output_desc,
                                                     entry_ptr->conv_entry.fwd_algo,
                                                     &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);
  CUDNN_CALL(cudnnConvolutionForward(entry_ptr->handle,
                                     CuDNNDataType::GetConst<1>(entry_ptr->conv_entry.data_type),
                                     entry_ptr->conv_entry.input_desc,
                                     x->data,
                                     entry_ptr->conv_entry.filter_desc,
                                     w->data,
                                     entry_ptr->conv_entry.conv_desc,
                                     entry_ptr->conv_entry.fwd_algo,
                                     entry_ptr->conv_entry.workspace,
                                     workspace_size,
                                     CuDNNDataType::GetConst<0>(entry_ptr->conv_entry.data_type),
                                     entry_ptr->conv_entry.output_desc,
                                     y->data));
}


void OutputShape(
  int format,
  int dims,
  int groups,
  const int pad[],
  const int stride[],
  const int dilation[],
  const int x_dim[],
  const int w_dim[],
  void *out_shape,
  const std::string& data_dtype,
  const std::string& conv_dtype) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();

  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(data_dtype));
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Dims includes N and C
  int full_dims = dims + 2;

  // conv desc
  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc,
                                             dims,
                                             pad,
                                             stride,
                                             dilation,
                                             CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));

  if (dims == 2 && entry_ptr->conv_entry.tensor_format ==  CUDNN_TENSOR_NHWC) {
    // Set Input
    CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                          entry_ptr->conv_entry.tensor_format,
                                          data_type,
                                          x_dim[0],
                                          x_dim[3],
                                          x_dim[1],
                                          x_dim[2]));

    // filter desc
    CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                          data_type,
                                          entry_ptr->conv_entry.tensor_format,
                                          w_dim[0],
                                          w_dim[3],
                                          w_dim[1],
                                          w_dim[2]));

    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(entry_ptr->conv_entry.conv_desc,
                                                     entry_ptr->conv_entry.input_desc,
                                                     entry_ptr->conv_entry.filter_desc,
                                                     static_cast<int*>(out_shape),
                                                     static_cast<int*>(out_shape) + 3,
                                                     static_cast<int*>(out_shape) + 1,
                                                     static_cast<int*>(out_shape) + 2));
  } else {
    // Set Input
    std::vector<int> tensor_stride(full_dims);
    GetCudnnStride(full_dims, x_dim, tensor_stride.data());

    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc,
                                          data_type,
                                          full_dims,
                                          x_dim,
                                          tensor_stride.data()));
    // filter desc
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc,
                                          data_type,
                                          entry_ptr->conv_entry.tensor_format,
                                          full_dims,
                                          w_dim));

    CUDNN_CALL(cudnnGetConvolutionNdForwardOutputDim(entry_ptr->conv_entry.conv_desc,
                                                     entry_ptr->conv_entry.input_desc,
                                                     entry_ptr->conv_entry.filter_desc,
                                                     full_dims,
                                                     static_cast<int*>(out_shape)));
  }
}


void FindAlgo(
  int format,
  int dims,
  int groups,
  const int pad[],
  const int stride[],
  const int dilation[],
  const int x_dim[],
  const int w_dim[],
  const int y_dim[],
  const std::string& data_dtype,
  const std::string& conv_dtype,
  TVMRetValue *ret) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();

  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(String2DLDataType(data_dtype));
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // Dims includes N and C
  int full_dims = dims + 2;

  // conv desc
  CUDNN_CALL(cudnnSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, groups));
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc,
                                             dims,
                                             pad,
                                             stride,
                                             dilation,
                                             CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));

  std::vector<int> tensor_stride(full_dims);
  // input desc
  GetCudnnStride(full_dims, x_dim, tensor_stride.data());
  CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc,
                                        data_type,
                                        full_dims,
                                        x_dim,
                                        tensor_stride.data()));
  // filter desc
  CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc,
                                        data_type,
                                        entry_ptr->conv_entry.tensor_format,
                                        full_dims,
                                        w_dim));

  // output desc
  GetCudnnStride(full_dims, y_dim, tensor_stride.data());
  CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc,
                                        data_type,
                                        full_dims,
                                        y_dim,
                                        tensor_stride.data()));
  if (cudnnGetVersion() > 7000) {
    CUDNN_CALL(cudnnSetConvolutionMathType(entry_ptr->conv_entry.conv_desc, CUDNN_TENSOR_OP_MATH))
  }

  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(entry_ptr->handle,
                                                  entry_ptr->conv_entry.input_desc,
                                                  entry_ptr->conv_entry.filter_desc,
                                                  entry_ptr->conv_entry.conv_desc,
                                                  entry_ptr->conv_entry.output_desc,
                                                  CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                                  &returned_algo_count,
                                                  perf_results));

  const std::vector<std::string> fwd_algo_names{
      "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
      "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
      "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
      "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
      "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
      "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
      "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
      "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
  };

  auto best_algo = perf_results[0].algo;
  LOG(INFO) << "\tCUDNN Found " << returned_algo_count
            << " fwd algorithms, choosing " << fwd_algo_names[best_algo];
  for (int i = 0; i < returned_algo_count; ++i) {
    LOG(INFO) << "\t\t" << i << ") " << fwd_algo_names[perf_results[i].algo]
              << " - time: " << perf_results[i].time << " ms"
              << ", Memory: " << perf_results[i].memory;
  }

  ret[0] = best_algo;
}


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int mode = args[0];
  int format = args[1];
  int algo = args[2];
  int pad_v[2], stride_v[2], dilation_v[2];
  for (int i = 0; i < 2; i++) {
    pad_v[i] = args[3 + i];
    stride_v[i] = args[5 + i];
    dilation_v[i] = args[7 + i];
  }
  DLTensor* x = args[9];
  DLTensor* w = args[10];
  DLTensor* y = args[11];
  std::string conv_dtype = args[12];
  int groups = args[13];

  ConvolutionForward(mode, format, algo, 2, groups, pad_v, stride_v,
                     dilation_v, x, w, y, conv_dtype);
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv3d.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int mode = args[0];
  int format = args[1];
  int algo = args[2];
  int pad_v[3], stride_v[3], dilation_v[3];
  for (int i = 0; i < 3; i++) {
    pad_v[i] = args[3 + i];
    stride_v[i] = args[6 + i];
    dilation_v[i] = args[9 + i];
  }
  DLTensor *x = args[12];
  DLTensor *w = args[13];
  DLTensor *y = args[14];
  std::string conv_dtype = args[15];
  int groups = args[16];

  ConvolutionForward(mode, format, algo, 3, groups, pad_v, stride_v,
                     dilation_v, x, w, y, conv_dtype);
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.output_shape")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int format = args[0];
  int dims = args[1];
  int* pad = static_cast<int*>(static_cast<void*>(args[2]));
  int* stride = static_cast<int*>(static_cast<void*>(args[3]));
  int* dilation = static_cast<int*>(static_cast<void*>(args[4]));
  int* x_dim = static_cast<int*>(static_cast<void*>(args[5]));
  int* w_dim = static_cast<int*>(static_cast<void*>(args[6]));
  void* out_shape = args[7];
  std::string data_dtype = args[8];
  std::string conv_dtype = args[9];
  int groups = args[10];

  OutputShape(format, dims, groups, pad, stride, dilation, x_dim,
              w_dim, out_shape, data_dtype, conv_dtype);
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.find_algo")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int format = args[0];
  int dims = args[1];
  int* pad = static_cast<int*>(static_cast<void*>(args[2]));
  int* stride = static_cast<int*>(static_cast<void*>(args[3]));
  int* dilation = static_cast<int*>(static_cast<void*>(args[4]));
  int* x_dim = static_cast<int*>(static_cast<void*>(args[5]));
  int* w_dim = static_cast<int*>(static_cast<void*>(args[6]));
  int* y_dim = static_cast<int*>(static_cast<void*>(args[7]));
  std::string data_dtype = args[8];
  std::string conv_dtype = args[9];
  int groups = args[10];

  FindAlgo(format, dims, groups, pad, stride, dilation, x_dim,
           w_dim, y_dim, data_dtype, conv_dtype, ret);
});

}  // namespace contrib
}  // namespace tvm
