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
#include <tvm/runtime/util.h>
#include <tvm/packed_func_ext.h>
#include <tvm/ir.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;
using tvm::ir::IntImm;

void ConvolutionForward(
  int mode,
  int format,
  int algo,
  int dims,
  const int pad_v[],
  const int stride_v[],
  const int dilation_v[],
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
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2TVMType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Dims includes N and C
  int full_dims = dims + 2;

  std::vector<int> dim_v(full_dims);
  std::vector<int> tensor_stride_v(full_dims);

  // Note: For 2D tenor, using ND setters causes CUDNN_STATUS_NOT_SUPPORTED error
  // in following cudnnGetConvolutionForwardWorkspaceSize() when data type is fp16, int
  if (dims == 2) {
  // Set Desc
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(entry_ptr->conv_entry.conv_desc,
                                               pad_v[0],
                                               pad_v[1],
                                               stride_v[0],
                                               stride_v[1],
                                               dilation_v[0],
                                               dilation_v[1],
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
                                               pad_v,
                                               stride_v,
                                               dilation_v,
                                               entry_ptr->conv_entry.mode,
                                               entry_ptr->conv_entry.data_type));

    // Set Filter
    for (int i = 0; i < full_dims; i++) {
      dim_v[i] = static_cast<int>(w->shape[i]);
    }
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc,
                                          data_type,
                                          entry_ptr->conv_entry.tensor_format,
                                          full_dims,
                                          dim_v.data()));
    // Set Input
    for (int i = 0; i < full_dims; i++) {
      dim_v[i] = static_cast<int>(x->shape[i]);
    }
    GetCudnnStride(full_dims, dim_v.data(), tensor_stride_v.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc,
                                          data_type,
                                          full_dims,
                                          dim_v.data(),
                                          tensor_stride_v.data()));
    // Set Output
    for (int i = 0; i < full_dims; i++) {
      dim_v[i] = static_cast<int>(y->shape[i]);
    }
    GetCudnnStride(full_dims, dim_v.data(), tensor_stride_v.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc,
                                          data_type,
                                          full_dims,
                                          dim_v.data(),
                                          tensor_stride_v.data()));
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
  const int pad_v[],
  const int stride_v[],
  const int dilation_v[],
  const int x_dim_v[],
  const int w_dim_v[],
  void *out_shape,
  const std::string& data_dtype,
  const std::string& conv_dtype) {
  // Dims includes N and C
  int full_dims = dims + 2;

  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();

  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2TVMType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(String2TVMType(data_dtype));
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);

  // conv desc
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc,
                                             dims,
                                             pad_v,
                                             stride_v,
                                             dilation_v,
                                             CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));

  if (dims == 2 && entry_ptr->conv_entry.tensor_format ==  CUDNN_TENSOR_NHWC) {
    // Set Input
    CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                          entry_ptr->conv_entry.tensor_format,
                                          data_type,
                                          x_dim_v[0],
                                          x_dim_v[3],
                                          x_dim_v[1],
                                          x_dim_v[2]));

    // filter desc
    CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                          data_type,
                                          entry_ptr->conv_entry.tensor_format,
                                          w_dim_v[0],
                                          w_dim_v[3],
                                          w_dim_v[1],
                                          w_dim_v[2]));

    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(entry_ptr->conv_entry.conv_desc,
                                                     entry_ptr->conv_entry.input_desc,
                                                     entry_ptr->conv_entry.filter_desc,
                                                     static_cast<int*>(out_shape),
                                                     static_cast<int*>(out_shape) + 3,
                                                     static_cast<int*>(out_shape) + 1,
                                                     static_cast<int*>(out_shape) + 2));
  } else {
    // Set Input
    std::vector<int> tensor_stride_v(full_dims);
    GetCudnnStride(full_dims, x_dim_v, tensor_stride_v.data());
    CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc,
                                          data_type,
                                          full_dims,
                                          x_dim_v,
                                          tensor_stride_v.data()));
    // filter desc
    CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc,
                                          data_type,
                                          entry_ptr->conv_entry.tensor_format,
                                          full_dims,
                                          w_dim_v));

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
  const int pad_v[],
  const int stride_v[],
  const int dilation_v[],
  const int x_dim_v[],
  const int w_dim_v[],
  const int y_dim_v[],
  const std::string& data_dtype,
  const std::string& conv_dtype,
  TVMRetValue *ret) {
  // Dims includes N and C
  int full_dims = dims + 2;

  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();

  // Set Data Type
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(String2TVMType(conv_dtype));
  cudnnDataType_t data_type = CuDNNDataType::DLTypeToCuDNNType(String2TVMType(data_dtype));
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // conv desc
  CUDNN_CALL(cudnnSetConvolutionNdDescriptor(entry_ptr->conv_entry.conv_desc,
                                             dims,
                                             pad_v,
                                             stride_v,
                                             dilation_v,
                                             CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));

  std::vector<int> tensor_stride_v(full_dims);
  // input desc
  GetCudnnStride(full_dims, x_dim_v, tensor_stride_v.data());
  CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.input_desc,
                                        data_type,
                                        full_dims,
                                        x_dim_v,
                                        tensor_stride_v.data()));
  // filter desc
  CUDNN_CALL(cudnnSetFilterNdDescriptor(entry_ptr->conv_entry.filter_desc,
                                        data_type,
                                        entry_ptr->conv_entry.tensor_format,
                                        full_dims,
                                        w_dim_v));

  // output desc
  GetCudnnStride(full_dims, y_dim_v, tensor_stride_v.data());
  CUDNN_CALL(cudnnSetTensorNdDescriptor(entry_ptr->conv_entry.output_desc,
                                        data_type,
                                        full_dims,
                                        y_dim_v,
                                        tensor_stride_v.data()));
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


void PrepareCommonArgs(const TVMArgs& args,
                       int offset,
                       int dims,
                       std::vector<int>* pad_v,
                       std::vector<int>* stride_v,
                       std::vector<int>* dilation_v) {
  int* pad = static_cast<int*>(static_cast<void*>(args[offset]));
  int* stride = static_cast<int*>(static_cast<void*>(args[offset + 1]));
  int* dilation = static_cast<int*>(static_cast<void*>(args[offset + 2]));

  std::copy(pad, pad + dims, pad_v->begin());
  std::copy(stride, stride + dims, stride_v->begin());
  std::copy(dilation, dilation + dims, dilation_v->begin());
}


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int mode = args[0];
  int format = args[1];
  int algo = args[2];
  int dims = args[3];

  Array<Expr> pads =  args[4];
  Array<Expr> strides =  args[5];
  Array<Expr> dilations =  args[6];


  std::vector<int> pad_v(dims), stride_v(dims), dilation_v(dims);
  for (auto &pad : pads) {
    int i = 0;
    pad_v[i++] = pad.as<IntImm>()->value;
  }
  for (auto &stride : strides) {
    int i = 0;
    pad_v[i++] = strides.as<IntImm>()->value;
  }
  for (auto &dilation : dilations) {
    int i = 0;
    pad_v[i++] = dilations.as<IntImm>()->value;
  }

  DLTensor* x = args[7];
  DLTensor* w = args[8];
  DLTensor* y = args[9];
  std::string conv_dtype = args[10];

  ConvolutionForward(mode, format, algo, 2, pad_v.data(), stride_v.data(), dilation_v.data(),
                     x, w, y, conv_dtype);
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.output_shape")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int format = args[0];
  int dims = args[1];

  std::vector<int> pad_v(dims), stride_v(dims), dilation_v(dims);
  PrepareCommonArgs(args, 2, dims, &pad_v, &stride_v, &dilation_v);

  int* x_dim = static_cast<int*>(static_cast<void*>(args[5]));
  int* w_dim = static_cast<int*>(static_cast<void*>(args[6]));
  std::vector<int> x_dim_v(x_dim, x_dim + dims + 2);
  std::vector<int> w_dim_v(w_dim, w_dim + dims + 2);

  void* out_shape = args[7];
  std::string data_dtype = args[8];
  std::string conv_dtype = args[9];

  OutputShape(format, 2, pad_v.data(), stride_v.data(), dilation_v.data(), x_dim_v.data(),
              w_dim_v.data(), out_shape, data_dtype, conv_dtype);
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv.find_algo")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int format = args[0];
  int dims = args[1];
  std::vector<int> pad_v(dims), stride_v(dims), dilation_v(dims);
  PrepareCommonArgs(args, 2, dims, &pad_v, &stride_v, &dilation_v);

  int* x_dim = static_cast<int*>(static_cast<void*>(args[5]));
  int* w_dim = static_cast<int*>(static_cast<void*>(args[6]));
  int* y_dim = static_cast<int*>(static_cast<void*>(args[7]));
  std::vector<int> x_dim_v(x_dim, x_dim + dims + 2);
  std::vector<int> w_dim_v(w_dim, w_dim + dims + 2);
  std::vector<int> y_dim_v(y_dim, y_dim + dims + 2);

  std::string data_dtype = args[8];
  std::string conv_dtype = args[9];

  FindAlgo(format, 2, pad_v.data(), stride_v.data(), dilation_v.data(), x_dim_v.data(),
           w_dim_v.data(), y_dim_v.data(), data_dtype, conv_dtype, ret);
});

}  // namespace contrib
}  // namespace tvm
