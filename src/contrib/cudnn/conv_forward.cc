/*!
 *  Copyright (c) 2017 by Contributors
 * \file Use external cudnn utils function
 */
#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <tvm/runtime/device_api.h>
#include "cudnn_utils.h"

namespace tvm {
namespace contrib {

using namespace runtime;


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.forward")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  int mode = args[0];
  int format = args[1];
  int algo = args[2];
  int pad_h = args[3];
  int pad_w = args[4];
  int stride_h = args[5];
  int stride_w = args[6];
  int dilation_h = args[7];
  int dilation_w = args[8];
  DLTensor *x = args[9];
  DLTensor *w = args[10];
  DLTensor *y = args[11];
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
  entry_ptr->conv_entry.data_type = CuDNNDataType::DLTypeToCuDNNType(x->dtype);
  // Set Desc
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(entry_ptr->conv_entry.conv_desc,
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             entry_ptr->conv_entry.mode,
                                             entry_ptr->conv_entry.data_type));
  // Set Filter
  CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                        entry_ptr->conv_entry.data_type,
                                        CUDNN_TENSOR_NCHW,
                                        static_cast<int>(w->shape[0]),
                                        static_cast<int>(w->shape[1]),
                                        static_cast<int>(w->shape[2]),
                                        static_cast<int>(w->shape[3])));
  // Set Input
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(x->shape[0]),
                                        static_cast<int>(x->shape[1]),
                                        static_cast<int>(x->shape[2]),
                                        static_cast<int>(x->shape[3])));
  // Set Output
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.output_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        entry_ptr->conv_entry.data_type,
                                        static_cast<int>(y->shape[0]),
                                        static_cast<int>(y->shape[1]),
                                        static_cast<int>(y->shape[2]),
                                        static_cast<int>(y->shape[3])));
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
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.output_shape")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  int format = args[0];
  int pad_h = args[1];
  int pad_w = args[2];
  int stride_h = args[3];
  int stride_w = args[4];
  int dilation_h = args[5];
  int dilation_w = args[6];
  int x_dim0 = args[7];
  int x_dim1 = args[8];
  int x_dim2 = args[9];
  int x_dim3 = args[10];
  int w_dim0 = args[11];
  int w_dim1 = args[12];
  int w_dim2 = args[13];
  int w_dim3 = args[14];
  void *out_shape = args[15];
  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // conv desc
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(entry_ptr->conv_entry.conv_desc,
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));
  // input desc
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        CUDNN_DATA_FLOAT,
                                        x_dim0,
                                        x_dim1,
                                        x_dim2,
                                        x_dim3));
  // filter desc
  CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        w_dim0,
                                        w_dim1,
                                        w_dim2,
                                        w_dim3));

  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(entry_ptr->conv_entry.conv_desc,
                                                   entry_ptr->conv_entry.input_desc,
                                                   entry_ptr->conv_entry.filter_desc,
                                                   static_cast<int*>(out_shape),
                                                   static_cast<int*>(out_shape) + 1,
                                                   static_cast<int*>(out_shape) + 2,
                                                   static_cast<int*>(out_shape) + 3));
});


TVM_REGISTER_GLOBAL("tvm.contrib.cudnn.conv2d.find_algo")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  CuDNNThreadEntry* entry_ptr = CuDNNThreadEntry::ThreadLocal();
  int format = args[0];
  int pad_h = args[1];
  int pad_w = args[2];
  int stride_h = args[3];
  int stride_w = args[4];
  int dilation_h = args[5];
  int dilation_w = args[6];
  int x_dim0 = args[7];
  int x_dim1 = args[8];
  int x_dim2 = args[9];
  int x_dim3 = args[10];
  int w_dim0 = args[11];
  int w_dim1 = args[12];
  int w_dim2 = args[13];
  int w_dim3 = args[14];
  int y_dim0 = args[15];
  int y_dim1 = args[16];
  int y_dim2 = args[17];
  int y_dim3 = args[18];

  // Set Format
  entry_ptr->conv_entry.tensor_format = static_cast<cudnnTensorFormat_t>(format);
  // conv desc
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(entry_ptr->conv_entry.conv_desc,
                                             pad_h,
                                             pad_w,
                                             stride_h,
                                             stride_w,
                                             dilation_h,
                                             dilation_w,
                                             CUDNN_CROSS_CORRELATION,
                                             entry_ptr->conv_entry.data_type));
  // input desc
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.input_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        CUDNN_DATA_FLOAT,
                                        x_dim0,
                                        x_dim1,
                                        x_dim2,
                                        x_dim3));
  // filter desc
  CUDNN_CALL(cudnnSetFilter4dDescriptor(entry_ptr->conv_entry.filter_desc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        w_dim0,
                                        w_dim1,
                                        w_dim2,
                                        w_dim3));

  // output desc
  CUDNN_CALL(cudnnSetTensor4dDescriptor(entry_ptr->conv_entry.output_desc,
                                        entry_ptr->conv_entry.tensor_format,
                                        entry_ptr->conv_entry.data_type,
                                        y_dim0,
                                        y_dim1,
                                        y_dim2,
                                        y_dim3));

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
});

}  // namespace contrib
}  // namespace tvm
