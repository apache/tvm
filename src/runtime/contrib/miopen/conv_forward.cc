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
 * \file Use external miopen utils function
 */
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <cassert>

#include "miopen_utils.h"

namespace tvm {
namespace contrib {
namespace miopen {

using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.miopen.conv2d.setup").set_body([](TVMArgs args, TVMRetValue* ret) {
  const int mode = args[0];
  const int dtype = args[1];
  const int pad_h = args[2];
  const int pad_w = args[3];
  const int stride_h = args[4];
  const int stride_w = args[5];
  const int dilation_h = args[6];
  const int dilation_w = args[7];
  const int x_dim0 = args[8];
  const int x_dim1 = args[9];
  const int x_dim2 = args[10];
  const int x_dim3 = args[11];
  const int w_dim0 = args[12];
  const int w_dim1 = args[13];
  const int w_dim2 = args[14];
  const int w_dim3 = args[15];
  const int n_group = args[16];
  void* out_shape = args[17];

  MIOpenThreadEntry* entry_ptr = MIOpenThreadEntry::ThreadLocal();
  assert(n_group > 0 && "Group Size > 0 is expected");
  if (n_group > 1) assert(mode > 1 && "Group /Depthwise Conv mode when num of groups > 1");
  // Set Mode
  entry_ptr->conv_entry.mode = static_cast<miopenConvolutionMode_t>(mode);
  // Set Device
  entry_ptr->conv_entry.device = Device{kDLROCM, 0};
  // Set Data Type
  entry_ptr->conv_entry.data_type =
      static_cast<miopenDataType_t>(dtype);  // MIOpen supports fp32(miopenFloat), fp16(miopenHalf),
                                             // int32, int8 at this moment.
  // Set Desc
  MIOPEN_CALL(miopenInitConvolutionDescriptor(entry_ptr->conv_entry.conv_desc,
                                              entry_ptr->conv_entry.mode, pad_h, pad_w, stride_h,
                                              stride_w, dilation_h, dilation_w));
  if (n_group > 1)
    MIOPEN_CALL(miopenSetConvolutionGroupCount(entry_ptr->conv_entry.conv_desc, n_group));
  // Set Filter
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.filter_desc,
                                          entry_ptr->conv_entry.data_type, w_dim0, w_dim1 / n_group,
                                          w_dim2, w_dim3));
  // Set Input
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.input_desc,
                                          entry_ptr->conv_entry.data_type, x_dim0, x_dim1, x_dim2,
                                          x_dim3));

  // Set Output shape
  MIOPEN_CALL(miopenGetConvolutionForwardOutputDim(
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.input_desc,
      entry_ptr->conv_entry.filter_desc, static_cast<int*>(out_shape),
      static_cast<int*>(out_shape) + 1, static_cast<int*>(out_shape) + 2,
      static_cast<int*>(out_shape) + 3));

  const int* oshape = static_cast<int*>(out_shape);
  // Set Output
  MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.output_desc,
                                          entry_ptr->conv_entry.data_type, oshape[0], oshape[1],
                                          oshape[2], oshape[3]));

  // Set workspace
  size_t workspace_size = 0;
  MIOPEN_CALL(miopenConvolutionForwardGetWorkSpaceSize(
      entry_ptr->handle, entry_ptr->conv_entry.filter_desc, entry_ptr->conv_entry.input_desc,
      entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc, &workspace_size));
  entry_ptr->conv_entry.UpdateWorkspace(workspace_size);

  const size_t input_size = x_dim0 * x_dim1 * x_dim2 * x_dim3;
  const size_t filter_size = w_dim0 * w_dim1 * w_dim2 * w_dim3;
  const size_t output_size = oshape[0] * oshape[1] * oshape[2] * oshape[3];

  runtime::DeviceAPI* rocm_api = entry_ptr->conv_entry.rocm_api;
  float* input_buf = static_cast<float*>(
      rocm_api->AllocWorkspace(entry_ptr->conv_entry.device, input_size * sizeof(float)));
  float* filter_buf = static_cast<float*>(
      rocm_api->AllocWorkspace(entry_ptr->conv_entry.device, filter_size * sizeof(float)));
  float* output_buf = static_cast<float*>(
      rocm_api->AllocWorkspace(entry_ptr->conv_entry.device, output_size * sizeof(float)));

  const int request_algo_count = 4;
  const bool exhaustive_search = false;
  void* workspace = entry_ptr->conv_entry.workspace;
  if (workspace_size == 0) workspace = nullptr;
  int returned_algo_count = 0;
  miopenConvAlgoPerf_t perfs[4];

  MIOPEN_CALL(miopenFindConvolutionForwardAlgorithm(
      entry_ptr->handle, entry_ptr->conv_entry.input_desc, input_buf,
      entry_ptr->conv_entry.filter_desc, filter_buf, entry_ptr->conv_entry.conv_desc,
      entry_ptr->conv_entry.output_desc, output_buf, request_algo_count, &returned_algo_count,
      perfs, workspace, workspace_size, exhaustive_search));

  rocm_api->FreeWorkspace(entry_ptr->conv_entry.device, input_buf);
  rocm_api->FreeWorkspace(entry_ptr->conv_entry.device, filter_buf);
  rocm_api->FreeWorkspace(entry_ptr->conv_entry.device, output_buf);

  const std::vector<std::string> fwd_algo_names{
      "miopenConvolutionFwdAlgoGEMM",
      "miopenConvolutionFwdAlgoDirect",
      "miopenConvolutionFwdAlgoFFT",
      "miopenConvolutionFwdAlgoWinograd",
  };
  const auto best_algo = perfs[0].fwd_algo;
  LOG(INFO) << "\tMIOpen Found " << returned_algo_count << " fwd algorithms, choosing "
            << fwd_algo_names[best_algo];
  for (int i = 0; i < returned_algo_count; ++i) {
    LOG(INFO) << "\t\t" << i << ") " << fwd_algo_names[perfs[i].fwd_algo]
              << " - time: " << perfs[i].time << " ms"
              << ", Memory: " << perfs[i].memory;
  }
  // Set Algo
  ret[0] = static_cast<int>(best_algo);
});

TVM_REGISTER_GLOBAL("tvm.contrib.miopen.conv2d.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      const int mode = args[0];
      const int dtype = args[1];
      const int pad_h = args[2];
      const int pad_w = args[3];
      const int stride_h = args[4];
      const int stride_w = args[5];
      const int dilation_h = args[6];
      const int dilation_w = args[7];
      const int algo = args[8];
      const DLTensor* x = args[9];
      const DLTensor* w = args[10];
      const DLTensor* y = args[11];

      MIOpenThreadEntry* entry_ptr = MIOpenThreadEntry::ThreadLocal();
      entry_ptr->conv_entry.fwd_algo = static_cast<miopenConvFwdAlgorithm_t>(algo);
      // Set Mode
      entry_ptr->conv_entry.mode = static_cast<miopenConvolutionMode_t>(mode);
      // Set Device
      entry_ptr->conv_entry.device = x->device;
      // Set Data Type
      entry_ptr->conv_entry.data_type =
          static_cast<miopenDataType_t>(dtype);  // MIOpen supports fp32(miopenFloat),
                                                 // fp16(miopenHalf) at this moment.
      // Set Desc
      MIOPEN_CALL(miopenInitConvolutionDescriptor(entry_ptr->conv_entry.conv_desc,
                                                  entry_ptr->conv_entry.mode, pad_h, pad_w,
                                                  stride_h, stride_w, dilation_h, dilation_w));
      // Set Filter
      MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.filter_desc,
                                              entry_ptr->conv_entry.data_type, w->shape[0],
                                              w->shape[1], w->shape[2], w->shape[3]));
      // Set Input
      MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.input_desc,
                                              entry_ptr->conv_entry.data_type, x->shape[0],
                                              x->shape[1], x->shape[2], x->shape[3]));
      // Set Output
      MIOPEN_CALL(miopenSet4dTensorDescriptor(entry_ptr->conv_entry.output_desc,
                                              entry_ptr->conv_entry.data_type, y->shape[0],
                                              y->shape[1], y->shape[2], y->shape[3]));

      // Set workspace
      size_t workspace_size = 0;
      MIOPEN_CALL(miopenConvolutionForwardGetWorkSpaceSize(
          entry_ptr->handle, entry_ptr->conv_entry.filter_desc, entry_ptr->conv_entry.input_desc,
          entry_ptr->conv_entry.conv_desc, entry_ptr->conv_entry.output_desc, &workspace_size));
      entry_ptr->conv_entry.UpdateWorkspace(workspace_size);

      const float alpha = 1.f;
      const float beta = 0.f;

      const int request_algo_count = 4;
      const bool exhaustive_search = true;
      void* workspace = entry_ptr->conv_entry.workspace;
      if (workspace_size == 0) workspace = nullptr;
      int returned_algo_count = 0;
      miopenConvAlgoPerf_t perfs[4];

      MIOPEN_CALL(miopenFindConvolutionForwardAlgorithm(
          entry_ptr->handle, entry_ptr->conv_entry.input_desc, x->data,
          entry_ptr->conv_entry.filter_desc, w->data, entry_ptr->conv_entry.conv_desc,
          entry_ptr->conv_entry.output_desc, y->data, request_algo_count, &returned_algo_count,
          perfs, workspace, workspace_size, exhaustive_search));

      MIOPEN_CALL(miopenConvolutionForward(
          entry_ptr->handle, &alpha, entry_ptr->conv_entry.input_desc, x->data,
          entry_ptr->conv_entry.filter_desc, w->data, entry_ptr->conv_entry.conv_desc,
          entry_ptr->conv_entry.fwd_algo, &beta, entry_ptr->conv_entry.output_desc, y->data,
          entry_ptr->conv_entry.workspace, entry_ptr->conv_entry.workspace_size));
    });

}  // namespace miopen
}  // namespace contrib
}  // namespace tvm
