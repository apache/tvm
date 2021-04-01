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
 * \file Use external nnpack library call.
 */
#include <nnpack.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include "nnpack_utils.h"

namespace tvm {
namespace contrib {
using namespace runtime;

TVM_REGISTER_GLOBAL("tvm.contrib.nnpack.convolution_inference")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      NNPackThreadLocalEntry* entry = NNPackThreadLocalEntry::ThreadLocal();
      static std::once_flag flag;
      std::call_once(flag, []() { ICHECK_EQ(nnp_initialize(), nnp_status_success); });
      DLTensor* input = args[0];
      DLTensor* kernel = args[1];
      DLTensor* bias = nullptr;
      if (args[2].type_code() == kTVMDLTensorHandle) {
        bias = args[2];
      }
      DLTensor* output = args[3];
      uint64_t pad_top = args[4], pad_right = args[5], pad_bottom = args[6], pad_left = args[7];
      nnp_padding input_padding{pad_top, pad_right, pad_bottom, pad_left};
      uint64_t stride_width = args[8], stride_height = args[9];
      nnp_size stride_size{stride_width, stride_height};
      NNPackConfig(args[10]);

      uint64_t algo_ = args[11];
      nnp_convolution_algorithm algo = static_cast<nnp_convolution_algorithm>(algo_);
      ICHECK_EQ(input->ndim, 4);
      ICHECK_EQ(kernel->ndim, 4);
      if (bias) {
        ICHECK_EQ(bias->ndim, 1);
      }
      ICHECK_EQ(output->ndim, 4);
      ICHECK_EQ(input->shape[1], kernel->shape[1]);
      ICHECK_EQ(input->shape[0], output->shape[0]);
      size_t input_channels = input->shape[1];
      ICHECK_EQ(output->shape[1], kernel->shape[0]);
      if (bias) {
        ICHECK_EQ(output->shape[1], bias->shape[0]);
      }
      size_t output_channels = output->shape[1];
      nnp_size input_size{static_cast<size_t>(input->shape[2]),
                          static_cast<size_t>(input->shape[3])};
      nnp_size kernel_size{static_cast<size_t>(kernel->shape[2]),
                           static_cast<size_t>(kernel->shape[3])};
      ICHECK(input->strides == nullptr);
      ICHECK(kernel->strides == nullptr);
      if (bias) {
        ICHECK(bias->strides == nullptr);
      }

      ICHECK(TypeMatch(input->dtype, kDLFloat, 32));
      ICHECK(TypeMatch(kernel->dtype, kDLFloat, 32));
      if (bias) {
        ICHECK(TypeMatch(bias->dtype, kDLFloat, 32));
      }
      ICHECK(TypeMatch(output->dtype, kDLFloat, 32));

      // Allocate a zero-bias if we don't pass one in.
      std::unique_ptr<std::vector<float>> zero_bias;
      if (!bias) {
        zero_bias.reset(new std::vector<float>(output->shape[1], 0.0));
      }

      size_t workspace_size = 0;
      nnp_status status = nnp_convolution_inference(
          algo, nnp_convolution_transform_strategy_compute, input_channels, output_channels,
          input_size, input_padding, kernel_size, stride_size, nullptr, nullptr, nullptr, nullptr,
          nullptr, &workspace_size, nnp_activation_identity, nullptr, entry->threadpool, nullptr);
      ICHECK_EQ(status, nnp_status_success);

      // Division with rounding up, in case size is not multiple of sizeof(float)
      const size_t workspace_elements = (workspace_size + sizeof(float) - 1) / sizeof(float);

      Device dev = input->device;
      DLDataType type_hint = input->dtype;

      DeviceAPI* cpu_api = DeviceAPI::Get(dev);
      void* workspace_buffer =
          cpu_api->AllocWorkspace(dev, workspace_elements * sizeof(float), type_hint);
      ICHECK(workspace_buffer != nullptr);

      for (auto n = 0; n < input->shape[0]; ++n) {
        nnp_status status = nnp_convolution_inference(
            algo, nnp_convolution_transform_strategy_compute, input_channels, output_channels,
            input_size, input_padding, kernel_size, stride_size,
            static_cast<float*>(input->data) +
                n * input->shape[1] * input->shape[2] * input->shape[3],
            static_cast<float*>(kernel->data),
            bias ? static_cast<float*>(bias->data) : zero_bias->data(),
            static_cast<float*>(output->data) +
                n * output->shape[1] * output->shape[2] * output->shape[3],
            workspace_buffer, &workspace_size, nnp_activation_identity, nullptr, entry->threadpool,
            nullptr);

        ICHECK_EQ(status, nnp_status_success);
      }
      cpu_api->FreeWorkspace(dev, workspace_buffer);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.nnpack.convolution_inference_without_weight_transform")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      NNPackThreadLocalEntry* entry = NNPackThreadLocalEntry::ThreadLocal();
      static std::once_flag flag;
      std::call_once(flag, []() { ICHECK_EQ(nnp_initialize(), nnp_status_success); });
      DLTensor* input = args[0];
      DLTensor* transformed_kernel = args[1];
      DLTensor* bias = nullptr;
      if (args[2].type_code() == kTVMDLTensorHandle) {
        bias = args[2];
      }
      DLTensor* output = args[3];
      uint64_t pad_top = args[4], pad_right = args[5], pad_bottom = args[6], pad_left = args[7];
      nnp_padding input_padding{pad_top, pad_right, pad_bottom, pad_left};
      uint64_t stride_width = args[8], stride_height = args[9];
      nnp_size stride_size{stride_width, stride_height};
      NNPackConfig(args[10]);

      uint64_t algo_ = args[11];
      nnp_convolution_algorithm algo = static_cast<nnp_convolution_algorithm>(algo_);
      ICHECK_EQ(input->ndim, 4);
      if (bias) {
        ICHECK_EQ(bias->ndim, 1);
      }
      ICHECK_EQ(output->ndim, 4);
      ICHECK_EQ(input->shape[0], output->shape[0]);
      size_t input_channels = input->shape[1];
      if (bias) {
        ICHECK_EQ(output->shape[1], bias->shape[0]);
      }
      size_t output_channels = output->shape[1];
      nnp_size input_size{static_cast<size_t>(input->shape[2]),
                          static_cast<size_t>(input->shape[3])};
      nnp_size kernel_size{3, 3};
      ICHECK(input->strides == nullptr);
      ICHECK(transformed_kernel->strides == nullptr);
      if (bias) {
        ICHECK(bias->strides == nullptr);
      }

      ICHECK(TypeMatch(input->dtype, kDLFloat, 32));
      ICHECK(TypeMatch(transformed_kernel->dtype, kDLFloat, 32));
      if (bias) {
        ICHECK(TypeMatch(bias->dtype, kDLFloat, 32));
      }
      ICHECK(TypeMatch(output->dtype, kDLFloat, 32));

      // Allocate a zero-bias if we don't pass one in.
      std::unique_ptr<std::vector<float>> zero_bias;
      if (!bias) {
        zero_bias.reset(new std::vector<float>(output->shape[1], 0.0));
      }

      size_t workspace_size = 0;
      nnp_status status = nnp_convolution_inference(
          algo, nnp_convolution_transform_strategy_reuse, input_channels, output_channels,
          input_size, input_padding, kernel_size, stride_size, nullptr, nullptr, nullptr, nullptr,
          nullptr, &workspace_size, nnp_activation_identity, nullptr, entry->threadpool, nullptr);
      ICHECK_EQ(status, nnp_status_success);

      // Division with rounding up, in case size is not multiple of sizeof(float)
      const size_t workspace_elements = (workspace_size + sizeof(float) - 1) / sizeof(float);

      Device dev = input->device;
      DLDataType type_hint = input->dtype;

      DeviceAPI* cpu_api = DeviceAPI::Get(dev);
      void* workspace_buffer =
          cpu_api->AllocWorkspace(dev, workspace_elements * sizeof(float), type_hint);
      ICHECK(workspace_buffer != nullptr);

      for (auto n = 0; n < input->shape[0]; ++n) {
        nnp_status status = nnp_convolution_inference(
            algo, nnp_convolution_transform_strategy_reuse, input_channels, output_channels,
            input_size, input_padding, kernel_size, stride_size,
            static_cast<float*>(input->data) +
                n * input->shape[1] * input->shape[2] * input->shape[3],
            static_cast<float*>(transformed_kernel->data),
            bias ? static_cast<float*>(bias->data) : zero_bias->data(),
            static_cast<float*>(output->data) +
                n * output->shape[1] * output->shape[2] * output->shape[3],
            workspace_buffer, &workspace_size, nnp_activation_identity, nullptr, entry->threadpool,
            nullptr);
        ICHECK_EQ(status, nnp_status_success);
      }

      cpu_api->FreeWorkspace(dev, workspace_buffer);
    });

TVM_REGISTER_GLOBAL("tvm.contrib.nnpack.convolution_inference_weight_transform")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      NNPackThreadLocalEntry* entry = NNPackThreadLocalEntry::ThreadLocal();
      static std::once_flag flag;
      std::call_once(flag, []() { ICHECK_EQ(nnp_initialize(), nnp_status_success); });
      DLTensor* kernel = args[0];
      DLTensor* transformed_kernel = args[1];
      // Dummy sizes
      nnp_padding input_padding{1, 1, 1, 1};
      nnp_size stride_size{1, 1};

      nnp_size input_size{100, 100};

      NNPackConfig(args[2]);

      uint64_t algo_ = args[3];
      nnp_convolution_algorithm algo = static_cast<nnp_convolution_algorithm>(algo_);
      ICHECK_EQ(kernel->ndim, 4);
      size_t input_channels = kernel->shape[1];
      size_t output_channels = kernel->shape[0];
      ICHECK_EQ(kernel->shape[2], 3);
      ICHECK_EQ(kernel->shape[3], 3);
      nnp_size kernel_size{static_cast<size_t>(kernel->shape[2]),
                           static_cast<size_t>(kernel->shape[3])};
      ICHECK(kernel->strides == nullptr);
      ICHECK(TypeMatch(kernel->dtype, kDLFloat, 32));

      size_t transformed_kernel_size = 0;
      nnp_status status;
      status = nnp_convolution_inference(
          algo, nnp_convolution_transform_strategy_precompute, input_channels, output_channels,
          input_size, input_padding, kernel_size, stride_size, nullptr, nullptr, nullptr, nullptr,
          nullptr, &transformed_kernel_size, nnp_activation_identity, nullptr, entry->threadpool,
          nullptr);
      ICHECK_EQ(status, nnp_status_success);

      ICHECK_LE(transformed_kernel_size, GetDataSize(*transformed_kernel));

      status = nnp_convolution_inference(
          algo, nnp_convolution_transform_strategy_precompute, input_channels, output_channels,
          input_size, input_padding, kernel_size, stride_size, nullptr,
          static_cast<float*>(kernel->data), nullptr, nullptr,
          static_cast<float*>(transformed_kernel->data), &transformed_kernel_size,
          nnp_activation_identity, nullptr, entry->threadpool, nullptr);
      ICHECK_EQ(status, nnp_status_success);
    });
}  // namespace contrib
}  // namespace tvm
